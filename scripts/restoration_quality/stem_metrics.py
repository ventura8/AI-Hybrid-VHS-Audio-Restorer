"""Music and ambience conservation, read on the non-vocal stem.

The same separator (the app's own BS-RoFormer through audio-separator) splits the
source and the output; whatever the restoration did to the background then shows on
the instrumental stem alone: SI-SDR and log-spectral distance against the source's
stem, the per-octave energy ratio (a stripped room tone is an octave that dropped),
and the correlation of the two envelopes (pumping, gating).

The per-octave energy ratio (`stems.octave_ratio_db`) is displayed only since 2026-10-12:
it read hiss removal and the least-squares stem match as stripping. The gate and the music
grid read `stems.octave_programme_db` (`stem_octave.py`), the mixes read on the cells where
the source's level-true background leads the programme.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from modules.utils import MODELS_DIR
from scripts.restoration_quality import audio_io, dsp_metrics, judges, stem_octave
from scripts.restoration_quality.runner import UNAVAILABLE_ERRORS, row_slice
from scripts.score_reference import _lsd_db, _si_sdr_db

STEM_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OCTAVES_HZ = stem_octave.OCTAVES_HZ
PROGRAMME_OCTAVE = "stems.octave_programme_db"
PEAK_BLOCK = 1 << 20
ENVELOPE_S = 0.02
# The CRT line whistle is left out of every band reading (`dsp_metrics.band_bins`).
CRT_LINE_HZ = dsp_metrics.CRT_LINE_HZ
CRT_LINE_HALF_WIDTH_HZ = dsp_metrics.CRT_LINE_HALF_WIDTH_HZ
SILENT_STEM_RMS = 1e-4
BACKGROUND_MIN_DBFS = -45.0
BACKGROUND_MIN_RATIO_DB = -20.0


def load_separator(models_dir, output_dir=None):
    """The app's separator, loaded once, writing the instrumental stem only."""
    import logging

    from audio_separator.separator import Separator

    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=str(models_dir or MODELS_DIR),
        output_dir=str(output_dir),
        output_format="WAV",
        output_single_stem="Instrumental",
        use_soundfile=True,
    )
    separator.load_model(model_filename=STEM_MODEL)
    return separator


def separate_once(separator, wav_path, stems_dir, key):
    """The instrumental stem of `wav_path`, separated once and kept under `stems_dir`."""
    target = Path(stems_dir) / f"{key}_instrumental.wav"
    if not target.exists():
        separator.separate(str(wav_path), custom_output_names={"Instrumental": f"{key}_instrumental"})
    if not target.exists():
        raise RuntimeError(f"separator wrote no instrumental stem for {wav_path}")
    return target


def octave_ratio_db(source, output, rate):
    """The worst octave: 10 log10 of output over source band energy, minimum over the octaves the source carries."""
    freqs = np.fft.rfftfreq(len(source), 1.0 / rate)
    src = np.abs(np.fft.rfft(np.asarray(source, dtype=np.float64))) ** 2
    out = np.abs(np.fft.rfft(np.asarray(output, dtype=np.float64))) ** 2
    ratios = [_octave(freqs, src, out, low, high) for low, high in OCTAVES_HZ]
    ratios = [r for r in ratios if r is not None]
    return float(min(ratios)) if ratios else None


def _octave(freqs, src, out, low, high):
    band = dsp_metrics.band_bins(freqs, (low, high))
    src_energy = src[band].sum()
    if src_energy <= 0.0 or src_energy < SILENT_STEM_RMS**2 * len(src):
        return None
    return float(10.0 * np.log10((out[band].sum() + 1e-20) / (src_energy + 1e-20)))


def envelope_corr(source, output, rate):
    """Pearson correlation of the two 20 ms RMS envelopes; gating or pumping of the background lowers it."""
    frame = max(1, int(ENVELOPE_S * rate))
    count = min(len(source), len(output)) // frame
    if count < 4:
        return None
    src = np.sqrt(np.mean(np.asarray(source[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1))
    out = np.sqrt(np.mean(np.asarray(output[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1))
    if src.std() < 1e-9 or out.std() < 1e-9:
        return None
    return float(np.corrcoef(src, out)[0, 1])


def _stems(pair, registry):
    """`{src, out, rate, level_true}`: aligned, gain-matched mono instrumental stems, and the source stem at its mix's level.

    `level_true` is None when the separator capped the source stem's peak (`stem_octave.stem_scale`).
    """
    stems_dir = pair.cache_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    separator = registry.get("separator", lambda _device, models_dir: load_separator(models_dir, stems_dir))
    src_stem = separate_once(separator, pair.source_wav, stems_dir, pair.source_key)
    out_stem = separate_once(separator, pair.output_wav, stems_dir, pair.output_key)
    src_audio, rate = audio_io.load_audio(src_stem)
    out_audio, _rate = audio_io.load_audio(out_stem)
    # The pair's own aligner and prior, so a stem pair the aligner abstains on is trimmed where the pair was.
    method, prior_s = getattr(pair, "align_method", None), getattr(pair, "align_prior_s", None)
    src_mono = audio_io.to_mono(src_audio)
    src, out, _lag = audio_io.align_pair(src_mono, audio_io.to_mono(out_audio), rate, method, prior_s)
    scale = stem_octave.stem_scale(file_peak(pair.source_wav), file_peak(src_stem))
    return {"src": src, "out": out, "rate": rate, "level_true": None if scale is None else src_mono * np.float32(scale)}


def file_peak(path):
    """The largest absolute sample of a sound file over every channel, read in blocks (an hour of tape is 1.3 GB)."""
    return max(
        (float(np.abs(block).max()) for block in sf.blocks(str(path), blocksize=PEAK_BLOCK, dtype="float32") if block.size), default=0.0
    )


def levelled(pair, stems):
    """`(stem, mix_source, mix_output)` at the stem's rate on the pair's timeline, or None without a level-true stem.

    The source stem shares the source file's timeline, so it takes the pair's own lag; the
    output mix keeps its least-squares gain, which the reading's level match cancels.
    """
    stem, rate = stems["level_true"], stems["rate"]
    if stem is None:
        return None
    lag = int(round(pair.lag * rate / pair.rate))
    return (stem[-lag:] if lag < 0 else stem), pair.at("source", rate), pair.at("output", rate)


def _window_readings(src, out, rate):
    return {
        "stems.si_sdr_db": _si_sdr_db(src.astype(np.float64), out.astype(np.float64)),
        "stems.lsd_db": _lsd_db(src.astype(np.float64), out.astype(np.float64)),
        "stems.octave_ratio_db": octave_ratio_db(src, out, rate),
        "stems.envelope_corr": envelope_corr(src, out, rate),
    }


def has_background(stem, mix):
    """Whether the source's stem carries a background worth conserving in this window.

    An interview with no music separates into a stem that is only residual room and hiss,
    20 dB under the mix; conservation is meaningless there and would read the denoiser's
    own work as stripping. The stem must sit above -45 dBFS and within 20 dB of the mix.
    """
    stem_rms = float(np.sqrt(np.mean(np.asarray(stem, dtype=np.float64) ** 2)) + 1e-12)
    mix_rms = float(np.sqrt(np.mean(np.asarray(mix, dtype=np.float64) ** 2)) + 1e-12)
    return 20.0 * np.log10(stem_rms) >= BACKGROUND_MIN_DBFS and 20.0 * np.log10(stem_rms / mix_rms) >= BACKGROUND_MIN_RATIO_DB


def score(pair, card, registry):
    """Fills every row that carries a background with the stem readings (paired: output side, zero source)."""
    stems = _stems(pair, registry)
    src, out, rate = stems["src"], stems["out"], stems["rate"]
    mixes = levelled(pair, stems)
    for row in card.rows:
        begin, end = int(round(row.start_s * rate)), int(round(row.end_s * rate))
        mix = pair.source[row_slice(pair, row)]
        row.output["stems.has_background"] = float(end <= len(src) and has_background(src[begin:end], mix))
        if not row.output["stems.has_background"]:
            continue
        readings = {**_window_readings(src[begin:end], out[begin:end], rate), PROGRAMME_OCTAVE: programme_octave(mixes, begin, end, rate)}
        for name, value in readings.items():
            row.source[name], row.output[name] = 0.0, value
    _mert(pair, card, registry)


def programme_octave(mixes, begin, end, rate):
    """`stem_octave.octave_programme_db` on one window of `levelled`'s arrays; None without them."""
    if mixes is None:
        return None
    stem, mix_source, mix_output = mixes
    return stem_octave.octave_programme_db(stem[begin:end], mix_source[begin:end], mix_output[begin:end], rate)


def _mert(pair, card, registry):
    """The MERT distance on the full mix, after the separator is released; a missing model costs only this reading."""
    registry.release()
    try:
        judges.score_mert(pair, card, registry)
    except UNAVAILABLE_ERRORS as exc:
        print(f"{judges.MERT_KEY} unavailable: {type(exc).__name__}: {exc}")
