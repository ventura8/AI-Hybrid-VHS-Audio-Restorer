"""Music and ambience conservation, read on the non-vocal stem.

The same separator (the app's own BS-RoFormer through audio-separator) splits the
source and the output; whatever the restoration did to the background then shows on
the instrumental stem alone: SI-SDR and log-spectral distance against the source's
stem, the per-octave energy ratio (a stripped room tone is an octave that dropped),
and the correlation of the two envelopes (pumping, gating).
"""

from pathlib import Path

import numpy as np

from modules.utils import MODELS_DIR
from scripts.restoration_quality import audio_io
from scripts.score_reference import _lsd_db, _si_sdr_db

STEM_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# From 125 Hz: both engines run an 80 Hz rumble high-pass by design, so the 63-125 Hz octave
# of a bass-heavy source always reads as lost and says nothing about the music above it.
OCTAVES_HZ = (
    (125.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 16000.0),
)
ENVELOPE_S = 0.02
# The CRT line whistle (15625 Hz PAL, 15734 Hz NTSC) can carry most of a VHS capture's
# 8-16 kHz energy; removing it is restoration, not music lost, so the octave reading
# leaves these bands out on both sides.
CRT_LINE_HZ = (15625.0, 15734.0)
CRT_LINE_HALF_WIDTH_HZ = 250.0
SILENT_STEM_RMS = 1e-4
BACKGROUND_MIN_DBFS = -45.0
BACKGROUND_MIN_RATIO_DB = -20.0


def load_separator(device, models_dir, output_dir=None):
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
    band = (freqs >= low) & (freqs < high)
    for line in CRT_LINE_HZ:
        band &= np.abs(freqs - line) > CRT_LINE_HALF_WIDTH_HZ
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
    """Aligned, gain-matched mono instrumental stems of the pair."""
    stems_dir = pair.cache_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    separator = registry.get("separator", lambda device, models_dir: load_separator(device, models_dir, stems_dir))
    src_stem = separate_once(separator, pair.source_wav, stems_dir, pair.source_key)
    out_stem = separate_once(separator, pair.output_wav, stems_dir, pair.output_key)
    src_audio, rate = audio_io.load_audio(src_stem)
    out_audio, _rate = audio_io.load_audio(out_stem)
    src, out, _lag = audio_io.align_pair(audio_io.to_mono(src_audio), audio_io.to_mono(out_audio))
    return src, out, rate


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
    """Fills every row that carries a background with the four stem readings (paired: output side, zero source)."""
    src, out, rate = _stems(pair, registry)
    for row in card.rows:
        begin, end = int(round(row.start_s * rate)), int(round(row.end_s * rate))
        mix = pair.source[int(round(row.start_s * pair.rate)) : int(round(row.end_s * pair.rate))]
        row.output["stems.has_background"] = float(end <= len(src) and has_background(src[begin:end], mix))
        if not row.output["stems.has_background"]:
            continue
        for name, value in _window_readings(src[begin:end], out[begin:end], rate).items():
            row.source[name], row.output[name] = 0.0, value
