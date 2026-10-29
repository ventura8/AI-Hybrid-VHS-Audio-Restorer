"""The cuts a listening block plays: aligned, faded, loudness-matched on speech, and where to take them.

`scripts/listen_ab.py` plays 8-12 s of each stimulus. A cut is only a fair comparison when the
cuts differ in nothing but the restoration, so:

- every stimulus is decoded at 44.1 kHz (`ffmpeg` for video or another rate, `soundfile` for a
  44.1 kHz WAV), half a second wider than the cut on each side, into a temporary directory that
  is deleted once the cuts are in memory;
- each cut is aligned to the first stimulus by `auditory.estimate_lag` over the whole decoded
  segment (cut plus margins, within +-4096 samples) and multiplied by the polarity it returns.
  The lag alone is not enough: an inverted candidate mixed into the incumbent at x = 0.5 on the
  blend continuum would cancel, and the listener would hear an artefact neither output holds;
- every cut is brought to one speech-active level, `TARGET_ACTIVE_DBFS` (-23 dBFS) unless the
  session names another, read on speech-active frames (20 ms frames of the reference cut at or
  above its median level and -50 dBFS), not on integrated loudness: an output that deepens the
  pauses would otherwise play louder in its speech, and level is the first difference a listener
  hears. One level for every block makes the session's volume setting, and the threshold run's
  dBFS-to-sensation-level map (`listen_threshold`), hold for all of them. Until 2026-10-12 the
  cuts were matched to the first stimulus within +-20 dB and the block played at that
  stimulus's level: Session 0's sanity block held the shipped APL at -20 dB against the raw
  tape, and Session 1 Block 1 (APL first) would have held the tape at +20 dB where it needs
  +21.5. A stimulus whose level on those frames is under -50 dBFS (`audio_io.SILENCE_DBFS`: a
  start past the end of the file, a wrong or empty file) is reported as silent and one whose
  peak would pass -0.18 dBFS (`PEAK_MAX`) at the target as unlevelled; either refuses the
  block instead of playing it (`info["fits_at_dbfs"]` names the highest target every stimulus
  fits). 10 ms fades keep the cut edges from clicking. Measured 2026-10-12 on the Session 0
  and Session 1 stimuli (10 s cuts): the restorations' speech-active level is -13.8 to -19.0
  dBFS with peaks at -1.0 to -5.5 dBFS (crest 12.8-17.7 dB), the raw Tele7abc tape at 80 s
  -36.1 dBFS with its peak 20.0 dB above it, so -23 dBFS leaves that tape 2.9 dB of headroom
  and plays the restorations 4-9 dB quieter than Session 0 did at the same volume setting;
- the catch trials' stimuli (`catch_cuts`, scored apart, `listen_protocol`): the reference cut
  through a +6 dB high shelf at 7.5 kHz (the app's air-shelf corner) brought back to the
  target on the same frames, and, with a raw tape, half reference and half raw tape; the
  familiarisation's (`familiar_cuts`): the first stimulus and the difference to the second
  stimulus scaled up (`EXAGGERATION`), never part of a scored trial;
- `quietest_span` offers the page a loop over the quietest 2 s of the reference cut, where a
  pause or hiss question is answered;
- without a chosen start, the cut goes where the first two stimuli differ most audibly. Both are
  decoded whole as mono at 32 kHz (`PICK_RATE`, which keeps the NMR's bands up to 16 kHz) into a
  temporary directory, each WAV deleted as soon as it is read (a 30-minute tape is ~230 MB
  there, nothing stays in the temp directory). They are aligned by `auditory.estimate_lag`
  (lag and polarity), and every cut-long window (half-window hop) is scored by the share of its
  frames whose noise-to-mask ratio exceeds 0 dB (`auditory.frame_nmr` against the first
  stimulus, the playback level placed as `auditory.audibility` places it, on the whole file's
  loud frames). Only when no window has an audible frame does the pick fall back to the spectral
  proxy: the mean absolute third-octave level difference (100 Hz-16 kHz, 93 ms frames, frames
  above -70 dBFS so digital silence does not count). The proxy counts every frame down to -70
  dBFS, so on its own it drifts towards pause-level differences whatever the question.
  Measured 2026-10-09 on SOTI's air-shelf pair (v4_air, +1 dB against +2 dB, 434 s): decoding
  both and scoring 85 windows of 10 s took 1.3 s, the NMR pick lands at 220 s (45.9% of its
  frames audible, median window 16.2%), the proxy alone would have taken 215 s (40.8% there);
  the temp directory was empty afterwards.
"""

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from modules.utils import FFMPEG_BIN
from scripts import cli_paths
from scripts.restoration_quality import audio_io, auditory
from scripts.restoration_quality.dsp_metrics import frame_levels, framed_psd
from scripts.restoration_quality.listen_modes import CATCH_AIR, CATCH_RAW, FAMILIAR_A, FAMILIAR_B

RATE = audio_io.NATIVE_RATE
PICK_RATE = 32000
MARGIN_S = 0.5
FADE_S = 0.01
FRAME_S = 0.02
ACTIVE_PERCENTILE = 50.0
MIN_ACTIVE_FRAMES = 10
PEAK_MAX = 0.98
PEAK_MAX_DBFS = 20.0 * float(np.log10(PEAK_MAX))
TARGET_ACTIVE_DBFS = -23.0
LEVEL_FLOOR = 1e-12
LOOP_SECONDS = 2.0
AIR_CORNER_HZ = 7500.0
AIR_CATCH_DB = 6.0
RAW_CATCH_SHARE = 0.5
EXAGGERATION = {"abx": 2.0, "blend": 1.0}
PICK_FRAME = 4096
PICK_FLOOR_DBFS = -70.0
THIRD_OCTAVES_HZ = 100.0 * 2.0 ** (np.arange(0, 23) / 3.0)
TEMP_PREFIX = "ai_restore_listen_ab_"
DECODE_TIMEOUT_S = 1800


def wav_bytes(audio, rate=RATE):
    """`audio` as a 24-bit PCM WAV file in memory."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), rate, format="WAV", subtype="PCM_24")
    return buffer.getvalue()


def cut_sha256(cut):
    """The decoded cut's PCM hash, as `auditory.pcm_sha256` hashes audio (float32 samples, rate and shape)."""
    return auditory.pcm_sha256(cut, RATE)


def ffmpeg_command(path, start_s, seconds, target):
    """The ffmpeg call that decodes `seconds` of the first audio stream from `start_s`, at RATE, to a float WAV."""
    start = cli_paths.checked_number(start_s, "start", minimum=0.0)
    length = cli_paths.checked_number(seconds, "seconds", minimum=0.001)
    stream = ["-map", "0:a:0", "-vn", "-acodec", "pcm_f32le", "-ar", str(RATE)]
    return [FFMPEG_BIN, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path), *stream, str(target)]


def mono_command(path, target, rate=PICK_RATE):
    """The ffmpeg call that decodes the whole first audio stream as mono float at `rate`."""
    stream = ["-map", "0:a:0", "-vn", "-ac", "1", "-acodec", "pcm_f32le", "-ar", str(int(rate))]
    return [FFMPEG_BIN, "-y", "-v", "error", "-i", str(path), *stream, str(target)]


def _ffmpeg_segment(path, start_s, seconds, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / f"{audio_io.file_key(path)}_{start_s:.3f}_{seconds:.3f}.wav"
    if not target.exists():
        subprocess.run(ffmpeg_command(path, start_s, seconds, target), check=True, capture_output=True, timeout=600)
    audio, _rate = sf.read(str(target), dtype="float32", always_2d=True)
    return audio


def read_segment(path, start_s, seconds, work_dir):
    """`(frames, channels)` float32 at RATE, `seconds` from `start_s` (fewer at the end of the file); ffmpeg decodes into `work_dir`."""
    path = Path(path)
    if path.suffix.lower() == ".wav" and sf.info(str(path)).samplerate == RATE:
        begin, frames = int(round(start_s * RATE)), int(round(seconds * RATE))
        audio, _rate = sf.read(str(path), start=begin, frames=frames, dtype="float32", always_2d=True)
        return audio
    return _ffmpeg_segment(path, start_s, seconds, Path(work_dir))


def _stereo(audio):
    return audio[:, :2] if audio.shape[1] >= 2 else np.repeat(audio, 2, axis=1)


def _segments(paths, start_s, seconds, work_dir):
    """Every stimulus's segment as stereo; decoded into a temporary directory, deleted afterwards, when `work_dir` is None."""
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, ignore_cleanup_errors=True) as temp:
            return _segments(paths, start_s, seconds, temp)
    return {label: _stereo(read_segment(path, start_s, seconds, work_dir)) for label, path in paths.items()}


def window_at(audio, begin, length):
    """`length` frames of `audio` from `begin`, zero-padded where the file does not reach."""
    out = np.zeros((length, audio.shape[1]), dtype=np.float32)
    low, high = max(begin, 0), min(begin + length, len(audio))
    if high > low:
        out[slice(low - begin, high - begin)] = audio[low:high]
    return out


def _aligned_window(reference, audio, offset, length):
    """`audio`'s window matching `reference`'s at `offset`: shifted by the lag, multiplied by the polarity."""
    lag, polarity = auditory.estimate_lag(audio_io.to_mono(reference), audio_io.to_mono(audio), RATE)
    return window_at(audio, offset + lag, length) * np.float32(polarity)


def _level_db(mono):
    return 20.0 * np.log10(frame_levels(mono, int(FRAME_S * RATE)) + 1e-9)


def speech_active(level_db):
    """Frames at or above the median level and the silence floor; every frame when too few qualify."""
    mask = level_db >= max(float(np.percentile(level_db, ACTIVE_PERCENTILE)), audio_io.SILENCE_DBFS)
    return mask if mask.sum() >= MIN_ACTIVE_FRAMES else np.ones_like(mask)


def _active_rms(mono, active):
    levels = frame_levels(mono, int(FRAME_S * RATE))
    return float(np.sqrt(np.mean(levels[active] ** 2)))


def active_levels_db(cuts, reference):
    """Each cut's RMS level in dBFS, read on the reference cut's speech-active frames."""
    active = speech_active(_level_db(audio_io.to_mono(cuts[reference])))
    return {label: 20.0 * float(np.log10(max(_active_rms(audio_io.to_mono(cut), active), LEVEL_FLOOR))) for label, cut in cuts.items()}


def target_gains_db(levels_db, target_dbfs=TARGET_ACTIVE_DBFS):
    """Each cut's gain to `target_dbfs`; 0 dB for a silent cut, which refuses the block anyway."""
    return {label: 0.0 if level < audio_io.SILENCE_DBFS else target_dbfs - level for label, level in levels_db.items()}


def peak_dbfs(cut):
    """The cut's sample peak in dBFS."""
    return 20.0 * float(np.log10(max(float(np.max(np.abs(cut), initial=0.0)), LEVEL_FLOOR)))


def unlevelled_labels(peaks_db):
    """The cuts whose peak passes PEAK_MAX at the target level."""
    return sorted(label for label, peak in peaks_db.items() if peak > PEAK_MAX_DBFS)


def silent_labels(levels_db):
    """The cuts whose speech-active level is under the silence floor."""
    return sorted(label for label, level in levels_db.items() if level < audio_io.SILENCE_DBFS)


def _faded(cut):
    ramp = np.linspace(0.0, 1.0, int(FADE_S * RATE), dtype=np.float32)[:, None]
    out = cut.copy()
    head, tail = len(ramp), len(out) - len(ramp)
    out[:head] *= ramp
    out[tail:] *= ramp[::-1]
    return out


def finish_cuts(cuts, gains):
    """Cuts at their linear `gains`, faded in and out, as float32."""
    return {label: _faded(cut * np.float32(gains[label])).astype(np.float32) for label, cut in cuts.items()}


def prepare_cuts(paths, start_s, seconds, work_dir=None, target_dbfs=TARGET_ACTIVE_DBFS):
    """`(cuts, info)`: every stimulus cut at the first one's window, aligned, faded and brought to `target_dbfs`.

    `info` holds `target_dbfs`, `levels_dbfs` (speech-active level before the gain, on the first
    cut's frames), `gains_db`, `peaks_dbfs` (after the gain), `silent` (labels under the silence
    floor), `unlevelled` (labels whose peak passes PEAK_MAX at the target) and `fits_at_dbfs`,
    the highest target every stimulus fits at. Play neither a silent nor an unlevelled block.
    """
    cuts, reference = aligned_cuts(paths, start_s, seconds, work_dir)
    return levelled_cuts(cuts, reference, target_dbfs)


def aligned_cuts(paths, start_s, seconds, work_dir=None):
    """`(cuts, reference label)`: every stimulus cut at the first one's window and aligned to it (lag and polarity)."""
    lead = min(MARGIN_S, start_s)
    raw = _segments(paths, start_s - lead, seconds + lead + MARGIN_S, work_dir)
    reference = next(iter(raw))
    offset, length = int(round(lead * RATE)), int(round(seconds * RATE))
    return {label: _aligned_window(raw[reference], audio, offset, length) for label, audio in raw.items()}, reference


def levelled_cuts(cuts, reference, target_dbfs=TARGET_ACTIVE_DBFS):
    """`(cuts, info)` of `prepare_cuts`: the aligned cuts brought to `target_dbfs` on the reference's speech frames, faded."""
    levels = active_levels_db(cuts, reference)
    gains_db = target_gains_db(levels, target_dbfs)
    finished = finish_cuts(cuts, {label: 10.0 ** (gain / 20.0) for label, gain in gains_db.items()})
    peaks = {label: peak_dbfs(cut) for label, cut in finished.items()}
    info = {"target_dbfs": float(target_dbfs), "levels_dbfs": levels, "gains_db": gains_db, "peaks_dbfs": peaks}
    info.update(silent=silent_labels(levels), unlevelled=unlevelled_labels(peaks))
    info["fits_at_dbfs"] = float(target_dbfs) + PEAK_MAX_DBFS - max(peaks.values())
    return finished, info


def quietest_span(cut, seconds=LOOP_SECONDS):
    """`[start_s, end_s]` of the quietest `seconds` of `cut` (20 ms frames): where a pause or hiss loop plays."""
    power = frame_levels(audio_io.to_mono(cut), int(FRAME_S * RATE)) ** 2
    width = max(1, min(len(power), int(round(seconds / FRAME_S))))
    sums = np.convolve(power, np.ones(width), mode="valid")
    start = int(np.argmin(sums)) * FRAME_S
    return [round(start, 3), round(start + width * FRAME_S, 3)]


def high_shelf(audio, gain_db, corner_hz, rate=RATE):
    """`audio` through an RBJ high shelf (slope 1) of `gain_db` at `corner_hz`, along its first axis."""
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * corner_hz / rate
    cos, alpha = np.cos(w0), np.sin(w0) / 2.0 * np.sqrt(2.0)
    root = 2.0 * np.sqrt(amp) * alpha
    b = amp * np.array([(amp + 1) + (amp - 1) * cos + root, -2.0 * ((amp - 1) + (amp + 1) * cos), (amp + 1) + (amp - 1) * cos - root])
    a = np.array([(amp + 1) - (amp - 1) * cos + root, 2.0 * ((amp - 1) - (amp + 1) * cos), (amp + 1) - (amp - 1) * cos - root])
    return scipy.signal.lfilter(b / a[0], a / a[0], audio, axis=0).astype(np.float32)


def _peak(cut):
    return float(np.max(np.abs(cut), initial=0.0))


def _under_peak(cut):
    """`(cut, gain_db)`: `cut` scaled down just enough to keep its peak at PEAK_MAX (gain 0 when it fits)."""
    scale = min(1.0, PEAK_MAX / max(_peak(cut), LEVEL_FLOOR))
    return (cut * np.float32(scale)).astype(np.float32), 20.0 * float(np.log10(scale))


def catch_cuts(cuts, reference, raw_label=None):
    """`(catch stimuli, gains_db)`: the reference with +6 dB air brought back to its speech level, and half raw tape.

    The air catch is levelled on the reference's speech-active frames, so it differs in timbre and
    not in level, then kept under PEAK_MAX; the raw-tape catch (only with `raw_label`) mixes two
    levelled cuts, whose peak cannot pass the larger of theirs.
    """
    plain = cuts[reference]
    shelved = high_shelf(plain, AIR_CATCH_DB, AIR_CORNER_HZ)
    levels = active_levels_db({reference: plain, CATCH_AIR: shelved}, reference)
    back_db = levels[reference] - levels[CATCH_AIR]
    air, held_db = _under_peak(shelved * np.float32(10.0 ** (back_db / 20.0)))
    stimuli, gains = {CATCH_AIR: air}, {CATCH_AIR: back_db + held_db}
    if raw_label is not None:
        stimuli[CATCH_RAW] = ((1.0 - RAW_CATCH_SHARE) * plain + RAW_CATCH_SHARE * cuts[raw_label]).astype(np.float32)
        gains[CATCH_RAW] = 0.0
    return stimuli, gains


def familiar_cuts(first, second, factor):
    """The familiarisation pair: `first` and `first + factor * (second - first)`, under one gain that keeps both under PEAK_MAX."""
    exaggerated = first + np.float32(factor) * (second - first)
    scale = np.float32(min(1.0, PEAK_MAX / max(_peak(exaggerated), _peak(first), LEVEL_FLOOR)))
    return {FAMILIAR_A: (first * scale).astype(np.float32), FAMILIAR_B: (exaggerated * scale).astype(np.float32)}


def _band_matrix(freqs):
    low, high = THIRD_OCTAVES_HZ[:-1], THIRD_OCTAVES_HZ[1:]
    return ((freqs[None, :] >= low[:, None]) & (freqs[None, :] < high[:, None])).astype(np.float64)


def disagreement_db(first, second, rate=PICK_RATE):
    """Mean absolute third-octave level difference, in dB, on the frames where `first` is above -70 dBFS."""
    freqs, power_a, level = framed_psd(first, rate, PICK_FRAME)
    _freqs, power_b, _level = framed_psd(second, rate, PICK_FRAME)
    bands = _band_matrix(freqs)
    keep = 20.0 * np.log10(level + 1e-12) > PICK_FLOOR_DBFS
    if not keep.any():
        return 0.0
    diff = 10.0 * np.log10((power_a[keep] @ bands.T + 1e-20) / (power_b[keep] @ bands.T + 1e-20))
    return float(np.mean(np.abs(diff)))


def window_spans(n_samples, seconds, rate=PICK_RATE):
    """`seconds`-long windows over `n_samples` on a half-window hop (one window when the signal is shorter)."""
    length = int(seconds * rate)
    return [slice(begin, begin + length) for begin in range(0, max(1, n_samples - length + 1), max(1, length // 2))]


def audible_shares(reference, candidate, spans, rate=PICK_RATE):
    """Per span, the share of frames where the difference exceeds the reference's masked threshold (NMR > 0 dB)."""
    ref, cand = np.asarray(reference)[:, None], np.asarray(candidate)[:, None]
    spl_offset_db = auditory.PLAYBACK_SPL_LOUD_FRAMES - 10.0 * np.log10(auditory.loud_frame_power(ref))
    return [_audible_share(ref[span], cand[span], rate, spl_offset_db) for span in spans]


def _audible_share(reference, candidate, rate, spl_offset_db):
    ref = reference.astype(np.float64)
    nmr = auditory.frame_nmr(ref, candidate.astype(np.float64) - ref, rate, spl_offset_db)
    return float(np.mean(nmr > auditory.AUDIBLE_NMR_DB))


def window_scores(reference, candidate, seconds, rate=PICK_RATE):
    """`(spans, scores)` over aligned signals: the audible shares, or the spectral proxy when no window has an audible frame."""
    spans = window_spans(len(reference), seconds, rate)
    shares = audible_shares(reference, candidate, spans, rate)
    if max(shares) > 0.0:
        return spans, shares
    return spans, [disagreement_db(reference[span], candidate[span], rate) for span in spans]


def best_window(first, second, seconds, rate=PICK_RATE):
    """The start, in seconds of `first`, of the `seconds` window where `second` differs from it most audibly."""
    lag, polarity = auditory.estimate_lag(first, second, rate)
    reference, candidate = auditory.null_aligned(np.asarray(first), np.asarray(second), lag, polarity)
    spans, scores = window_scores(reference, candidate, seconds, rate)
    return (spans[int(np.argmax(scores))].start + max(0, -lag)) / float(rate)


def decode_mono(path, work_dir, rate=PICK_RATE):
    """The whole first audio stream of `path` as mono float32 at `rate`; the decoded WAV is deleted once read."""
    target = Path(work_dir) / f"{audio_io.file_key(path)}_mono_{int(rate)}.wav"
    subprocess.run(mono_command(path, target, rate), check=True, capture_output=True, timeout=DECODE_TIMEOUT_S)
    audio, _rate = sf.read(str(target), dtype="float32")
    target.unlink()
    return audio_io.to_mono(audio)


def pick_start(first_path, second_path, seconds):
    """Where to cut when no start is given: the window where the first two stimuli differ most audibly."""
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, ignore_cleanup_errors=True) as temp:
        first, second = decode_mono(first_path, temp), decode_mono(second_path, temp)
    return best_window(first, second, seconds, PICK_RATE)
