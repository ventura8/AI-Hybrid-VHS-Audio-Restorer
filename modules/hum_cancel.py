"""Harmonic hum cancellation for the full-mix chain, by tracked sinusoidal subtraction.

Neither released mode removes mains hum on real tape. cathar's dehum runs at whatever
frequency the shared scanner reports and the scanner misses nearly half the tapes that
carry hum; run at the right frequency it takes a median 2.07 dB of harmonic excess from a
median 9.2, at 0.25 dB of movement in the speech that shares the band, and inside the
`auto_pure_linear` chain that collapsed to 0.74 dB at 1.88 dB of movement, because a notch
cuts a band whether or not hum is in it at that moment.

This estimates the hum itself and subtracts it. Each harmonic is complex-demodulated on an
analysis grid, its slowly varying amplitude and phase smoothed with a bandwidth of a few
hertz -- real mains is stable to well under half a hertz, and what wanders is the transport's
wow, which scales with the harmonic number -- shrunk against the harmonic's own spectral
neighbourhood so a harmonic that is not there is not invented, capped so a voiced harmonic
parked on a mains line for a moment cannot be taken with it, and resynthesised and
subtracted. A harmonic is cancelled only where it stands above its neighbourhood, so the
same stage takes an EMI buzz running to dozens of harmonics and leaves alone a harmonic the
pre-conditioning notch already flattened. The estimate is per channel: linear-track hum
differs in amplitude and phase between channels, and a shared estimate would leave a
full-level residual in the channel it did not fit.

It runs ahead of the noise probe, so the profile the subtraction learns from the quietest
stretch no longer carries the hum and the subtraction spends its floor on hiss instead --
the mechanism behind the in-chain collapse of dehum. Two streaming passes, so a two-hour
capture is never held in memory; what is held is the baseband, a few hundred megabytes at
most at forty harmonics.

Detection stays on the mode's own reader in `spectral_denoise`, which reads the whole
recording and chooses 50 or 60 Hz by the evidence of eight harmonics. Anti-phase hum
cancels in that mono downmix and is missed; it is rare and it is documented.
"""

from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

from .config import APL_ENABLE_HUM_CANCEL, APL_HUM_BANDWIDTH_HZ, APL_HUM_MAX_HARMONICS, APL_HUM_SKIP_NOTCHED, APL_TONAL_FLATNESS_MAX
from .spectral_denoise import (
    MAINS_HARMONICS,
    _scannable_mono,
    detect_mains_hz,
    estimate_tonality,
    harmonic_triples,
    quiet_psd,
)
from .utils import log_msg

# Analysis grid: a 4096-sample Hann window every 1024 samples gives a 43 Hz frame rate,
# whose baseband Nyquist of 21.5 Hz is the window's main-lobe half-width, so what folds
# into the few-hertz envelope band is 35 dB down or more.
ANALYSIS_WINDOW = 4096
ANALYSIS_HOP = 1024
# Samples per streaming block, both passes.
BLOCK_SAMPLES = 1 << 20
# Envelope bandwidth grows with harmonic number: recorded hum carries the transport's wow,
# and a 0.2 % excursion at the fundamental is eight times that at the eighth harmonic.
BANDWIDTH_SLOPE_HZ = 0.5
BANDWIDTH_MAX_HZ = 5.0
# What a harmonic's envelope may rise to over its own recent level before the excess is
# read as programme sharing the line rather than hum, and the window that level is read
# over. A compander swell lifts hum faster than this follows; the residual is measured.
CAP_DB = 6.0
CAP_WINDOW_S = 2.0
# A harmonic is cancelled only where it stands this far above its neighbourhood; the bar
# is higher past the mains series proper, where a line is more likely to be programme.
HARMONIC_MIN_EXCESS_DB = 3.0
EXTENDED_MIN_EXCESS_DB = 6.0
# How far, in bins of the quiet-frame spectrum, a line may sit from its nominal harmonic
# and still be that harmonic: two bins is 5.4 Hz, which covers what the tapes showed.
LINE_TOLERANCE_BINS = 2
# Fundamental refinement: real mains sits within this of its nominal value, and a wider
# range would let a bass note pull the series off the hum. The search looks a little past
# the range so a line just outside it is still read and then held to the edge, rather than
# a noise bin inside the range standing in for it.
MAX_REFINE_HZ = 0.5
REFINE_SEARCH_FACTOR = 1.5
REFINE_SAMPLES = 1 << 20
# The running median steadies a line that stays put; a line tracked with a wide band is one
# that moves, and its baseband turns a good part of a cycle inside the median's span, where
# a median of the parts of a turning phasor is no estimate of it. Past this bandwidth the
# low-pass works alone.
MEDIAN_MAX_BANDWIDTH_HZ = 10.0
# Below this many frames the envelope filter has nothing to settle on.
MIN_FRAMES = 64
# The harmonics the shared pre-conditioning notches when the scanner reports mains hum.
PRECONDITIONED_HARMONICS = (1, 2)
TOP_MARGIN_HZ = 100.0
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def bandwidths_for(harmonics, base_hz=APL_HUM_BANDWIDTH_HZ):
    """The envelope bandwidth of each harmonic in Hz, widening with its number to a cap."""
    return [min(base_hz + BANDWIDTH_SLOPE_HZ * (harmonic - 1), BANDWIDTH_MAX_HZ) for harmonic in harmonics]


def _excess_db(peak, floor):
    """A harmonic's standing above its neighbourhood, in dB."""
    return float(10.0 * np.log10(peak / floor))


def _stands_out(harmonic, peak, floor):
    """Whether one harmonic clears the bar for cancellation."""
    bar = HARMONIC_MIN_EXCESS_DB if harmonic <= MAINS_HARMONICS else EXTENDED_MIN_EXCESS_DB
    return _excess_db(peak, floor) >= bar


def _line_near(freqs, psd, hz):
    """Where the spectrum's line nearest the given frequency sits, or None when the peak there belongs to a neighbour.

    Real hum harmonics are not exact multiples of one fundamental: on the tapes measured,
    lines sit one to eight hertz off the series, past any envelope band, so each is
    cancelled at the frequency it has. A programme partial a few hertz further off raises
    the line's bins through its own main lobe, and then the neighbourhood's maximum sits
    on the partial rather than near the line, which is the case refused here.
    """
    index = int(np.argmin(np.abs(freqs - hz)))
    around_lo, around_hi = max(index - 4, 0), index + 5
    local = around_lo + int(np.argmax(psd[around_lo:around_hi]))
    if abs(local - index) > LINE_TOLERANCE_BINS or local in (0, len(psd) - 1):
        return None
    before, after = local - 1, local + 2
    left, centre, right = np.log(psd[before:after] + 1e-30)
    denominator = left - 2.0 * centre + right
    shift = 0.0 if denominator >= 0.0 else 0.5 * (left - right) / denominator
    return float(freqs[local] + shift * (freqs[1] - freqs[0]))


def _gated_line(freqs, psd, triple, f0, ceiling_hz, skip):
    """(harmonic, line frequency, neighbourhood floor) when the harmonic stands out as a line of its own, else None."""
    harmonic, peak, floor = triple
    if harmonic in skip:
        return None
    line = _line_near(freqs, psd, harmonic * f0) if harmonic * f0 < ceiling_hz else None
    if line is not None and _stands_out(harmonic, peak, floor):
        return harmonic, line, floor
    return None


def gate_harmonics(freqs, psd, f0, count, ceiling_hz, skip=()):
    """The harmonics to cancel as (harmonic, line frequency, neighbourhood floor): those standing out as lines of their own."""
    lines = [_gated_line(freqs, psd, triple, f0, ceiling_hz, skip) for triple in harmonic_triples(freqs, psd, f0, count)]
    return [line for line in lines if line is not None]


def plan_harmonics(mono_signal, sample_rate, f0, count=APL_HUM_MAX_HARMONICS, skip=()):
    """The fundamental the recording carries and the harmonics to cancel at it, from the quiet stretches.

    Read over the quietest frames, not the whole recording: a sustained partial of the
    programme sitting a few hertz from a line would otherwise gate that line open and then
    be tracked as hum. Hum is there when the programme is not. The mains series proper
    (eight harmonics at the nominal frequency) refines the fundamental first, and the full
    series is then gated at the refined value, where a high harmonic actually sits.
    """
    freqs, psd = quiet_psd(mono_signal, sample_rate)
    if psd is None:
        return f0, []
    ceiling = sample_rate / 2.0 - TOP_MARGIN_HZ
    core = [harmonic for harmonic, _line, _floor in gate_harmonics(freqs, psd, f0, MAINS_HARMONICS, ceiling, skip)]
    refined = refine_f0(mono_signal, sample_rate, f0, core)
    if refined is None:
        return None, []
    return refined, gate_harmonics(freqs, psd, refined, count, ceiling, skip)


def _peak_offset_hz(spectrum, freqs, target_hz, search_hz):
    """Where the line nearest the target actually sits, refined between bins, as an offset from the target."""
    window = np.abs(freqs - target_hz) <= search_hz
    indices = np.flatnonzero(window)
    local = int(indices[np.argmax(spectrum[indices])])
    if local == 0 or local == len(spectrum) - 1:
        return freqs[local] - target_hz
    before, after = local - 1, local + 2
    left, centre, right = np.log(spectrum[before:after] + 1e-30)
    denominator = left - 2.0 * centre + right
    shift = 0.0 if denominator >= 0.0 else 0.5 * (left - right) / denominator
    return freqs[local] + shift * (freqs[1] - freqs[0]) - target_hz


def refine_f0(mono_signal, sample_rate, f0, harmonics):
    """The fundamental the harmonics agree on, or None when they place it outside the refinement range.

    A single long transform of the opening stretch: bins of 0.04 Hz, refined between bins
    by the peak's curvature, and the per-harmonic estimates -- each offset divided by its
    harmonic number -- combined as a median so one harmonic sitting under a bass note does
    not carry the vote. Mains is stable to well under half a hertz; a series whose lines
    agree on a fundamental further off than that is a chord, not hum -- a G major sits at
    98, 147 and 196 Hz and reads as the second, third and fourth harmonics of 49 Hz -- and
    it is refused rather than held to the edge of the range, which is where every such
    series used to land.
    """
    segment = mono_signal[:REFINE_SAMPLES]
    if len(segment) < MIN_FRAMES * ANALYSIS_HOP or not harmonics:
        return f0
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=REFINE_SAMPLES))
    freqs = np.fft.rfftfreq(REFINE_SAMPLES, 1.0 / sample_rate)
    search = MAX_REFINE_HZ * REFINE_SEARCH_FACTOR
    offsets = [_peak_offset_hz(spectrum, freqs, harmonic * f0, search * harmonic) / harmonic for harmonic in harmonics]
    offset = float(np.median(offsets))
    return None if abs(offset) > MAX_REFINE_HZ else f0 + offset


def _demod_kernel(freqs_hz, sample_rate, length=ANALYSIS_WINDOW):
    """The windowed demodulation kernel, (window x lines), normalised so a sinusoid of amplitude A reads A."""
    window = np.hanning(length)
    samples = np.arange(length)[:, None]
    phase = -2j * np.pi * samples * np.asarray(freqs_hz)[None, :] / sample_rate
    return (2.0 / window.sum()) * window[:, None] * np.exp(phase)


def _frame_starts(offset, available, hop, length):
    """Absolute start samples of every analysis frame that fits in the buffered stretch."""
    first = -(-offset // hop) * hop
    last = offset + available - length
    return np.arange(first, last + 1, hop) if last >= first else np.zeros(0, dtype=np.int64)


def _absolute_phase(freqs_hz, starts, sample_rate):
    """The rotation that refers each frame's baseband to time zero, computed modulo a cycle so it stays exact."""
    turns = np.mod(np.outer(starts.astype(np.float64), np.asarray(freqs_hz)), sample_rate) / sample_rate
    return np.exp(-2j * np.pi * turns)


def _demodulate(buffer, offset, kernel, freqs_hz, sample_rate, hop):
    """One buffered stretch's frames, demodulated: (frames x channels x lines), and the next buffer offset."""
    starts = _frame_starts(offset, len(buffer), hop, len(kernel))
    if not len(starts):
        return np.zeros((0, buffer.shape[1], kernel.shape[1]), dtype=np.complex64), offset
    frames = np.lib.stride_tricks.sliding_window_view(buffer, len(kernel), axis=0)[starts - offset]
    baseband = np.einsum("fcw,wl->fcl", frames, kernel) * _absolute_phase(freqs_hz, starts, sample_rate)[:, None, :]
    return baseband.astype(np.complex64), int(starts[-1] + hop)


def analyse(source_wav, freqs_hz, hop=ANALYSIS_HOP, window=ANALYSIS_WINDOW):
    """Every line's baseband across the recording, per channel: (frames x channels x lines), and the sample rate.

    Streamed a block at a time on an absolute frame grid, so the frames are the same
    whatever the block size. A finer hop raises the frame rate and with it the widest
    envelope band a line can be tracked with; a shorter window lets a line that moves
    inside a frame's span still read as one line.
    """
    with sf.SoundFile(str(source_wav)) as handle:
        kernel = _demod_kernel(freqs_hz, handle.samplerate, window)
        buffer, offset, pieces = np.zeros((0, handle.channels), dtype=np.float32), 0, []
        for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
            buffer = np.concatenate((buffer, block))
            baseband, next_offset = _demodulate(buffer, offset, kernel, freqs_hz, handle.samplerate, hop)
            pieces.append(baseband)
            consumed = next_offset - offset
            buffer, offset = buffer[consumed:], next_offset
        return np.concatenate(pieces) if pieces else np.zeros((0, handle.channels, len(freqs_hz)), dtype=np.complex64), handle.samplerate


def _median_over(values, frames):
    """A running median of a complex series, part by part, over an odd number of frames."""
    size = (max(int(frames), 1) | 1,) + (1,) * (values.ndim - 1)
    real = scipy.ndimage.median_filter(values.real, size=size, mode="nearest")
    imag = scipy.ndimage.median_filter(values.imag, size=size, mode="nearest")
    return real + 1j * imag


def smooth_envelope(baseband, bandwidths_hz, frame_rate):
    """Each line's baseband as a slowly varying envelope: a running median, then a zero-phase low-pass at its bandwidth.

    The median first, because a speech partial sweeping through a line spends a frame or
    two inside its band on every pass, and a low-pass alone would follow each pass; a
    median over the span the bandwidth allows ignores what is there for a few frames and
    keeps what is there throughout, which is the hum.
    """
    smoothed = np.empty_like(baseband)
    for line, bandwidth in enumerate(bandwidths_hz):
        steadied = baseband[..., line]
        if bandwidth < MEDIAN_MAX_BANDWIDTH_HZ:
            steadied = _median_over(steadied, frame_rate / bandwidth)
        sos = scipy.signal.butter(4, bandwidth, fs=frame_rate, output="sos")
        smoothed[..., line] = scipy.signal.sosfiltfilt(sos, steadied, axis=0)
    return smoothed


def shrink_to_floor(envelope, floors, bandwidths_hz):
    """Wiener shrinkage of each line's envelope against what its neighbourhood alone would put there.

    A sinusoid of amplitude A reads A; noise of density S inside a band of width B reads
    an expected power of 4 S B on the same scale. Where the tracked power is no more than
    that, the line is not there and the estimate goes to zero rather than subtracting
    noise shaped like hum.
    """
    expected = 4.0 * np.asarray(floors) * np.asarray(bandwidths_hz)
    power = np.abs(envelope) ** 2 + 1e-30
    return envelope * np.maximum(0.0, 1.0 - expected / power)


def cap_envelope(envelope, frame_rate):
    """Each line's envelope held to a bound over its own running level, so a passing programme harmonic is not taken."""
    size = max(3, int(round(CAP_WINDOW_S * frame_rate)) | 1)
    magnitude = np.abs(envelope)
    running = scipy.ndimage.median_filter(magnitude, size=(size,) + (1,) * (envelope.ndim - 1), mode="nearest")
    bound = running * 10.0 ** (CAP_DB / 20.0)
    return envelope * np.minimum(1.0, bound / (magnitude + 1e-30))


def _synthesise(envelope, freqs_hz, start, count, sample_rate, hop, window):
    """The lines' waveform over one block of samples, from the frame-rate envelope interpolated to sample rate."""
    times = (np.arange(len(envelope)) * hop + window / 2.0) / sample_rate
    samples = np.arange(start, start + count)
    hum = np.zeros((count, envelope.shape[1]), dtype=np.float64)
    for line, hz in enumerate(freqs_hz):
        carrier = np.exp(2j * np.pi * np.mod(samples * float(hz), sample_rate) / sample_rate)
        for channel in range(envelope.shape[1]):
            real = np.interp(samples / sample_rate, times, envelope[:, channel, line].real)
            imag = np.interp(samples / sample_rate, times, envelope[:, channel, line].imag)
            hum[:, channel] += np.real((real + 1j * imag) * carrier)
    return hum


def subtract(source_wav, target_wav, envelope, freqs_hz, hop=ANALYSIS_HOP, window=ANALYSIS_WINDOW):
    """Writes the source less the resynthesised lines, a block at a time."""
    with (
        sf.SoundFile(str(source_wav)) as handle,
        sf.SoundFile(str(target_wav), "w", samplerate=handle.samplerate, channels=handle.channels, subtype="FLOAT") as out,
    ):
        start = 0
        for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
            hum = _synthesise(envelope, freqs_hz, start, len(block), handle.samplerate, hop, window)
            out.write((block - hum).astype(np.float32))
            start += len(block)
    return target_wav


def cancel_lines(source_wav, target_wav, freqs_hz, bandwidths_hz, floors, hop=ANALYSIS_HOP, window=ANALYSIS_WINDOW):
    """Cancels the given spectral lines from a recording; returns the target, or None when too short to track."""
    baseband, sample_rate = analyse(source_wav, freqs_hz, hop, window)
    if len(baseband) < MIN_FRAMES:
        return None
    frame_rate = sample_rate / hop
    envelope = cap_envelope(shrink_to_floor(smooth_envelope(baseband, bandwidths_hz, frame_rate), floors, bandwidths_hz), frame_rate)
    return subtract(source_wav, target_wav, envelope, freqs_hz, hop, window)


def cancel_mains(source_wav, target_wav, f0, gated):
    """Cancels the gated harmonics, each at the line frequency it was found at.

    `f0` is the refined fundamental the series was gated against; the lines themselves may
    sit a few hertz off its multiples, and that is where they are cancelled.
    """
    del f0
    harmonics = [harmonic for harmonic, _line, _floor in gated]
    freqs = [line for _harmonic, line, _floor in gated]
    floors = [floor for _harmonic, _line, floor in gated]
    return cancel_lines(source_wav, target_wav, freqs, bandwidths_for(harmonics), floors)


def _series_length(source_wav):
    """How far up the series to look: the mains series proper on tonal material, the full count elsewhere.

    Sustained programme partials are what a tonal recording is made of, and its quiet
    frames are quiet only relatively; past the eighth harmonic a line there is as likely
    to be a note as a buzz, so the extended series is left to material with real pauses.
    """
    tonality = estimate_tonality(source_wav)
    if tonality is not None and tonality < APL_TONAL_FLATNESS_MAX:
        return MAINS_HARMONICS
    return APL_HUM_MAX_HARMONICS


def _scanner_notch_hz(strategy):
    """The mains frequency the shared scanner reported, 0 when it did not."""
    if not isinstance(strategy, dict):
        return 0.0
    value = strategy.get("profile", {}).get("notch_hz")
    if value is None:
        value = strategy.get("precondition_filters", {}).get("notch_hz", 0.0)
    return float(value or 0.0)


def notched_harmonics(strategy):
    """The harmonics the pre-conditioning has already notched on this recording, when they are left to it."""
    if APL_HUM_SKIP_NOTCHED and _scanner_notch_hz(strategy) > 0:
        return PRECONDITIONED_HARMONICS
    return ()


def _plan(source_wav, skip=()):
    """The refined fundamental and the gated harmonics, or a reason to skip."""
    f0 = detect_mains_hz(source_wav)
    if not f0:
        return None, "unreadable" if f0 is None else "no mains hum"
    mono_signal, sample_rate = _scannable_mono(source_wav)
    refined, gated = plan_harmonics(mono_signal, sample_rate, f0, _series_length(source_wav), skip)
    if refined is None:
        return None, "the lines do not agree on a mains fundamental"
    if not gated:
        return None, "no harmonic stands above its neighbourhood"
    return (refined, gated), None


def apply_when_needed(source_wav, audio_dir, strategy=None):
    """Cancels mains hum when the recording carries it; returns the new path, or the input untouched.

    The frequency is read from the recording, not the scanner, which misses nearly half
    the tapes that carry hum; the scanner's report says only which harmonics the
    pre-conditioning has already notched.
    """
    if not APL_ENABLE_HUM_CANCEL:
        return source_wav
    plan, reason = _plan(source_wav, notched_harmonics(strategy))
    if plan is None:
        log_msg(f"    [Hum Cancel] Skipped: {reason}.")
        return source_wav
    output_dir = Path(audio_dir) / "hum_cancel"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"humcancel_{Path(source_wav).name}"
    try:
        produced = cancel_mains(source_wav, target, *plan)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Hum Cancel] Skipped after failure: {exc}")
        return source_wav
    if produced is None:
        log_msg("    [Hum Cancel] Skipped: too short to track.")
        return source_wav
    log_msg(f"    [Hum Cancel] Cancelled {len(plan[1])} harmonics of {plan[0]:.2f} Hz.")
    return produced
