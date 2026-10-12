"""The pre-conditioning's azimuth skew read on the loudest spans (`azimuth_estimate_method: spans`).

The default reading (`head`, `filters._head_azimuth_skew`) cross-correlates the first 32768
samples (0.74 s at 44.1 kHz) and takes the integer peak, with the sign the graph does not
correct: a right channel one sample late reads -0.02 ms and the graph delays the right
channel one more. The head of a tape is often a lead-in, a jingle or silence, and the plain
cross-correlation follows the bass. On the IA hunt (`tune_ia_hunt_v3`, 2026-10-12, 232
stereo clips) it delayed a channel on 43, and the mono sum's 4-8 kHz fell on 40 of them
(median -1.43 dB against the same graph without the delay). The clips the octave
investigation took for lag-0 sources read lag 0 only over the full band, where a lag-0
component above the programme band holds most bins; in 150-7000 Hz they carry 0.6-1.3
samples, which the head doubled. cathar's own GCC-PHAT azimuth stage re-aligns the channels
after the pre-conditioning; `auto_pure_linear` runs that stage only when the scan reads
0.05 ms or more, so a one-sample head reading stays in its output.

This reading takes the lag where the programme is, and only when the programme agrees:

- the spans: up to seven of 1.5 s, the loudest non-overlapping ones of the scanned audio,
  by the energy of both channels (an inverted pair would cancel in their sum);
- on each span, GCC-PHAT between the Hann-windowed channels: the cross-spectrum's phase
  alone, over 150-7000 Hz and only in the bins where both channels carry power within 30 dB
  of their in-band median, so neither the bass's phase nor a band one side lacks pulls the
  peak; the peak of its magnitude within the azimuth window, refined to a fraction of a
  sample by a parabola through the peak and its two neighbours; its height normalised to
  the phase coherence (1 for a pure delay, near 0 for unrelated channels);
- a span counts when its coherence reaches `MIN_COHERENCE`;
- the vote: the lag the most counted spans agree on within `AGREE_SAMPLES`, the median of
  those; at least two must agree (one when the audio holds room for one span only) and more
  than half of the counted ones, so a tie, such as a print stitched from two parts skewed
  opposite ways, abstains rather than doubling one part's skew;
- otherwise the reading abstains, and the caller applies no delay.

The lag is how far the right channel runs behind the left, in samples (positive: the right
channel is late), which is the sign the pre-conditioning graph corrects: a positive delay
in milliseconds delays the left channel (`adelay=NS|0`).

Nothing here is shared with `scripts/pair_alignment.py`, which reads a source against its
restored output on the same principle; modules never import scripts.
"""

from dataclasses import dataclass

import numpy as np
import scipy.fft

SPAN_S = 1.5
SPANS = 7
BAND_HZ = (150.0, 7000.0)
BIN_FLOOR = 1e-3
MIN_BINS = 64
MIN_COHERENCE = 0.3
AGREE_SAMPLES = 0.5
MIN_AGREE = 2
MIN_SPAN_SAMPLES = 2048
# The window is searched two samples wider than it is accepted, so a peak at its edge is seen
# as one outside it rather than taken for the edge.
SEARCH_MARGIN = 2
# The graph delays a channel by whole samples, so a lag under half a sample is no delay.
MIN_DELAY_SAMPLES = 0.5


@dataclass(frozen=True)
class SkewReading:
    """A `spans` reading: the lag in samples (None when it abstained), the vote behind it and why it abstained."""

    lag: float | None
    spans: int = 0
    agree: int = 0
    coherence: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class SpanPeak:
    """One span's GCC-PHAT peak: its lag in samples (a fraction of one included) and its signed phase coherence."""

    lag: float
    coherence: float


def span_length(rate, total):
    """The span the vote reads, in samples at `rate`, never longer than the audio."""
    return min(int(round(SPAN_S * rate)), total)


def span_energies(energy, length):
    """`(hop, energies)`: the energy of every span of `length` starting on a half-span grid of `energy` (per-sample power)."""
    hop = max(1, length // 2)
    blocks = len(energy) // hop
    per_block = energy[: blocks * hop].reshape(blocks, hop).sum(axis=1)
    return hop, per_block[:-1] + per_block[1:]


def span_starts(left, right, length, count=SPANS):
    """Starts of up to `count` loudest non-overlapping spans of `length`, by both channels' energy; silent spans are skipped."""
    energy = np.square(left, dtype=np.float64) + np.square(right, dtype=np.float64)
    hop, energies = span_energies(energy, length)
    chosen = []
    for index in np.argsort(-energies, kind="stable")[: int(np.count_nonzero(energies > 0.0))]:
        _keep_apart(chosen, int(index) * hop, length)
        if len(chosen) == count:
            break
    return sorted(chosen)


def _keep_apart(chosen, start, length):
    """Appends `start` to `chosen` unless a span already chosen overlaps the one it starts."""
    if all(abs(start - other) >= length for other in chosen):
        chosen.append(start)


def band_mask(spectrum_l, spectrum_r, rate, nfft):
    """The bins GCC-PHAT reads: inside the band, and carrying power within 30 dB of each channel's in-band median."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
    band = (freqs >= BAND_HZ[0]) & (freqs <= min(BAND_HZ[1], 0.45 * rate))
    mask = band.copy()
    if not band.any():
        return mask
    for spectrum in (spectrum_l, spectrum_r):
        power = np.abs(spectrum) ** 2
        mask &= power > float(np.median(power[band])) * BIN_FLOOR
    return mask


def phase_coherence(cross, mask, nfft, reach):
    """`(lags, coherence)` over +-`reach` samples: the cross-spectrum's phase in the masked bins, transformed back; 1 for a pure delay."""
    gcc = np.fft.irfft(np.where(mask, cross / (np.abs(cross) + 1e-30), 0.0), nfft)
    values = np.concatenate((gcc[-reach:], gcc[: reach + 1])) * nfft / (2.0 * mask.sum())
    return np.arange(-reach, reach + 1), values


def parabolic_offset(before, peak, after):
    """The fraction of a sample (within +-0.5) the vertex of the parabola through three heights sits from the middle one."""
    curvature = before - 2.0 * peak + after
    if curvature >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))


def refined_lag(lags, heights, index):
    """The lag of the peak at `index`, refined by its neighbours; a peak on the searched window's edge stays whole."""
    if index in (0, len(heights) - 1):
        return float(lags[index])
    return float(lags[index]) + parabolic_offset(heights[index - 1], heights[index], heights[index + 1])


def span_peak(left_span, right_span, rate, reach):
    """The GCC-PHAT peak of one span (the right channel's lag behind the left), or None when the band holds too few live bins."""
    size = len(left_span)
    nfft = scipy.fft.next_fast_len(2 * size, real=True)
    window = np.hanning(size)
    spectrum_l = np.fft.rfft(np.asarray(left_span, dtype=np.float64) * window, nfft)
    spectrum_r = np.fft.rfft(np.asarray(right_span, dtype=np.float64) * window, nfft)
    mask = band_mask(spectrum_l, spectrum_r, rate, nfft)
    if int(mask.sum()) < MIN_BINS:
        return None
    lags, values = phase_coherence(spectrum_r * np.conj(spectrum_l), mask, nfft, min(reach, size - 1))
    heights = np.abs(values)
    index = int(np.argmax(heights))
    return SpanPeak(refined_lag(lags, heights, index), float(values[index]))


def span_peaks(left, right, rate, max_lag):
    """`(peaks read, spans tried)`: the GCC-PHAT peak of every loudest span that had live bins."""
    length = span_length(rate, len(left))
    starts = span_starts(left, right, length)
    reach = max_lag + SEARCH_MARGIN
    peaks = (span_peak(left[start:][:length], right[start:][:length], rate, reach) for start in starts)
    return [peak for peak in peaks if peak is not None], len(starts)


def counted_peaks(peaks):
    """The peaks whose phase coherence reaches `MIN_COHERENCE`."""
    return [peak for peak in peaks if abs(peak.coherence) >= MIN_COHERENCE]


def agreeing_peaks(peaks):
    """The spans agreeing with the lag most counted spans agree on (ties: the smaller shift)."""
    lags = np.array([peak.lag for peak in peaks])
    support = (np.abs(lags[:, None] - lags[None, :]) <= AGREE_SAMPLES).sum(axis=1)
    finalists = lags[support == support.max()]
    centre = float(finalists[np.argmin(np.abs(finalists))])
    return [peak for peak, lag in zip(peaks, lags) if abs(lag - centre) <= AGREE_SAMPLES]


def needed_agreement(counted, spans):
    """How many counted spans must agree: more than half of them (a tie disagrees), and two unless the audio held one span only."""
    return max(MIN_AGREE if spans > 1 else 1, counted // 2 + 1)


def _why_none_counted(read):
    """Why no span counted: no span had live bins (no programme), or none cleared the coherence floor."""
    return "weak coherence" if read else "no programme"


def _decided(agreeing, spans):
    """The reading of a vote that carried: the agreeing spans' median lag and median coherence."""
    lag = float(np.median([peak.lag for peak in agreeing]))
    coherence = float(np.median([abs(peak.coherence) for peak in agreeing]))
    return SkewReading(lag, spans, len(agreeing), round(coherence, 4))


def _usable_audio(left, right):
    """Why the channels cannot be read (too short, or holding a NaN or an infinity), or an empty string."""
    if min(len(left), len(right)) < MIN_SPAN_SAMPLES:
        return "too short"
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return "non-finite samples"
    return ""


def read_skew(left, right, rate, max_lag):
    """The right channel's lag behind the left, voted by the loudest spans' GCC-PHAT peaks, or an abstention that says why."""
    refusal = _usable_audio(left, right)
    if refusal:
        return SkewReading(None, reason=refusal)
    read, spans = span_peaks(left, right, rate, max_lag)
    counted = counted_peaks(read)
    if not counted:
        return SkewReading(None, len(read), reason=_why_none_counted(read))
    agreeing = agreeing_peaks(counted)
    if len(agreeing) < needed_agreement(len(counted), spans):
        return SkewReading(None, len(read), len(agreeing), reason="spans disagree")
    return _decided(agreeing, len(read))


def delay_ms(lag, rate, max_lag):
    """A voted lag in samples as the graph's delay in ms (to 0.001 ms); 0.0 when abstained, under half a sample or outside the window."""
    if lag is None or not MIN_DELAY_SAMPLES <= abs(lag) <= max_lag:
        return 0.0
    return round(float(lag) / float(rate) * 1000.0, 3)


def describe(reading, delay):
    """One log line's worth of a reading: the delay it gives and the vote behind it, or why it abstained."""
    if reading.lag is None:
        return f"no delay ({reading.reason}; {reading.agree} of {reading.spans} spans agree)"
    vote = f"{reading.agree} of {reading.spans} spans agree on {reading.lag:+.2f} samples, coherence {reading.coherence:.2f}"
    return f"delay {delay:+.3f} ms ({vote})" if delay else f"no delay ({vote})"
