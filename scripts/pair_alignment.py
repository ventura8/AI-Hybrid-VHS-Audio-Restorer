"""The lag between a source and its restored output, searched over the whole file.

Every paired reading of the harness (`restoration_quality.audio_io.align_pair`, the trade
metric, the hum reading, the reference and defect-repair scorers, the listening cuts)
trims the pair to one timeline first. The aligner they shared until 2026-10-11 (kept as
`xcorr_head`) cross-correlated the first 81920 samples (1.86 s at 44.1 kHz) and took the
peak of |corr|. The lag an output really carries is the mux limiter's 219 samples
(4.97 ms) plus whatever shift the app's sync stage applied (its run log prints it). On
the 205 shift-synced clips of the IA hunt (`tune_ia_hunt_v3`, 2026-10-11) the head search
read that lag within 8 samples on 137 cathar outputs and 199 APL ones, this aligner on
203 and 202 (median error 0, p95 1, max 4 samples; the rest abstain); on the four Tata
tapes it read all eight shipped outputs exactly, the head search missed five by 6-181
samples. A quiet or silent head gives the head search nothing to correlate, its waveform
correlation follows the bass, whose phase cathar's 80 Hz dewind turns, and |corr| hides
an inverted output.

`gcc_phat_spans`, the default, reads the lag on several of the source's loudest spans,
wherever they sit in the file:

- the spans: up to seven of 1.5 s, the loudest non-overlapping ones of the source;
- on each span, GCC-PHAT between the Hann-windowed source and output spans at the same
  place: the cross-spectrum's phase alone, over 150-7000 Hz and only in the bins where
  both sides carry power within 30 dB of their in-band median, so neither the bass's
  phase nor a band one side lacks pulls the peak; the peak within +-4096 samples, its
  height normalised to the phase coherence (1 for a pure delay, about 0.02 at most for
  unrelated noise), its sign the polarity;
- a span counts when its coherence reaches 0.1;
- the vote: the lag the most counted spans agree on within 2 samples, the median of
  those; at least two must agree (one when the pair is too short for two spans) and at
  least half of the counted ones;
- otherwise the aligner abstains: the lag is unknown (None), the pair is trimmed at the
  prior when one is given and at 0 otherwise, and the result says why.

An optional prior (a lag in samples, such as the mux's 219 at 44.1 kHz) breaks ties
only: between peaks within 10 % of a span's highest, and between equally supported lags.
Without one, ties go to the smaller shift.

`xcorr_head` reproduces the stored scores: `AI_RESTORE_ALIGN_METHOD=xcorr_head`, or
`method="xcorr_head"`, selects it everywhere. Every result names the method that made it.
"""

import os
from dataclasses import asdict, dataclass

import numpy as np
import scipy.fft
import scipy.signal

ALIGN_GCC = "gcc_phat_spans"
ALIGN_LEGACY = "xcorr_head"
ALIGN_METHODS = (ALIGN_GCC, ALIGN_LEGACY)
DEFAULT_METHOD = ALIGN_GCC
METHOD_ENV = "AI_RESTORE_ALIGN_METHOD"

STATUS_OK = "ok"
STATUS_ABSTAINED = "abstained"
STATUS_LEGACY = "legacy"

MAX_LAG = 4096
# The span `xcorr_head` correlates: 10 * 1024 * 8 samples, 1.86 s at 44.1 kHz.
LEGACY_SPAN = 81920
NATIVE_RATE = 44100
# The app's mux delays every output by its limiter's lookahead: 219 samples at 44.1 kHz.
MUX_DELAY_S = 219.0 / NATIVE_RATE

SPAN_S = 1.5
SPANS = 7
BAND_HZ = (150.0, 7000.0)
BIN_FLOOR = 1e-3
MIN_BINS = 64
MIN_COHERENCE = 0.1
AGREE_SAMPLES = 2
MIN_AGREE = 2
TIE_RATIO = 0.9
MIN_SPAN_SAMPLES = 2048


@dataclass(frozen=True)
class Vote:
    """The evidence behind a `gcc_phat_spans` reading: spans read, spans agreeing, their median coherence."""

    spans: int = 0
    agree: int = 0
    coherence: float | None = None


@dataclass(frozen=True)
class Alignment:
    """How a pair was put on one timeline.

    `lag` is the output's lag behind the source in samples (positive: the output runs late),
    None when the aligner abstained; `applied_lag` is the lag the pair was trimmed at;
    `polarity` is +1, -1 (the output is inverted) or None when unknown.
    """

    method: str
    lag: int | None
    applied_lag: int
    polarity: int | None
    status: str
    reason: str = ""
    vote: Vote = Vote()

    def as_dict(self):
        """The JSON-ready record a scored document carries (the vote's fields at the top level)."""
        record = asdict(self)
        record.update(record.pop("vote"))
        record["lag_samples"] = record.pop("lag")
        record["applied_lag_samples"] = record.pop("applied_lag")
        return record


@dataclass(frozen=True)
class SpanPeak:
    """One span's GCC-PHAT peak: its lag and its signed phase coherence."""

    lag: int
    coherence: float


def resolve_method(method=None):
    """`method`, else `AI_RESTORE_ALIGN_METHOD`, else the default; an unknown name is a ValueError."""
    chosen = method or os.environ.get(METHOD_ENV) or DEFAULT_METHOD
    if chosen not in ALIGN_METHODS:
        raise ValueError(f"unknown alignment method {chosen!r}; expected one of {', '.join(ALIGN_METHODS)}")
    return chosen


def trim(reference, output, lag):
    """Both signals on one timeline at `lag` (positive: the output is late), cut to their common length."""
    if lag > 0:
        output = output[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    length = min(len(reference), len(output))
    return reference[:length], output[:length]


def legacy_alignment(reference, output):
    """`xcorr_head`: the peak of |corr| over the first 81920 samples, clamped to +-4096, exactly as stored scores read it."""
    span = min(len(reference), len(output), LEGACY_SPAN)
    correlation = scipy.signal.correlate(output[:span], reference[:span], mode="full")
    peak = int(np.argmax(np.abs(correlation)))
    lag = max(-MAX_LAG, min(MAX_LAG, peak - (span - 1)))
    polarity = -1 if correlation.size and correlation[peak] < 0 else 1
    return Alignment(ALIGN_LEGACY, lag, lag, polarity, STATUS_LEGACY)


def span_length(rate):
    """The span the vote reads, in samples at `rate`."""
    return int(round(SPAN_S * rate))


def span_energies(reference, length, limit):
    """`(hop, energies)`: the energy of every span of `length` starting on a half-span grid inside `reference[:limit]`."""
    hop = max(1, length // 2)
    blocks = (limit - length) // hop + 2 if limit >= length else 0
    if blocks < 2:
        return hop, np.zeros(0)
    frames = np.asarray(reference)[: blocks * hop].reshape(blocks, hop)
    energy = np.einsum("ij,ij->i", frames, frames, dtype=np.float64)
    return hop, energy[:-1] + energy[1:]


def span_starts(reference, length, count=SPANS, limit=None):
    """Starts of up to `count` loudest non-overlapping spans of `length` in `reference[:limit]`; spans of digital silence are skipped."""
    limit = len(reference) if limit is None else min(limit, len(reference))
    hop, energies = span_energies(reference, length, limit)
    chosen = []
    for index in np.argsort(-energies, kind="stable")[: np.count_nonzero(energies > 0.0)]:
        _keep_apart(chosen, int(index) * hop, length)
        if len(chosen) == count:
            break
    return sorted(chosen)


def _keep_apart(chosen, start, length):
    """Appends `start` to `chosen` unless a span already chosen overlaps the one it starts."""
    if all(abs(start - other) >= length for other in chosen):
        chosen.append(start)


def _band_mask(spectrum_r, spectrum_e, rate, nfft):
    """The bins GCC-PHAT reads: inside the band, and carrying power within 30 dB of each side's in-band median."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
    band = (freqs >= BAND_HZ[0]) & (freqs <= min(BAND_HZ[1], 0.45 * rate))
    mask = band.copy()
    for spectrum in (spectrum_r, spectrum_e):
        power = np.abs(spectrum) ** 2
        floor = float(np.median(power[band])) * BIN_FLOOR if band.any() else 0.0
        mask &= power > floor
    return mask


def span_peak(ref_span, out_span, rate, prior_lag=None, max_lag=MAX_LAG):
    """The GCC-PHAT peak of one span pair (the output's lag behind the reference), or None when the band holds too few live bins."""
    size = len(ref_span)
    nfft = scipy.fft.next_fast_len(2 * size, real=True)
    window = np.hanning(size)
    spectrum_r = np.fft.rfft(np.asarray(ref_span, dtype=np.float64) * window, nfft)
    spectrum_e = np.fft.rfft(np.asarray(out_span, dtype=np.float64) * window, nfft)
    mask = _band_mask(spectrum_r, spectrum_e, rate, nfft)
    if int(mask.sum()) < MIN_BINS:
        return None
    lags, values = phase_coherence(spectrum_e * np.conj(spectrum_r), mask, nfft, min(max_lag, size - 1))
    index = _tie_break(lags, np.abs(values), prior_lag)
    return SpanPeak(int(lags[index]), float(values[index]))


def phase_coherence(cross, mask, nfft, reach):
    """`(lags, coherence)` over +-`reach`: the inverse transform of the cross-spectrum's phase in the masked bins, 1 for a pure delay."""
    gcc = np.fft.irfft(np.where(mask, cross / (np.abs(cross) + 1e-30), 0.0), nfft)
    values = np.concatenate((gcc[-reach:], gcc[: reach + 1])) * nfft / (2.0 * mask.sum())
    return np.arange(-reach, reach + 1), values


def _tie_break(lags, heights, prior_lag):
    """The index of the highest peak, or among those within 10 % of it the one nearest the prior (0 without one)."""
    contenders = np.flatnonzero(heights >= TIE_RATIO * heights.max())
    target = 0 if prior_lag is None else prior_lag
    return int(contenders[np.argmin(np.abs(lags[contenders] - target))])


def agreeing_peaks(peaks, prior_lag):
    """The spans agreeing with the lag most counted spans agree on (ties: nearest the prior, then the smaller shift)."""
    target = 0 if prior_lag is None else prior_lag
    lags = np.array([peak.lag for peak in peaks])
    support = (np.abs(lags[:, None] - lags[None, :]) <= AGREE_SAMPLES).sum(axis=1)
    finalists = lags[support == support.max()]
    centre = int(finalists[np.argmin(np.abs(finalists - target))])
    return [peak for peak, lag in zip(peaks, lags) if abs(lag - centre) <= AGREE_SAMPLES]


def _needed(counted, spans):
    """How many counted spans must agree: half of them, and two unless the pair held room for one span only."""
    return max(MIN_AGREE if spans > 1 else 1, int(np.ceil(counted / 2.0)))


def _abstain(reason, prior_lag, spans=0, counted=()):
    """The abstention record: no lag, the pair trimmed at the prior (0 without one)."""
    applied = 0 if prior_lag is None else int(prior_lag)
    return Alignment(ALIGN_GCC, None, applied, None, STATUS_ABSTAINED, reason, Vote(spans, len(counted)))


def _decided(agreeing, spans):
    """The record of a vote that carried: the agreeing spans' median lag, the sign of their summed coherence."""
    lag = int(round(float(np.median([peak.lag for peak in agreeing]))))
    total = sum(peak.coherence for peak in agreeing)
    coherence = float(np.median([abs(peak.coherence) for peak in agreeing]))
    vote = Vote(spans, len(agreeing), round(coherence, 4))
    return Alignment(ALIGN_GCC, lag, lag, 1 if total >= 0 else -1, STATUS_OK, "", vote)


def span_peaks(reference, output, rate, prior_lag=None):
    """`(peaks read, spans)`: the GCC-PHAT peak of every loudest span that had live bins, and how many spans were tried."""
    length = min(span_length(rate), len(reference), len(output))
    starts = span_starts(reference, length, limit=min(len(reference), len(output)))
    peaks = (span_peak(reference[start:][:length], output[start:][:length], rate, prior_lag) for start in starts)
    return [peak for peak in peaks if peak is not None], len(starts)


def counted_peaks(peaks):
    """The peaks whose phase coherence reaches `MIN_COHERENCE` (unrelated noise stays near 0.02)."""
    return [peak for peak in peaks if abs(peak.coherence) >= MIN_COHERENCE]


def _why_none_counted(read):
    """Why no span counted: no span had live bins (no programme), or none cleared the coherence floor."""
    return "weak correlation" if read else "no programme"


def gcc_alignment(reference, output, rate=NATIVE_RATE, prior_lag=None):
    """`gcc_phat_spans`: the loudest spans' GCC-PHAT peaks voting, or an abstention that says why."""
    if min(len(reference), len(output)) < MIN_SPAN_SAMPLES:
        return _abstain("too short", prior_lag)
    read, spans = span_peaks(reference, output, rate, prior_lag)
    counted = counted_peaks(read)
    if not counted:
        return _abstain(_why_none_counted(read), prior_lag, len(read))
    agreeing = agreeing_peaks(counted, prior_lag)
    if len(agreeing) < _needed(len(counted), spans):
        return _abstain("spans disagree", prior_lag, len(read), agreeing)
    return _decided(agreeing, len(read))


def read_alignment(reference, output, rate=NATIVE_RATE, method=None, prior_lag=None):
    """The `Alignment` of `output` against `reference` under `method` (resolved by `resolve_method`)."""
    if resolve_method(method) == ALIGN_LEGACY:
        return legacy_alignment(reference, output)
    return gcc_alignment(reference, output, rate, prior_lag)


def align(reference, output, rate=NATIVE_RATE, method=None, prior_lag=None):
    """`(reference, output, Alignment)`: the pair trimmed to one timeline at the alignment's applied lag."""
    alignment = read_alignment(reference, output, rate, method, prior_lag)
    trimmed_reference, trimmed_output = trim(reference, output, alignment.applied_lag)
    return trimmed_reference, trimmed_output, alignment
