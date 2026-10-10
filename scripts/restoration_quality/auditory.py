"""Auditory building blocks the v3 readings share: the ERB scale, programme cells, the programme gain match.

Ear v3 reads timbre the way a listener hears it, not the way a least-squares gain match sees
it. Three pieces every timbre reading needs:

- the ERB-number scale (Glasberg & Moore 1990: E = 21.4 log10(4.37 f / 1000 + 1)) and
  1-ERB bands, so a change is weighed per auditory filter, not per FFT bin;
- programme cells: time-frequency cells where the SOURCE programme stands clearly above the
  source's own floor (loud frames, bin >= floor + 12 dB). Removing hiss leaves those cells
  alone, so a denoiser reads as neutral there while a shelf or a tilt reads at its full size;
- the programme gain match: the median output/source level over programme cells in
  300-3000 Hz. The whole-file least-squares match (`score_reference._match_gain`) can absorb a
  tilt; this one cannot, because it is taken in the speech core only.

The air shelf at +2 dB lifted the whole APL output ~1.6 dB above 4 kHz, and no reading saw it:
the sibilance reading is net of the plain frames and the gap readings are relative to the
same side's loud frames, so a change that lifts every frame cancels in both.

Audibility (plan 1.2): is a candidate audibly different from the incumbent at all? The
sibilance loop v3 spent its rounds on guard moves of -70..-100 dBFS that nobody could hear;
a tie found before scoring costs nothing. `scripts/audibility_check.py` runs it on two files.

- `audio_sha256` / `pcm_sha256`: SHA-256 of the decoded float32 PCM (`audio_io.load_audio`,
  so DC removed) plus its rate and shape. Equal hashes mean equal decoded samples and
  nothing more: the container and its tags do not count. `compare_files` reads a finite
  pair with equal hashes as identical without the NMR, and `listen_cuts` hashes each cut
  with `pcm_sha256` (the ledger's `context.cut_sha256`). The self-driving loop does not
  mark candidates inert with either: it uses its own
  `autotune_restoration.exact_audio_sha256` (the samples as stored, DC kept, streamed in
  blocks), so a DC-only difference is never inert there. Its audibility tie reaches
  `pcm_sha256` only through `compare_files` (the identical shortcut, kept as `identical`
  in `<slug>.audibility.json`), so changing the hash or the shortcut changes the loop's
  tie verdicts. Here a DC-only difference usually still changes the hash too: `remove_dc`
  subtracts the mean in float32 and the rounding differs (measured: a voice and the same
  voice + 0.01 hash apart), so `compare_files` sends such a pair through the NMR. The
  whole file is decoded at once (a 460 s stereo excerpt is 162 MB).
- the null test (`audibility`): the candidate is aligned to the incumbent by the
  cross-correlation lag (within +-4096 samples, the reference scorer's bound) and the sign
  of the peak, never gain-matched. `diff_db` is the mean-square difference over every
  aligned sample relative to the incumbent's loud frames (>= p70 of 1024-sample frame
  power); `changed_share` is the share of samples that moved by more than -120 dBFS.
- the NMR: per 15 s window, the incumbent's power in 1-ERB bands from 50 Hz to
  min(16 kHz, Nyquist) on 2048-sample Hann frames (hop 1024) is its excitation; spread
  across bands (27 dB per ERB towards lower bands, 12 dB per ERB towards higher ones) and
  lowered by `MASKING_OFFSET_DB` it is the masked threshold, floored at the threshold in
  quiet (Terhardt 1979). The NMR per band and frame is 10 log10(E_diff / threshold), with
  E_diff the band power of the time-domain difference. A frame is audible when its largest
  NMR over bands and channels exceeds 0 dB; `audible_frac` is the largest per-window share
  of audible frames and `nmr_max` the largest NMR anywhere. An event frame is one whose NMR
  reaches `EVENT_NMR_DB` (+6 dB); `event_frames` is the largest per-window count of them.
- the verdict: audible when some window reaches 5% audible frames (a spread change: a
  shelf, a level, a noise floor) or holds an event (a sparse change, the kind the declick,
  plosive, pause floor and sibilant stages make): at least `EVENT_MIN_FRAMES` (2) event
  frames, or one frame at `SINGLE_EVENT_NMR_DB` (+12 dB). The share alone called plainly
  audible events a tie. Measured on the tests' synthetic voice (15 s): four 1 ms clicks at
  0.3 in the pauses read nmr_max +45.4, audible_frac 0.012, 8 event frames; one such click
  +44.8, 0.003, 2; a 50 ms muted dropout in a vowel +26.7, 0.005, 3: ties under the share
  rule, audible with the event clause. Two frames alone are not enough: a transient sits
  under the two frames that overlap it, whose Hann weights sum to about 1 in amplitude, and
  near a hop boundary one of them holds nearly all of it and the other almost nothing
  (-47 dB in power 44 samples in). Swept across one hop in 32-sample steps through a pause,
  one 1 ms click read a single event frame and a tie in 4 of 32 positions at 0.3 (nmr_max
  +40.0..+45.7), 13 at 0.03 (+20.0..+25.7) and 24 at 0.01 (+10.5..+16.2); with the single
  frame at +12 dB none of them is a tie anywhere, because the louder frame keeps at least
  half the amplitude (-6 dB). Clicks at 0.005 (+4.6..+10.4) read a tie in every position
  under either rule. A coherent level change of 1 dB caps the NMR at +5.7 dB (+1 dB
  broadband: 0 event frames, share 0.81), so a change near the threshold never trips the
  clause and stays the share's call (a coherent 2 dB change, cap +12.3, can trip the single
  frame; it lies above the 0.5-1 dB intensity JND); -90 dBFS noise (-13.7) and +0.4 dB
  (-2.5) still read 0 event frames, a tie. The calibration anchors below cannot move: an
  event needs +6 dB in two frames or +12 dB in one, and v3 vs v2 reads nmr_max -21.2..0.0
  on all four tapes and the 1-sample shift -200.
- lengths: `length_mismatch_s` is what the alignment leaves uncompared at both ends: the
  |lag| head `null_aligned` drops from the late side plus the tail one side holds past the
  other, |lag| + |len(incumbent) - len(candidate) + lag| samples. The NMR reads only the
  common span, so on the tests' voice a candidate cut to half, to 1 s or to nothing, or one
  with 10 s of loud noise appended, read lag 0, nmr_max -200, a tie before the tail was
  counted, and one missing its first 91 ms (4000 samples) or with 91 ms of noise at 0.3 put
  in front read lag -4000 or +4000, nmr_max -200, a tie before the head was counted. Past
  `LENGTH_TOLERANCE_S` (50 ms) the pair reads audible: the check cannot vouch for a span
  it never compared. A pure delay counts too: silence put in front reads its own length,
  and an equal-length shift twice it (the silent head and the lost tail), so a delay over
  50 ms, or a shift over 25 ms, reads audible, the conservative side of the one-sided
  design. The anchor pairs below have equal frame counts (read from the stored files'
  headers) and lag 0 or -1, so they leave 0 or 2 samples.
- channels: a mono side against a multichannel one is repeated on every channel of the
  other and compared per channel, so the worst channel decides and collapsing a stereo
  image to mono reads as the change it is: two different voices on L and R against their
  mean read diff_db -4.8, nmr_max +29.5, audible (compared as downmixes they read -200, a
  tie); a dual-mono copy of a mono voice against the voice reads nothing. Two different
  multichannel counts (2 against 6) are compared as mono downmixes and read audible with
  `channel_mismatch` set: the check cannot vouch for a tie.
- non-finite samples (a NaN from a broken render): set to 0 for the analysis, so a hole
  reads as the dropout it plays as, counted in `nonfinite` (both inputs together), and the
  pair reads audible whatever the NMR says. Unhandled, 1000 NaN samples steered the lag
  search to its bound (-4096) and read diff_db -200, and an all-NaN candidate read as
  identical to its incumbent; now they read lag 0, nmr_max +23.3 and +24.0, audible.
  `audio_io.remove_dc` spreads one NaN over its whole channel (the channel mean turns NaN),
  so a file read through `compare_files` with a single NaN counts the whole channel, and
  two different renders with a NaN in each channel decode alike: a 440 Hz tone and a
  660 Hz tone with noise read identical, inaudible, nonfinite 0 until `compare_files` took
  its identical shortcut only for finite samples. Such a pair now goes through the NMR.
- the level is an assumption (`PLAYBACK_SPL_LOUD_FRAMES`): the incumbent's loud frames are
  taken to play at 65 dB SPL, which places the threshold in quiet. The harness cannot know
  the real playback level.

One-sided by design: "inaudible" is trusted, "audible" is not. The difference is taken in
the time domain, so anything a listener might hear moves it, and so does a harmless
fractional-sample delay or an all-pass phase change (both read as audible); the masking
model is simple and the offset leans towards calling a change audible. A candidate the
check calls inaudible is a tie (it cannot win and skips the learned judges); a candidate it
calls audible has still proved nothing about being better.

Calibration (2026-10-09, read-only on the stored listening files in
`D:/Tata/New folder/variants`, offset 24 dB; the three anchors of plan 1.2):

| pair | tape | diff_db | changed | audible_frac | windows >= 5% | nmr_max |
|---|---|---|---|---|---|---|
| air +1 vs +2 dB (v4_air B, A) | tele7abc | -42.2 | 0.985 | 0.42 | 9/9 | 7.1 |
| | soti | -40.9 | 0.977 | 0.34 | 25/29 | 5.5 |
| | vaccin | -44.3 | 0.969 | 0.45 | 31/31 | 11.5 |
| | gaudeamus5 | -42.2 | 0.997 | 0.97 | 20/20 | 22.9 |
| sibilance v3 final vs v2 (v3 final2, final3) | tele7abc | -73.5 | 0.014 | 0 | 0/9 | -1.8 |
| | soti | -77.3 | 0.316 | 0 | 0/29 | -4.8 |
| | vaccin | -101.6 | 0.024 | 0 | 0/31 | -21.2 |
| | gaudeamus5 | -75.5 | 0.005 | 0.0015 | 0/20 | 0.0 |
| 1-sample shift of tele7abc B | tele7abc | none left | 0 | 0 | 0/9 | -200 |

- +1 vs +2 reads audible on every tape and v3 vs v2 inaudible on every tape, as the plan
  requires. The audible frames sit in the 4.9-8.6 kHz bands of programme frames, where the
  shelf acts. The +1 vs +2 anchor rests on one unblinded preference; if the Session 0 ABX
  cannot tell the two apart, the offset moves.
- At a 20 dB offset soti's +1 vs +2 reaches 5% in only 3 of 29 windows (audible_frac 0.087)
  and its +1 vs off becomes a tie (0.026); at 30 dB every anchor still holds. 24 dB keeps
  margin on both sides.
- The +2 dB renders of vaccin and gaudeamus5 sit one sample early against the +1 dB ones
  (lag -1, the same in every window); the integer alignment removes it.
- gaudeamus5 reads nmr_max 22.9 dB (+1 vs +2) and 32.0 dB (+1 vs off) at 8.6 kHz, more than
  any 1 dB shelf can produce (a coherent change of g dB caps the NMR at
  20 log10(10^(g/20) - 1) + 24 dB, +5.7 dB for 1 dB): its renders differ there by more than
  the shelf (not traced yet).
- On a synthetic voice (harmonic buzz with 's' bursts, loud frames at -20.8 dBFS): a
  broadband gain of +1 dB reads nmr_max +5.7 (audible), +0.5 dB -0.6 (a tie); a shelf
  above 4 kHz of +2 dB +12.1, +1 dB +5.6, +0.5 dB -0.7; white noise added at -90 dBFS
  -14.3 (under the threshold in quiet in the pauses), at -60 dBFS +15.7; requantising to
  16 bits -24.8. The offset puts a coherent level change of about 0.5 dB at the threshold,
  the low end of the 0.5-1 dB intensity JND.
"""

import hashlib

import numpy as np
import scipy.signal
from numpy.lib.stride_tricks import sliding_window_view

from scripts.restoration_quality import audio_io

ERB_SCALE = 21.4
ERB_CORNER = 4.37 / 1000.0
PROGRAMME_FLOOR_PERCENTILE = 10.0
PROGRAMME_MARGIN_DB = 12.0
LOUD_FRAME_PERCENTILE = 70.0
GAIN_BAND_HZ = (300.0, 3000.0)
MIN_GAIN_CELLS = 50
EPS = 1e-20

# The playback level the audibility check assumes: the incumbent's loud frames play at
# 65 dB SPL (conversational speech at a metre). Absolute audibility needs a level; the
# harness cannot know the listener's volume, so this is a declared assumption.
PLAYBACK_SPL_LOUD_FRAMES = 65.0
# Masked threshold = spread excitation minus this offset. Published offsets run from ~5 dB
# (noise masking noise) to 14.5 + z dB on the Bark scale (tone masking noise, Johnston 1988);
# 24 dB leans towards "audible": a coherent level change of ~0.5 dB in a band reads at the
# threshold, and it reproduces the plan's three anchors on the stored Tata files with margin,
# where 20 dB leaves soti's +1 vs +2 dB air audible in 3 of 29 windows only (see the module
# docstring). Session 0's ABX may move it.
MASKING_OFFSET_DB = 24.0
# Spreading of the excitation across 1-ERB bands, in dB per band: masking falls steeply
# towards lower frequencies and reaches further upwards (the classic two-slope shape,
# ~27 dB/Bark below and 10-24 dB/Bark above depending on level; one ERB is a little under
# one Bark in the speech band).
SPREAD_LOWER_DB_PER_ERB = 27.0
SPREAD_UPPER_DB_PER_ERB = 12.0
NMR_BAND_HZ = (50.0, 16000.0)
NMR_FRAME = 2048
NMR_HOP = 1024
NMR_WINDOW_S = 15.0
AUDIBLE_NMR_DB = 0.0
# A candidate with no window where at least 5% of the frames are audible is a tie
# (design 5.2, a starting value to calibrate).
TIE_AUDIBLE_FRAC = 0.05
# ... unless some window holds at least 2 frames whose NMR reaches +6 dB: a click or a dropout
# changes too few frames for the 5% share, yet reads +26..+45 dB (see the module docstring).
# +6 dB sits above the +5.7 dB a coherent 1 dB change can reach, so a change near the
# threshold never trips it: whether a small level or shelf move is audible stays the share's call.
EVENT_NMR_DB = 6.0
EVENT_MIN_FRAMES = 2
# ... or one frame at +12 dB. The two Hann frames over a sample (hop = half the frame) share
# it about 1 to 0 near a hop boundary, so a click there makes one event frame: the 2-frame
# rule read a +45 dB click a tie in 4 of 32 positions. Their amplitude weights sum to about 1,
# so the louder one keeps >= 0.5 (-6 dB) and one frame 6 dB over the event level stands for
# the pair: a click whose best frame reaches about +16 dB now reads audible wherever it falls.
# The anchors cannot move: v3 vs v2 reads nmr_max <= 0.0 and the 1-sample shift -200, and a
# coherent 1 dB change caps at +5.7 dB (see the module docstring). This keeps the hop-1024 grid
# the share was calibrated on; a second grid at half a hop would double the NMR's cost.
SINGLE_EVENT_NMR_DB = EVENT_NMR_DB + 6.0
# The aligned pair may leave at most 50 ms uncompared, head and tail together: the NMR reads
# only the common span, so a render cut short, empty or missing its first 91 ms read as a tie.
# The anchor pairs have equal frame counts and lag 0 or -1: 0 or 2 samples.
LENGTH_TOLERANCE_S = 0.05
MAX_LAG_SAMPLES = 4096
LAG_SPAN_S = 30.0
LEVEL_FRAME = 1024
CHANGED_SAMPLE_FLOOR = 10.0 ** (-120.0 / 20.0)
DB_FLOOR = -200.0
FLOOR_RATIO = 10.0 ** (DB_FLOOR / 10.0)
# The verdict rule's version. The self-driving loop keys its stored verdicts on it
# (`autotune_guards.cached_audible`, `<slug>.audibility.json`), so a verdict an older rule
# wrote is read again, not reused. Bump it with any change to what `compare_files` calls
# audible: the masking model, the share, the event clause, the length, channel or non-finite
# flags, the identical shortcut. 2026-10-09: the 5% share, events (2 frames at +6 dB or one at
# +12 dB) and the head-and-tail length rule.
VERDICT_RULE = "2026-10-09-head-tail"


def erb_number(freq_hz):
    """ERB-number (Cams) of `freq_hz` (scalar or array)."""
    return ERB_SCALE * np.log10(ERB_CORNER * np.asarray(freq_hz, dtype=np.float64) + 1.0)


def erb_to_hz(erb):
    """The frequency in Hz of ERB-number `erb` (inverse of `erb_number`)."""
    return (10.0 ** (np.asarray(erb, dtype=np.float64) / ERB_SCALE) - 1.0) / ERB_CORNER


def erb_band_edges(f_lo, f_hi):
    """Band edges in Hz, one ERB apart, from `f_lo` to `f_hi` (the last band ends at `f_hi`)."""
    lo, hi = float(erb_number(f_lo)), float(erb_number(f_hi))
    if hi <= lo:
        return np.array([float(f_lo), float(f_hi)])
    count = max(1, int(np.floor(hi - lo)))
    return erb_to_hz(np.linspace(lo, hi, count + 1))


def band_index(freqs, edges):
    """For each frequency bin, the ERB band it falls in, or -1 outside the edges."""
    index = np.searchsorted(edges, freqs, side="right") - 1
    index[(freqs < edges[0]) | (freqs >= edges[-1])] = -1
    return index


def loud_frames(power):
    """Boolean mask of the frames at or above the 70th percentile of frame power (`power` is `(bins, frames)`)."""
    level = 10.0 * np.log10(power.sum(axis=0) + EPS)
    return level >= np.percentile(level, LOUD_FRAME_PERCENTILE)


def programme_cells(src_power):
    """Boolean `(bins, frames)` mask: loud source frames where the bin stands >= 12 dB over its own floor.

    The floor per bin is the 10th percentile of the source's power in that bin over the
    window's frames, a proxy for the minimum-statistics noise floor.
    """
    floor = np.percentile(src_power, PROGRAMME_FLOOR_PERCENTILE, axis=1, keepdims=True)
    above = src_power >= floor * 10.0 ** (PROGRAMME_MARGIN_DB / 10.0)
    return above & loud_frames(src_power)[np.newaxis, :]


def cell_ratio_db(src_power, out_power):
    """Per-cell output-over-source level in dB."""
    return 10.0 * np.log10((out_power + EPS) / (src_power + EPS))


def programme_gain_db(src_power, out_power, freqs, cells):
    """The median output/source dB over programme cells in 300-3000 Hz, or None with too few cells."""
    in_band = (freqs >= GAIN_BAND_HZ[0]) & (freqs < GAIN_BAND_HZ[1])
    mask = cells & in_band[:, np.newaxis]
    if int(mask.sum()) < MIN_GAIN_CELLS:
        return None
    return float(np.median(cell_ratio_db(src_power, out_power)[mask]))


def pcm_sha256(audio, rate):
    """SHA-256 of float32 PCM and its rate and shape: equal only when every sample is equal."""
    pcm = np.ascontiguousarray(audio, dtype="<f4")
    digest = hashlib.sha256(f"{int(rate)}|{pcm.shape}".encode("ascii"))
    digest.update(pcm.reshape(-1).view(np.uint8))
    return digest.hexdigest()


def audio_sha256(path):
    """SHA-256 of a file's decoded PCM (`audio_io.load_audio`, DC removed): the container and its tags do not count.

    Equal hashes mean equal decoded samples, nothing more. Removing the DC does not make a
    DC-only difference hash alike: the float32 rounding of the subtraction differs.
    """
    audio, rate = audio_io.load_audio(path)
    return pcm_sha256(audio, rate)


def estimate_lag(reference, candidate, rate):
    """`(lag, polarity)` of mono `candidate` against mono `reference`: the cross-correlation peak within +-4096 samples.

    A positive lag means the candidate is that many samples late; the polarity is the sign
    of the peak (-1: the candidate is inverted). Silence on either side reads `(0, 1)`.
    """
    span = min(len(reference), len(candidate), int(LAG_SPAN_S * rate))
    ref = np.asarray(reference[:span], dtype=np.float64)
    cand = np.asarray(candidate[:span], dtype=np.float64)
    if not (np.any(ref) and np.any(cand)):
        return 0, 1
    correlation = scipy.signal.correlate(cand, ref, mode="full", method="fft")
    lags = np.arange(-(span - 1), span)
    inside = np.abs(lags) <= MAX_LAG_SAMPLES
    peak = int(np.argmax(np.abs(correlation[inside])))
    return int(lags[inside][peak]), -1 if correlation[inside][peak] < 0 else 1


def null_aligned(incumbent, candidate, lag, polarity):
    """Both signals trimmed to their common span at `lag`, the candidate multiplied by `polarity`."""
    if lag > 0:
        candidate = candidate[lag:]
    elif lag < 0:
        incumbent = incumbent[-lag:]
    length = min(len(incumbent), len(candidate))
    return incumbent[:length], candidate[:length] * np.float32(polarity)


def unmatched_samples(incumbent_length, candidate_length, lag):
    """Samples `null_aligned` never compares at `lag`: the |lag| head it drops from the late side plus the tail of the longer side."""
    head = abs(int(lag))
    return head + abs(int(incumbent_length) - int(candidate_length) + int(lag))


def _two_d(audio):
    array = np.asarray(audio, dtype=np.float32)
    return array[:, np.newaxis] if array.ndim == 1 else array


def finite_part(audio):
    """`(audio as (frames, channels) with every NaN and inf set to 0, how many samples were non-finite)`."""
    array = _two_d(audio)
    bad = ~np.isfinite(array)
    return np.where(bad, np.float32(0.0), array), int(np.count_nonzero(bad))


def matched_channels(incumbent, candidate):
    """`(incumbent, candidate, mismatch)`: both `(frames, channels)` with one channel count.

    A mono side is repeated on every channel of the other, so the per-channel comparison
    sees a stereo image collapsed to mono while a dual-mono copy of a mono voice reads
    nothing. Two different multichannel counts are compared as mono downmixes, `mismatch` True.
    """
    inc, cand = _two_d(incumbent), _two_d(candidate)
    counts = (inc.shape[1], cand.shape[1])
    if counts[0] == counts[1]:
        return inc, cand, False
    if min(counts) == 1:
        width = max(counts)
        return np.repeat(inc, width // counts[0], axis=1), np.repeat(cand, width // counts[1], axis=1), False
    return audio_io.to_mono(inc)[:, np.newaxis], audio_io.to_mono(cand)[:, np.newaxis], True


def loud_frame_power(audio):
    """Mean-square level of the loud 1024-sample frames (>= p70 of frame power over every channel) of `(frames, channels)`."""
    count = len(audio) // LEVEL_FRAME
    if count == 0:
        return float(np.sum(np.square(audio, dtype=np.float64)) / max(audio.size, 1)) + EPS
    frames = audio[: count * LEVEL_FRAME].reshape(count, LEVEL_FRAME, -1)
    power = np.einsum("ijk,ijk->i", frames, frames, dtype=np.float64) / (LEVEL_FRAME * frames.shape[2])
    return float(np.mean(power[power >= np.percentile(power, LOUD_FRAME_PERCENTILE)])) + EPS


def frame_power(mono):
    """Hann STFT power `(bins, frames)` (2048 / hop 1024), scaled so a band's sum is its mean-square level.

    The signal is padded by a hop in front and a frame behind, so every sample sits under
    the peak of some window.
    """
    padded = np.pad(np.asarray(mono, dtype=np.float64), (NMR_HOP, NMR_FRAME))
    window = np.hanning(NMR_FRAME)
    spectrum = np.fft.rfft(sliding_window_view(padded, NMR_FRAME)[::NMR_HOP] * window, axis=1)
    return (2.0 / (NMR_FRAME * np.sum(window**2))) * (np.abs(spectrum) ** 2).T


def nmr_bands(rate):
    """`(matrix (bands, bins), centres in Hz)`: the 1-ERB bands of the NMR, 50 Hz to min(16 kHz, Nyquist)."""
    edges = erb_band_edges(NMR_BAND_HZ[0], min(NMR_BAND_HZ[1], rate / 2.0))
    index = band_index(np.fft.rfftfreq(NMR_FRAME, 1.0 / rate), edges)
    matrix = (index[np.newaxis, :] == np.arange(len(edges) - 1)[:, np.newaxis]).astype(np.float64)
    centres = erb_to_hz((erb_number(edges[:-1]) + erb_number(edges[1:])) / 2.0)
    return matrix, centres


def threshold_in_quiet_db(freq_hz):
    """Terhardt's (1979) absolute threshold of hearing in dB SPL."""
    khz = np.maximum(np.asarray(freq_hz, dtype=np.float64), 20.0) / 1000.0
    return 3.64 * khz**-0.8 - 6.5 * np.exp(-0.6 * (khz - 3.3) ** 2) + 1e-3 * khz**4


def spreading_matrix(count):
    """`(count, count)` power weights: entry `[j, i]` is how much band `i` excites band `j` (1 on the diagonal)."""
    distance = np.arange(count)[:, np.newaxis] - np.arange(count)[np.newaxis, :]
    slope = np.where(distance >= 0, SPREAD_UPPER_DB_PER_ERB, SPREAD_LOWER_DB_PER_ERB)
    return 10.0 ** (-slope * np.abs(distance) / 10.0)


def masked_threshold(excitation, centres, offset_db=MASKING_OFFSET_DB):
    """Masked threshold per band and frame (power, 1 = 0 dB SPL): the spread excitation minus the offset, floored in quiet."""
    spread = spreading_matrix(len(centres)) @ excitation
    quiet = 10.0 ** (threshold_in_quiet_db(centres) / 10.0)
    return np.maximum(spread * 10.0 ** (-offset_db / 10.0), quiet[:, np.newaxis])


def band_nmr(reference, difference, rate, spl_offset_db, offset_db=MASKING_OFFSET_DB):
    """NMR in dB per ERB band and frame `(bands, frames)`: mono `difference` against the masked threshold of `reference`.

    `spl_offset_db` turns a mean-square level in dB re full scale into dB SPL.
    """
    matrix, centres = nmr_bands(rate)
    gain = 10.0 ** (spl_offset_db / 10.0)
    excitation = matrix @ frame_power(reference) * gain
    noise = matrix @ frame_power(difference) * gain
    return floored_db(noise / masked_threshold(excitation, centres, offset_db))


def floored_db(ratio):
    """10 log10 of a power ratio (scalar or array); a ratio at or under 1e-20 (no difference) reads exactly -200 dB.

    NaN stays NaN: an undefined ratio is never read as "no difference".
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    return np.where((ratio > FLOOR_RATIO) | np.isnan(ratio), 10.0 * np.log10(np.maximum(ratio, FLOOR_RATIO)), DB_FLOOR)


def window_slices(n_samples, rate, seconds=NMR_WINDOW_S):
    """Consecutive `seconds`-long slices covering every sample; a tail under half a window joins the one before."""
    size = max(1, int(round(seconds * rate)))
    starts = list(range(0, n_samples, size))
    if len(starts) > 1 and n_samples - starts[-1] < size // 2:
        starts.pop()
    return [slice(start, end) for start, end in zip(starts, starts[1:] + [n_samples])]


def frame_nmr(reference, difference, rate, spl_offset_db, offset_db=MASKING_OFFSET_DB):
    """Per frame, the largest NMR over the bands and channels of `(samples, channels)` signals."""
    channels = range(reference.shape[1])
    return np.max([band_nmr(reference[:, ch], difference[:, ch], rate, spl_offset_db, offset_db).max(axis=0) for ch in channels], axis=0)


def _window_row(incumbent, candidate, rate, spl_offset_db, offset_db):
    """`[difference energy, changed samples, samples, audible frame share, largest NMR, event frames]` of one window."""
    inc = incumbent.astype(np.float64)
    diff = candidate.astype(np.float64) - inc
    nmr = frame_nmr(inc, diff, rate, spl_offset_db, offset_db)
    changed = np.count_nonzero(np.abs(diff) > CHANGED_SAMPLE_FLOOR)
    events = np.count_nonzero(nmr >= EVENT_NMR_DB)
    return [
        float(np.sum(diff**2)),
        float(changed),
        float(diff.size),
        float(np.mean(nmr > AUDIBLE_NMR_DB)),
        float(np.max(nmr)),
        float(events),
    ]


# The flags of a pair the check can vouch for: one channel count (or a mono side), every
# sample finite, and nothing left uncompared at either end once aligned.
VOUCHED = {"channel_mismatch": False, "nonfinite": 0, "length_mismatch_s": 0.0}


def cannot_vouch(flags):
    """True for a pair no NMR can call a tie: two multichannel counts, a non-finite sample, or over 50 ms left uncompared."""
    return any([bool(flags["channel_mismatch"]), flags["nonfinite"] > 0, flags["length_mismatch_s"] > LENGTH_TOLERANCE_S])


def is_audible(audible_frac, event_frames, nmr_max=DB_FLOOR, flags=None):
    """The verdict: a spread change (5% audible frames), an event (2 frames at +6 dB or one at +12 dB), or an unvouched pair.

    `flags` are those of `VOUCHED`; a pair the check `cannot_vouch` for reads audible whatever the NMR says.
    """
    flags = {**VOUCHED, **(flags or {})}
    readings = [audible_frac >= TIE_AUDIBLE_FRAC, event_frames >= EVENT_MIN_FRAMES, nmr_max >= SINGLE_EVENT_NMR_DB]
    return any(readings) or cannot_vouch(flags)


def _summary(lag, polarity, rows, loud_power, flags=None):
    flags = {**VOUCHED, **(flags or {})}
    table = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    energy, changed, samples = table[:, :3].sum(axis=0)
    samples = max(samples, 1.0)
    audible_frac = float(np.max(table[:, 3], initial=0.0))
    event_frames = int(np.max(table[:, 5], initial=0.0))
    nmr_max = float(np.max(table[:, 4], initial=DB_FLOOR))
    return {
        "lag": int(lag),
        "polarity": int(polarity),
        "diff_db": float(floored_db(energy / samples / loud_power)),
        "changed_share": float(changed / samples),
        "audible_frac": audible_frac,
        "nmr_max": nmr_max,
        "event_frames": event_frames,
        "audible": is_audible(audible_frac, event_frames, nmr_max, flags),
        "channel_mismatch": bool(flags["channel_mismatch"]),
        "nonfinite": int(flags["nonfinite"]),
        "length_mismatch_s": float(flags["length_mismatch_s"]),
        "window_audible_frac": [float(value) for value in table[:, 3]],
        "window_event_frames": [int(value) for value in table[:, 5]],
    }


def audibility(incumbent, candidate, rate, offset_db=MASKING_OFFSET_DB):
    """The null test and the NMR of `candidate` against `incumbent` (`(frames,)` or `(frames, channels)` at `rate`).

    Returns `{lag, polarity, diff_db, changed_share, audible_frac, nmr_max, event_frames,
    audible, channel_mismatch, nonfinite, length_mismatch_s, window_audible_frac,
    window_event_frames}`; `audible` is False (a tie) only when no window reaches 5% audible
    frames, 2 event frames or one frame at +12 dB, the channel counts match (or one side is
    mono), every sample is finite and the alignment leaves at most 50 ms uncompared (the
    |lag| head and the tail together, `length_mismatch_s`).
    """
    (inc, inc_bad), (cand, cand_bad) = finite_part(incumbent), finite_part(candidate)
    inc, cand, mismatch = matched_channels(inc, cand)
    lag, polarity = estimate_lag(audio_io.to_mono(inc), audio_io.to_mono(cand), rate)
    flags = {
        "channel_mismatch": mismatch,
        "nonfinite": inc_bad + cand_bad,
        "length_mismatch_s": unmatched_samples(len(inc), len(cand), lag) / float(rate),
    }
    inc, cand = null_aligned(inc, cand, lag, polarity)
    loud_power = loud_frame_power(inc)
    spl_offset_db = PLAYBACK_SPL_LOUD_FRAMES - 10.0 * np.log10(loud_power)
    rows = [_window_row(inc[span], cand[span], rate, spl_offset_db, offset_db) for span in window_slices(len(inc), rate)]
    return _summary(lag, polarity, rows, loud_power, flags)


def compare_files(incumbent_path, candidate_path, offset_db=MASKING_OFFSET_DB):
    """`audibility` of two files plus `identical` (equal, finite decoded PCM) and both PCM hashes.

    Identical files are not analysed: they read lag 0, no difference and nothing audible.
    A pair with a non-finite sample is never identical: `remove_dc` spreads one NaN over its
    channel, so two different broken renders can decode alike; it goes through `audibility`,
    which counts the samples and reads the pair audible. Files at different sample rates
    cannot be compared sample by sample (ValueError).
    """
    incumbent, incumbent_rate = audio_io.load_audio(incumbent_path)
    candidate, candidate_rate = audio_io.load_audio(candidate_path)
    hashes = {"incumbent_sha256": pcm_sha256(incumbent, incumbent_rate), "candidate_sha256": pcm_sha256(candidate, candidate_rate)}
    # Equal hashes mean bit-equal arrays, so the incumbent's finiteness is the candidate's.
    if hashes["incumbent_sha256"] == hashes["candidate_sha256"] and np.isfinite(incumbent).all():
        return {"identical": True, **_summary(0, 1, [], 1.0), **hashes}
    if incumbent_rate != candidate_rate:
        raise ValueError(f"sample rates differ: {incumbent_rate} Hz against {candidate_rate} Hz")
    return {"identical": False, **audibility(incumbent, candidate, incumbent_rate, offset_db), **hashes}
