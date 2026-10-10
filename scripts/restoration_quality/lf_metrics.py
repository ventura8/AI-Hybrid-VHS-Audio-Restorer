"""R11, the programme's low end: how far the output's 40-200 Hz moved against its mids, on the loud frames.

Round C3 (the cathar music profile) moves the low end of music: the dewind cutoff
(`cathar_music_dewind_cutoff` 40 / 60 / 80 Hz; music runs the speech chain's 80 Hz dewind
behind the pre-conditioning high pass today, a double high pass), repair off on music, the
music expander depth and CRT Q. Nothing read that low end. R1's body starts at 100 Hz and read
6 of 150 realistic-v2 music windows: music's bass never falls to a floor inside 15 s, and R1's
programme cells need a bin 12 dB over its own p10. R11 reads the band's power instead of its
cells. Per window (the runner's 15 s windows, native rate, `dsp.lf_programme_db`):

1. two Hann STFTs of the source and the output, each on one timeline, each dropping the frames
   more than 90 dB under the loudest source frame from both sides (R1's live frames): the gain
   STFT, 8192 / 2048 (5.4 Hz bins at 44.1 kHz), and the band STFT, whose frame follows the rate
   to hold about 1.35 Hz bins (`frame_length`: 32768 samples at 44.1 and 48 kHz, 65536 at 96)
   with a quarter-frame hop;
2. R1's programme gain match on the gain STFT (`matched_gain_db`: the median out/src dB over
   `balance_metrics.guarded_cells` in 300-3000 Hz), so a broadband gain cancels exactly and the
   low end reads against the mids. Its 0.19 s frames see the floor between notes that the
   cells' noise guard needs; the band STFT's 0.74 s frames straddle the notes and left no
   guarded cell on a bed gated 0.5 s on, 0.5 s off;
3. the band, on the band STFT: the bins in 40-200 Hz less +-2 bins (the Hann main lobe: 2.7 Hz
   at 44.1 kHz) round each harmonic of the mains R0 names (none when R0 names none; the runner
   passes no fallback): dehum is not a thinner bass, and `dsp.hum_excess_db` reads it. On the
   8192-sample frame the same +-2 bins were 10.8 Hz and took 39-61 Hz out round a 50 Hz mains,
   the band the dewind corners move: 40-60 Hz 20 dB down read -0.007 dB on a bass-led bed (-0.99
   now), and at 96 kHz 1 of the band's 14 bins was left. The band keeps about 100 bins at any
   rate, and reads None under 8 (`MIN_LF_BINS`);
4. 10 log10 of the output's band power over the source's, summed over the band STFT's loud
   source frames (`auditory.loud_frames`, >= p70), less the gain. Power-weighted, so for any
   filter it is the filter's power response averaged over the programme's own low end: it
   follows a shelf's gain and a high pass's corner in order whatever the spectrum.

None when either side is shorter than one band frame or under -70 dBFS RMS, with fewer than 30
live gain frames (1.4 s at 44.1 kHz), with under 100 guarded cells in the core to match the
gain on (`MIN_GAIN_CELLS`), with a gain under -60 dB (a dead render), with under 8 band bins,
or when the band holds under 1/10000 of the core's power on the loud frames
(`MIN_LF_SHARE_DB`: a programme high-passed at 400 Hz has no low end to judge). The runner also
skips the windows R1's mute guard skips.

Two-sided: negative is a thinner low end, positive more low end than the source against the
mids. Noise in the band counts (it is power on the loud frames), so rumble left reads nearer 0
than rumble removed: the loop ranks the distance from the accepted file's reading, never the
raw direction, and the target decides how much rumble removal is wanted.

Measured (2026-10-10, the band STFT, the runner's window path):

- calibration (`quality_degradations`, en / fr, benign floor 8.4e-6 dB): FFmpeg's `bass` shelf
  at 100 Hz, +1.5 / 3 / 6 dB, reads +0.56 / +1.15 / +2.41 on the en music bed (fr +0.62 /
  +1.27 / +2.66) and the cut the mirror (-0.53 / -1.04 / -2.01, fr -0.60 / -1.17 / -2.26); its
  high pass at 40 / 60 / 80 Hz reads -0.12 / -0.47 / -1.09 (fr -0.15 / -0.57 / -1.27), and on the
  speech targets -0.014 / -0.073 / -0.224 (fr -0.042 / -0.140 / -0.317). Hum added at R0's mains
  (0.003 / 0.01 / 0.03) moves it 0.022 at most; a spectral tilt above 1 kHz reads the other way
  (en -0.14 / -0.28 / -0.56 for +0.5 / 1 / 2 dB/oct), because the gain match takes in 1-3 kHz,
  as R1's body does. The 8192-sample frame read the same rows within 0.16 dB;
- oracle removal of the fixtures' own noise, 20 dB down (source the fixture's capture, output
  its clean programme plus the noise 20 dB down, one 15 s window each, R0's mains), over the
  200 tape-noise-only realistic-v2 fixtures: the beds (musiconly, musicled) move p50 0.03 / p90
  0.15 dB, at most 0.40 (read on 96 of 100); speech over a bed p50 0.09 / p90 0.61, at most 3.5;
  speech p50 0.08 / p90 1.04, at most 2.5. The 8192-sample frame read p90 0.17 / 0.77 / 0.70 on
  the same pairs. That noise is low-heavy (a median 7 dB more power per hertz in 40-200 Hz than
  in 300-3000 Hz, up to 17), and where the loud frames' band sits within a few dB of it (LF SNR
  -4..+3 dB) its removal is rumble removed, which R11 reads by design. On speech the band holds
  little programme (fr's target: 17.6 dB under the core), so there R11 is shown, not ranked,
  until a verdict says otherwise;
- read on the 8192-sample frame and not read again on the band STFT: the 12-clip music
  listening set (2026-09-22), APL a median -0.50 dB, cathar's music profile -0.68 and cathar's
  speech settings -1.28 (down to -16.8 on dances-with-wolves); gaudeamus5's music windows,
  cathar's round-one baseline -3.9 against APL's -0.74, its plateau final -1.55 against APL's
  -0.98.

Why these choices (read on the 8192-sample frame). The per-cell median R1 uses moved under the
same oracle p90 1.3-1.4 dB on music and 2.4-2.7 on speech, against the power's 0.35-0.47 and
0.87-0.98; every live frame in place of the loud ones moved music slightly more (p90
0.47-0.52). Under 100 gain cells the median belongs to noise-only cells: on two beds whose
programme ends at 1.3 kHz, 78 cells, half of them hiss, set the gain -19 and -12 dB and R11 read
+18.9 and +12.0; over the oracle's 784 windows the 12 under 100 cells moved p90 11.7 dB, the 20
at 100-200 cells 0.3 at most. R1's band guard (b) applied to the core missed both and dropped
half the beds' windows; clipping the gain band to R0's programme band missed both too. The gain
STFT is that frame still, so these hold. The high pass and the shelf in the calibration run
zero-phase (`degradations_v3`): run causally on a bass-led bed they turned the alignment by up
to 101 samples.
"""

import numpy as np
import scipy.signal

from scripts.restoration_quality import audio_io, auditory, balance_metrics

READING = "lf_programme_db"
LF_BAND_HZ = (40.0, 200.0)
# The gain match's STFT, as R11 always took it: 8192 / 2048 Hann, 5.4 Hz bins at 44.1 kHz. Its 0.19 s frames see the
# floor between notes that the gain match's noise guard needs; the band STFT's 0.74 s frames do not.
GAIN_FRAME = 8192
GAIN_HOP = 2048
# The band's STFT holds about 1.35 Hz bins at any rate (`frame_length`: 32768 samples at 44.1 and 48 kHz, 1.35 /
# 1.46 Hz), so a mains line leaves +-2.7 Hz of the band and round C3's 40 / 60 / 80 Hz corners sit about 15 bins
# apart. On the 8192-sample frame a 50 Hz mains took 39-61 Hz out, the band the dewind corners move, and 13 of the
# band's 14 bins at 96 kHz.
BIN_HZ = 1.35
# The band's hop, a quarter frame: 0.19 s at 44.1 kHz, 82 frames in a 15 s window.
HOP_FRACTION = 4
# A mains line's Hann main lobe spans +-2 bins: 2.7 Hz at 44.1 kHz, 2.9 Hz at 48 kHz.
HUM_HALF_WIDTH_BINS = 2.0
# The band must keep at least this many bins once the mains lines are out (it keeps about 100 at any rate).
MIN_LF_BINS = 8
# About 1.4 s of live gain frames at 44.1 kHz: the loud 30 % of fewer is too few frames to read.
MIN_LIVE_FRAMES = 30
# The band must hold at least 1/10000 of the 300-3000 Hz core's power on the loud frames to carry programme.
MIN_LF_SHARE_DB = -40.0
# Guarded programme cells the gain match needs in the core: under 100, noise-only cells can own its median.
MIN_GAIN_CELLS = 100
EPS = 1e-20


def frame_length(rate):
    """The band STFT's frame at `rate`: the power of two nearest `rate / BIN_HZ` samples (32768 at 44.1 and 48 kHz)."""
    return int(2 ** round(np.log2(rate / BIN_HZ)))


def hop_length(rate):
    """The band STFT's hop at `rate`: a quarter of `frame_length`."""
    return frame_length(rate) // HOP_FRACTION


def lf_programme_db(source, output, rate, mains_hz=None):
    """R11 on one window of a lag-aligned pair, in dB; None where the window cannot carry it.

    The output's 40-200 Hz power over the source's on the source's loud frames, the mains
    lines `mains_hz` names left out, less R1's programme gain match on 300-3000 Hz.
    """
    pair = _mono_pair(source, output, rate)
    gain = None if pair is None else gain_db(*pair, rate)
    if gain is None:
        return None
    band = band_powers(*pair, rate, mains_hz)
    if band is None:
        return None
    src_lf, out_lf = band
    return float(10.0 * np.log10((out_lf + EPS) / (src_lf + EPS))) - gain


def _mono_pair(source, output, rate):
    """The pair downmixed the way the runner's is, at its common length; None when a side is under one band frame or -70 dBFS."""
    source, output = audio_io.to_mono(np.asarray(source)), audio_io.to_mono(np.asarray(output))
    length = min(len(source), len(output))
    if length < frame_length(rate) or not (_audible(source[:length]) and _audible(output[:length])):
        return None
    return source[:length], output[:length]


def gain_db(source, output, rate):
    """R1's programme gain match on the gain STFT's live frames; None under `MIN_LIVE_FRAMES` of them or where it cannot match."""
    freqs, src_power, out_power = live_spectra(source, output, rate, GAIN_FRAME, GAIN_HOP)
    if src_power.shape[1] < MIN_LIVE_FRAMES:
        return None
    return matched_gain_db(src_power, out_power, freqs)


def band_powers(source, output, rate, mains_hz=None):
    """`(source, output)` power in the band, the mains lines out, on the band STFT's loud frames; None when it cannot be judged.

    None when the band keeps under `MIN_LF_BINS` bins or holds under `MIN_LF_SHARE_DB` of the
    core's power on those frames.
    """
    freqs, src_power, out_power = live_spectra(source, output, rate, frame_length(rate), hop_length(rate))
    loud = auditory.loud_frames(src_power)
    bins = lf_bins(freqs, rate, mains_hz)
    src_lf, out_lf = (float(power[bins][:, loud].sum()) for power in (src_power, out_power))
    if int(np.count_nonzero(bins)) < MIN_LF_BINS or lf_share_db(src_power, freqs, loud, src_lf) < MIN_LF_SHARE_DB:
        return None
    return src_lf, out_lf


def matched_gain_db(src_power, out_power, freqs):
    """R1's programme gain match on R11's gain frames; None with under `MIN_GAIN_CELLS` guarded cells in the core or under -60 dB.

    The median out/src dB over `balance_metrics.guarded_cells` in 300-3000 Hz; a gain 60 dB
    down is a dead render, not a restoration.
    """
    cells = balance_metrics.guarded_cells(src_power)
    if int(np.count_nonzero(cells[core_bins(freqs)])) < MIN_GAIN_CELLS:
        return None
    gain = auditory.programme_gain_db(src_power, out_power, freqs, cells)
    return gain if gain >= balance_metrics.MIN_PROGRAMME_GAIN_DB else None


def core_bins(freqs):
    """The bins of the 300-3000 Hz core the gain is matched on."""
    return (freqs >= auditory.GAIN_BAND_HZ[0]) & (freqs < auditory.GAIN_BAND_HZ[1])


def lf_bins(freqs, rate, mains_hz=None):
    """The band STFT's bins in 40-200 Hz, less those within the Hann main lobe of a harmonic of `mains_hz` (none when None)."""
    bins = (freqs >= LF_BAND_HZ[0]) & (freqs < LF_BAND_HZ[1])
    if not mains_hz or mains_hz <= 0:
        return bins
    width = HUM_HALF_WIDTH_BINS * rate / frame_length(rate)
    harmonics = mains_hz * np.arange(1, int((LF_BAND_HZ[1] + width) // mains_hz) + 1)
    near = np.abs(np.asarray(freqs)[:, np.newaxis] - harmonics[np.newaxis, :]) <= width
    return bins & ~near.any(axis=1)


def lf_share_db(src_power, freqs, loud, src_lf):
    """The source's low-end power over its 300-3000 Hz core power on the loud frames, in dB."""
    return float(10.0 * np.log10((src_lf + EPS) / (float(src_power[core_bins(freqs)][:, loud].sum()) + EPS)))


def live_spectra(source, output, rate, frame, hop):
    """`(freqs, src_power, out_power)` of one STFT over the source's live frames (R1's: within 90 dB of the loudest)."""
    freqs, src_power = stft_power(source, rate, frame, hop)
    live = _live(src_power)
    return freqs, src_power[:, live], stft_power(output, rate, frame, hop)[1][:, live]


def _audible(mono):
    """At or above R1's silence level (-70 dBFS RMS)."""
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    return 20.0 * np.log10(rms + EPS) >= balance_metrics.MIN_LEVEL_DBFS


def _live(src_power):
    """The source frames within R1's 90 dB of the loudest one: digital silence and padding are no programme."""
    level = src_power.sum(axis=0)
    return level > level.max() * 10.0 ** (-balance_metrics.DYNAMIC_RANGE_DB / 10.0)


def stft_power(mono, rate, frame, hop):
    """Hann STFT power `(bins, frames)` of `frame` / `hop` samples and its bin frequencies."""
    freqs, _times, spec = scipy.signal.stft(np.asarray(mono, dtype=np.float64), fs=rate, window="hann", nperseg=frame, noverlap=frame - hop)
    return freqs, np.abs(spec) ** 2
