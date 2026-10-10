"""R1, the programme-cell spectral balance: how far the output's timbre moved from the source's, per ERB band.

Ear v2 could not see a whole-spectrum tilt. The +2 dB air shelf lifted the whole APL output
about 1.6 dB above 4 kHz on every frame and the user heard the 's' go thin, yet no reading
moved: the sibilance reading is net of the plain frames, the gap readings are relative to the
same side's loud frames, and the whole-file least-squares gain match can absorb a tilt. R1
reads the balance absolutely and two-sidedly, on the cells that carry programme only. Per
window (the runner's 15 s windows, native rate):

1. STFT 2048/512 Hann of the source and the output on one timeline (`lag_aligned` applies
   `align_pair`'s lag and no gain; a stereo side is downmixed with `audio_io.to_mono`, as the
   runner's pair is). Frames more than 90 dB under the loudest source frame (digital
   silence, capture padding) are dropped from both sides before any statistic;
2. programme cells: `auditory.programme_cells` (loud source frames, bin >= its p10 floor +
   12 dB), narrowed by two guards below;
3. the programme gain match (`auditory.programme_gain_db`, median out/src dB over the
   programme cells in 300-3000 Hz) is subtracted, so any broadband gain cancels exactly, the
   least-squares gain `align_pair` puts on the output included;
4. per 1-ERB band (`auditory.erb_band_edges` / `band_index`) from 100 Hz to min(bandwidth,
   16 kHz, Nyquist), the median out/src dB over the band's programme cells. A band reads NaN
   with fewer than 200 cells or when it is noise-led (guard b). The CRT line and its
   +-250 Hz skirt (`dsp_metrics.band_bins`) never count: removing it is not a timbre change.

Readings, each None where no band carries it (the tilt needs 3 bands); all None when either
side is shorter than one STFT frame or under -70 dBFS RMS, with fewer than 100 live frames
(about 1.2 s), without 50 programme cells in 300-3000 Hz to match the gain, or when that gain
is under -60 dB (a dead render: an all-zero output used to read tilt +2.38 dB/oct, air +6.06,
body -8.12, the negative of the source's own spectrum):

- `balance_tilt_db_oct`: least-squares slope of the band levels against log2 of the band
  centre over 1 kHz-bandwidth, in dB per octave;
- `balance_presence_db`: mean band level over band centres in 2-5 kHz;
- `balance_air_db`: mean band level over band centres from 5 kHz to the bandwidth;
- `balance_body_db`: mean band level over band centres in 100-300 Hz (nothing read body or
  warmth before).

Every level is relative to the 300-3000 Hz programme median (step 3), so a change inside
1-3 kHz also shows, with the opposite sign, in body: a -1 dB/oct tilt above 1 kHz reads body
+0.53 dB on the sibilance tests' speech and +0.85 on the unit tests' harmonic programme
(realistic-v2 median +0.17 speech, +0.29 music). Under two-sided scoring one top-end change
moves both the bright/dull distance and the body distance.

The guards, and why. Benign case from the plan: oracle hiss removal on `artifacts/realistic-v2`
(the 200 tape-noise-only fixtures, 50 speech and 150 music, 15 s each; output = clean + the
fixture's own noise at -10 / -20 dB) must move R1 by under 0.3 dB. The programme cells as
first specified failed it: air moved by a median -1.7 / -2.9 dB and up to -7.8 / -15 dB, the
tilt by -0.8 / -1.4 dB/oct, because hiss-only cells passed as programme:

a. a bin's p10 sits 9.8 dB under the mean of steady noise (one STFT cell's power is
   exponential: p10 = -ln 0.9 = 0.105 x mean; minimum-statistics floors carry the same bias
   compensation, Martin 2001), so "floor + 12 dB" passed ~19 % of loud-frame noise cells.
   A cell must also stand 18 dB over the bias-corrected mean (`NOISE_MARGIN_DB`); 12 and
   15 dB left music air at p90 |move| 0.57-0.73 and 0.38-0.49 dB with tilt outliers of
   5.7-11 dB/oct. On harmonics 600 Hz apart over steady hiss, with guard (b) off, the bare
   12 dB rule reads the oracle as presence -4.0 / -9.2 dB and this guard as 0.01;
b. the corpus tape noise is heavy-tailed per bin (on en `mid00_speech_m09` the 12 kHz bin's
   p90 sits 65 dB over its p10), so whole bands above the programme filled with noise cells
   no per-cell margin can stop. The quiet frames give each bin's noise pass rate; a band whose
   expected noise-only cells exceed 20 % of its cells (`MAX_NOISE_SHARE`) reads NaN. Without
   it, air moved p90 8.8 / 16.9 dB on speech. The quiet frames are picked on the 300-3000 Hz
   core level, so a noise burst elsewhere cannot lift a frame out of the set that measures
   it: the quietest 30 %, cut at 6 dB over the core's 10th percentile (`QUIET_FRAME_SPAN_DB`,
   the fixtures' +-3 dB floor wander). The 30 % alone read speech whose pauses fill under
   30 % of the frames (the sibilance tests' vowel-and-'s' fixture, 15 s) NaN in every band
   from 914 Hz up, identity included: vowel onsets and decays joined the set and their
   harmonics counted as noise; the cut reads its -1 dB/oct tilt -1.01. The quietest 10 %
   alone broke the music oracle (air p90 0.55 / 0.72 dB, up to 20 dB; tilt up to 10
   dB/oct): on en `mid00_music_m09` (9 dB margin) the noise leads the core in the quiet
   frames, so the quietest 10 % are where the noise itself is quietest and under-count it
   (estimated noise share 0.01-0.62 where it is 0.94-1.00). A noise-led core is flat within
   6 dB, so there the cut keeps the whole 30 % (unit test: HF bursts under a +-3 dB, 0.3 Hz
   floor wander read air None with the cut, the oracle's -20 dB with the quietest 10 %).

Shipped, the same oracle moves R1 by (p90 |move| at -10 / -20 dB): speech tilt 0.02 / 0.03
dB/oct, presence 0.01 / 0.01, body 0.03 / 0.04; music tilt 0.07 / 0.09, presence 0.09 / 0.12,
air 0.18 / 0.24, body 0.07 / 0.10. Readable windows: speech tilt 37/50, presence 34/50, body
30/50, air 1/50; music tilt 121/150, presence 127/150, air 67/150, body 6/150 (music's bass
never falls to a floor inside 15 s). A -1 dB/oct tilt above 1 kHz reads -1.00 (median) on
both; an RBJ +2 dB shelf at 7.5 kHz (Q 0.707) reads music air +1.27 (median, p90 +1.91) and
tilt +0.24. The fixtures' speech ends near 7 kHz, so there the shelf lifts hiss only: air is
unreadable and the tilt reads +0.02; on band-limited (PAL linear) speech the above-band
reading carries the shelf. On the synthetic harmonic programme of the unit tests every band
reads the shelf's own response within 0.1 dB. Two-sided by construction: the loop ranks the
distance from the reading of the file the user accepted, never the raw direction.

Limits. On speech R1 reads the voiced loud frames only and cannot see the 's': on the
sibilance tests' fixture no 's' frame is a loud frame (0 of 388), a 12 dB dulling of the 's'
top (6-12 kHz) reads 0.00 in every reading, and the +2 dB shelf reads presence +0.06 with air
unreadable. The +1 vs +2 dB air separation on speech must come from R2 (`sib_abs_level_db`)
and the above-band reading, not from R1. Leading digital silence no longer moves the oracle
(8 s of zeros: tilt -2.25 dB/oct and air -5.04 dB before the live-frame trim, under 0.1
after), but an analog mute, a capture floor some 40 dB under the tape hiss filling the
quietest 10 % of a window, still reads hiss removal as dulling (2-8 s at -90 dBFS: tilt
-1.4 to -2.3 dB/oct, air -3.1 to -5.1 dB): the floor and the quiet frames measure the mute,
not the hiss. No level rule inside one window separates it from a Hi-Fi pause, so the runner
skips such windows against the source's own floor (`runner.holds_mute`: 5 % of the live
20 ms frames 15 dB under `runner.mute_floor_db`, the median window p10 over seven programme
windows or more, which a mute up to 7.5 s cannot own, else the higher of that median and the
p25 of their live frames, so a mute sets it only when it fills more than half the windows
and a quarter of their span; a Hi-Fi tape's pauses are that floor). A floor that wanders while its
HF tail sits at the pass edge can still be under-counted (synthetic: +-3 dB at 0.5 Hz,
bursts 20 dB over the hiss, gaps half the frames: air reads the oracle's -20 dB, as with the
30 % alone). Dropping the 30 % cap fixes that but costs music most of its readings on
realistic-v2 (6 dB span: tilt 121 -> 43/150, air 67 -> 11/150; a 9 or 12 dB span:
air 0/150), so the cap stays.
"""

import numpy as np
import scipy.signal

from scripts.restoration_quality import audio_io, auditory, dsp_metrics

STFT_FRAME = 2048
STFT_HOP = 512
BAND_FLOOR_HZ = 100.0
BAND_CEILING_HZ = 16000.0
MIN_BAND_CELLS = 200
MIN_TILT_BANDS = 3
# Guard (b)'s quiet frames: the quietest 30 %, none more than 6 dB (the fixtures' +-3 dB floor wander) over p10.
QUIET_FRAME_PERCENTILE = 30.0
QUIET_FRAME_SPAN_DB = 6.0
MAX_NOISE_SHARE = 0.2
MIN_LEVEL_DBFS = -70.0
DYNAMIC_RANGE_DB = 90.0
# Ten quiet frames at the 10th percentile: fewer live frames cannot measure guard (b)'s pass rate.
MIN_LIVE_FRAMES = 100
# An output 60 dB under the source's programme is a dead render, not a restoration.
MIN_PROGRAMME_GAIN_DB = -60.0
NOISE_MARGIN_DB = 18.0
# One STFT cell of steady noise has exponential power: a bin's p10 is 0.105 x its mean (guard a).
NOISE_MEAN_OVER_FLOOR = 1.0 / -np.log1p(-auditory.PROGRAMME_FLOOR_PERCENTILE / 100.0)
TILT_FROM_HZ = 1000.0
PRESENCE_HZ = (2000.0, 5000.0)
AIR_HZ = (5000.0, np.inf)
BODY_HZ = (100.0, 300.0)
READINGS = ("balance_tilt_db_oct", "balance_presence_db", "balance_air_db", "balance_body_db")
# Each reading's band-centre span (`_tilt`, `_band_mean`), for `bands_under`.
READING_SPANS_HZ = {
    "balance_tilt_db_oct": (TILT_FROM_HZ, np.inf),
    "balance_presence_db": PRESENCE_HZ,
    "balance_air_db": AIR_HZ,
    "balance_body_db": BODY_HZ,
}
# The rate `bands_under` assumes when none is given: the band cap only meets Nyquist above 16 kHz anyway.
LAYOUT_RATE = 44100


def lag_aligned(raw_source, raw_output, lag):
    """The raw pair on one timeline, `align_pair`'s lag applied and no gain (lag > 0: the output runs late)."""
    source, output = np.asarray(raw_source), np.asarray(raw_output)
    if lag > 0:
        output = output[lag:]
    elif lag < 0:
        source = source[-lag:]
    length = min(len(source), len(output))
    return source[:length], output[:length]


def balance_readings(source, output, rate, bandwidth_hz=None):
    """`{reading: value}` for one window of a lag-aligned pair; a value is None where the data cannot carry it."""
    profile = band_profile(source, output, rate, bandwidth_hz)
    if profile is None:
        return dict.fromkeys(READINGS)
    centres, levels = profile
    return {
        "balance_tilt_db_oct": _tilt(centres, levels),
        "balance_presence_db": _band_mean(centres, levels, PRESENCE_HZ),
        "balance_air_db": _band_mean(centres, levels, AIR_HZ),
        "balance_body_db": _band_mean(centres, levels, BODY_HZ),
    }


def band_profile(source, output, rate, bandwidth_hz=None):
    """`(centres_hz, levels_db)` per ERB band after the programme gain match, or None when the window cannot be read.

    A band reads NaN with fewer than `MIN_BAND_CELLS` programme cells, or when the quiet frames
    say more than `MAX_NOISE_SHARE` of those cells would pass on noise alone.
    """
    spectra = _spectra(source, output, rate)
    if spectra is None:
        return None
    freqs, src_power, out_power = spectra
    over = _over_noise(src_power)
    cells = auditory.programme_cells(src_power) & over
    gain = auditory.programme_gain_db(src_power, out_power, freqs, cells)
    if gain is None or gain < MIN_PROGRAMME_GAIN_DB:
        return None
    ratio = auditory.cell_ratio_db(src_power, out_power) - gain
    edges = auditory.erb_band_edges(BAND_FLOOR_HZ, _ceiling_hz(rate, bandwidth_hz))
    levels = _band_levels(ratio, cells, _chance_cells(src_power, freqs, over), _band_index(freqs, edges), len(edges) - 1)
    return _centres_hz(edges), levels


def guarded_cells(src_power):
    """Step 2's programme cells on any STFT: `auditory.programme_cells` that also stand guard (a)'s noise margin.

    `band_profile` matches its gain on these; R11 (`lf_metrics`) takes the same gain match on
    its own frames.
    """
    return auditory.programme_cells(src_power) & _over_noise(src_power)


def _spectra(source, output, rate):
    """`(freqs, src_power, out_power)` over the live frames of the common length, or None when the pair cannot be read.

    A stereo side is downmixed the way the runner's pair is (`audio_io.to_mono`). Either side
    too short or under the silence level, or fewer than `MIN_LIVE_FRAMES` live frames, reads None.
    """
    source, output = audio_io.to_mono(np.asarray(source)), audio_io.to_mono(np.asarray(output))
    length = min(len(source), len(output))
    if not (_readable(source[:length]) and _readable(output[:length])):
        return None
    freqs, src_power = _power(source[:length], rate)
    live = _live_frames(src_power)
    if int(live.sum()) < MIN_LIVE_FRAMES:
        return None
    return freqs, src_power[:, live], _power(output[:length], rate)[1][:, live]


def _readable(mono):
    """At least one STFT frame long and above the silence level."""
    if len(mono) < STFT_FRAME:
        return False
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    return 20.0 * np.log10(rms + auditory.EPS) >= MIN_LEVEL_DBFS


def _live_frames(src_power):
    """The source frames within `DYNAMIC_RANGE_DB` of the loudest one: digital silence and padding are not a noise floor."""
    level = src_power.sum(axis=0)
    return level > level.max() * 10.0 ** (-DYNAMIC_RANGE_DB / 10.0)


def _power(mono, rate):
    """STFT power `(bins, frames)` (2048/512, Hann) and its bin frequencies."""
    freqs, _times, spec = scipy.signal.stft(
        np.asarray(mono, dtype=np.float64), fs=rate, window="hann", nperseg=STFT_FRAME, noverlap=STFT_FRAME - STFT_HOP
    )
    return freqs, np.abs(spec) ** 2


def _over_noise(src_power):
    """Cells (every frame) standing the margin over the bin's mean noise and within `DYNAMIC_RANGE_DB` of the loudest cell."""
    floor = np.percentile(src_power, auditory.PROGRAMME_FLOOR_PERCENTILE, axis=1, keepdims=True)
    over_noise = src_power >= floor * NOISE_MEAN_OVER_FLOOR * 10.0 ** (NOISE_MARGIN_DB / 10.0)
    return over_noise & (src_power > src_power.max() * 10.0 ** (-DYNAMIC_RANGE_DB / 10.0))


def _chance_cells(src_power, freqs, over):
    """Per bin, how many loud-frame cells would pass on noise alone: the quiet frames' pass rate times the loud frames.

    The quiet frames are picked on the 300-3000 Hz programme core, so noise bursts elsewhere
    cannot lift a frame out of the set that measures them.
    """
    core = (freqs >= auditory.GAIN_BAND_HZ[0]) & (freqs < auditory.GAIN_BAND_HZ[1])
    quiet = _quiet_frames(10.0 * np.log10(src_power[core].sum(axis=0) + auditory.EPS))
    return over[:, quiet].mean(axis=1) * int(auditory.loud_frames(src_power).sum())


def _quiet_frames(level_db):
    """The quietest frames by core level: at most the quietest 30 %, none more than 6 dB over the 10th percentile.

    The cut keeps vowel onsets and decays out of short speech pauses; a flat, noise-led core
    (low-margin music) keeps the whole 30 %, where the quietest 10 % would under-count the noise.
    """
    floor_db = np.percentile(level_db, auditory.PROGRAMME_FLOOR_PERCENTILE)
    return level_db <= min(np.percentile(level_db, QUIET_FRAME_PERCENTILE), floor_db + QUIET_FRAME_SPAN_DB)


def _ceiling_hz(rate, bandwidth_hz):
    """The top of the read range: min(bandwidth, 16 kHz, Nyquist), never below the 100 Hz floor."""
    ceiling = min(BAND_CEILING_HZ, rate / 2.0)
    if bandwidth_hz:
        ceiling = min(ceiling, float(bandwidth_hz))
    return max(ceiling, BAND_FLOOR_HZ)


def _band_index(freqs, edges):
    """`auditory.band_index` with the CRT line bins (`dsp_metrics.band_bins`) marked outside every band."""
    index = auditory.band_index(freqs, edges)
    index[~dsp_metrics.band_bins(freqs, (0.0, np.inf))] = -1
    return index


def _band_levels(ratio, cells, chance, index, count):
    """`_band_level` for each of the `count` bands of `index`."""
    return np.array([_band_level(ratio, cells, chance, index == band) for band in range(count)])


def _band_level(ratio, cells, chance, rows):
    """The median of `ratio` over the programme cells of the bins in `rows`; NaN with too few or noise-led cells."""
    values = ratio[rows][cells[rows]]
    if values.size < MIN_BAND_CELLS or chance[rows].sum() > MAX_NOISE_SHARE * values.size:
        return np.nan
    return float(np.median(values))


def _centres_hz(edges):
    """Each band's centre: the frequency halfway between its edges on the ERB-number scale."""
    erb = auditory.erb_number(edges)
    return auditory.erb_to_hz((erb[:-1] + erb[1:]) / 2.0)


def bands_under(reading, bandwidth_hz, rate=LAYOUT_RATE):
    """How many of `band_profile`'s ERB bands under a `bandwidth_hz` cap centre inside `reading`'s span.

    The runner caps R1 at R0's programme band. The layout puts its last band's centre under the
    cap, so a reading's first band appears above the span's start (`readable_from_hz`): air
    (from 5 kHz) has none under a 5.2 kHz cap and its first under 5292 Hz, presence its first
    under 2125 Hz, the tilt its third under 1408 Hz.
    """
    centres = _centres_hz(auditory.erb_band_edges(BAND_FLOOR_HZ, _ceiling_hz(rate, bandwidth_hz)))
    low, high = READING_SPANS_HZ[reading]
    return int(np.count_nonzero((centres >= low) & (centres < high)))


def readable_under(reading, bandwidth_hz, rate=LAYOUT_RATE):
    """Whether the layout under the cap carries `reading` at all: one band, three for the tilt's slope."""
    needed = MIN_TILT_BANDS if reading == "balance_tilt_db_oct" else 1
    return bands_under(reading, bandwidth_hz, rate) >= needed


def readable_from_hz(reading, rate=LAYOUT_RATE):
    """The lowest band cap, to 1 Hz, under which the layout carries `reading` (`readable_under`)."""
    low, high = BAND_FLOOR_HZ, _ceiling_hz(rate, None)
    while high - low > 1.0:
        middle = (low + high) / 2.0
        low, high = (low, middle) if readable_under(reading, middle, rate) else (middle, high)
    return high


def _band_mean(centres, levels, span):
    """The mean level of the readable bands whose centre lies in `span`, None when there is none."""
    pick = (centres >= span[0]) & (centres < span[1]) & np.isfinite(levels)
    return float(np.mean(levels[pick])) if pick.any() else None


def _tilt(centres, levels):
    """The least-squares slope in dB per octave over the readable bands from 1 kHz up, None with fewer than 3."""
    pick = (centres >= TILT_FROM_HZ) & np.isfinite(levels)
    if int(pick.sum()) < MIN_TILT_BANDS:
        return None
    return float(np.polyfit(np.log2(centres[pick]), levels[pick], 1)[0])
