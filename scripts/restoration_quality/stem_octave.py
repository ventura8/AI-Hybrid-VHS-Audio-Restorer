"""The background's worst octave, read where the background leads the programme (`stems.octave_programme_db`).

Why (E1, 2026-10-12). The raw worst octave of the separated non-vocal stem
(`stem_metrics.octave_ratio_db`, now displayed only) read the denoising itself as a stripped
background, two ways:

1. Hiss. On tape the instrumental stem carries the hiss, so an octave the background never
   filled reads as lost once the hiss is gone. The APL final of `x_vid-20230606-121054` that
   the user accepted ("music fine on both", ledger record `r2-x_vid-20230606-121054-music-fine`)
   reads -38.8 dB; 13 of the 22 accepted v2 music finals the cached stems can read fail the
   -3 dB gate on the raw octave.
2. Phase. `stem_metrics._stems` matches the output stem to the source stem by least squares,
   a waveform measure: a restoration that keeps the magnitudes and moves the phase (cathar's
   STFT stages and its dewind) is scaled down as a whole. On the hunt's shipped cathar outputs
   the scale was -10.8 to -18.2 dB below the magnitude match, and every octave read 3-9 dB down.

The reading, per window, on three STFTs of one grid (2048 / 512 Hann, the stem's rate): the
source stem `S`, the source mix `M` and the output mix `O`, on the source's timeline:

- cells: the source stem's programme cells (`auditory.programme_cells`: loud frames, bin 12 dB
  over its p10 floor) that stand `NOISE_MARGIN_DB` over the stem's bias-corrected mean noise
  (`balance_metrics.over_noise`, guard a) and where the background leads the mix, `S` within
  `LEAD_DB` of `M`: at most a fifth of such a cell can be anything else, so removing all of
  it moves the mix by 1 dB at most;
- per octave of `OCTAVES_HZ` (125 Hz up, the CRT line and its skirt left out), the median
  output-over-source dB of the MIX over those cells, when the octave holds `MIN_OCTAVE_CELLS`
  and guard (b) (`balance_metrics.chance_cells`) expects no more than `MAX_NOISE_SHARE` of
  them to pass on noise alone, and when its cells hold at least a 1/10000 share
  (`SHARE_DB`, R11's rule) of the loudest such octave's source energy;
- less the mix's own level change: the median over octaves of the same reading on the mix's
  guarded programme cells (each octave one vote, so one stripped octave cannot move it), or,
  where the mix has no readable octave, the median of the background's own octaves;
- the worst octave, floored at -60 dB; None where no octave reads.

The output is never separated for this reading, so a separator that splits a cleaner output
differently (sibilants and room tone moving to the vocal stem) cannot read as stripping, and
the least-squares scale is gone. Removing hiss moves a cell standing the margin over the
noise by at most 10 log10(1 + 10^(-margin/10)) dB. A background stripped under a voice reads
its loss, since the voice holds the mix's level; a music-only programme turned down as a
whole is a level change and reads 0, while one stripped octave reads against the others.

The source stem must be level-true against its mix for the lead test: audio-separator scales
its input mix and each stem down to a 0.9 peak when they exceed it (`SEPARATOR_PEAK`).
`stem_metrics` undoes the mix's scaling from the input file's peak and reads None when the
stem itself was scaled (its peak sits at 0.9: 9 of the hunt's 235 sources, both v2 music
finals of `x_the-adventures-of-captain-pu`).

Measured on CPU (2026-10-12) from cached BS-RoFormer stems, no separator run:

- The 24 v2 music finals the user called "music fine" (round 2; each WAV checked against the
  ledger's sha256), from the v6 music loop's cached source stems: 22 read and none falls
  under -3 dB, the worst -2.42 (APL x_videoplayback-20260504-2342) and x_vid-20230606-121054
  APL -0.45; the raw octave failed 13 of the 21 it read. Both captain-pu finals are unread.
- `background_stripped` on the 2026-09-20 calibration's own stems (en): -5.9 / -11.1 /
  -14.9 dB at -6 / -12 / -24 dB, where the raw octave read -0.9 / -2.6 / -14.3, so it now
  fires at every level, the raw octave at the last only.
- realistic-v2 musiconly beds with their recorded noise, the bed as its own stem (50
  fixtures, 100 windows): the oracle (noise 20 dB down) fails 0 of 94 read windows (1 of 80
  under a voice). A planted mute of 250-500, 500-1000 or 1000-2000 Hz fails every window where
  that octave reads (17 of 17, 36 of 36, 65 of 69; under a voice 12 of 12, 32 of 32, 46 of 46),
  but the octave reads on only 17, 36 and 69 of the 100 windows (the blind spot below). The bed
  stripped 6 dB under a voice fails 75 of 80.
- The hunt's shipped outputs (235 IA clips per engine): cathar fails 66 of the 178 clips that
  read (raw octave 157 of 166), APL 18 of 178 (raw 126 of 192); 9 sources are capped. cathar's
  failing windows sit in every octave at -3 to -6 dB, APL's mostly above 4 kHz; not traced.

Blind spot: a background that never falls to a floor inside the 15 s window. Guard (a)'s
floor is the bin's own p10, which on a steady or slowly moving bed is the bed itself, so the
bed's cells never stand the margin over it and the octave is unread: a sustained 24-partial
chord over hiss, steady or swinging 20 dB at 0.5 Hz, reads None on every octave, stripped or
not, where the raw octave read -20 dB on hiss removal alone. Such a window is skipped by the
gate, never failed; the raw octave stays on the scorecard for it.
"""

import numpy as np

from scripts.restoration_quality import auditory, balance_metrics, dsp_metrics
from scripts.restoration_quality.lf_metrics import stft_power

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
FRAME = balance_metrics.STFT_FRAME
HOP = balance_metrics.STFT_HOP
NOISE_MARGIN_DB = balance_metrics.NOISE_MARGIN_DB
LEAD_DB = 1.0
MIN_OCTAVE_CELLS = 200
MAX_NOISE_SHARE = balance_metrics.MAX_NOISE_SHARE
SHARE_DB = 40.0
FLOOR_DB = -60.0
MIN_LIVE_FRAMES = balance_metrics.MIN_LIVE_FRAMES
LIVE_RANGE = 10.0 ** (-balance_metrics.DYNAMIC_RANGE_DB / 10.0)
SEPARATOR_PEAK = 0.9
# A stem whose peak sits this close to SEPARATOR_PEAK was scaled down by the separator.
CAPPED_TOLERANCE = 1e-4


def stem_scale(input_peak, stem_peak):
    """The factor that brings a separated stem back to its input's level, or None when the stem itself was capped.

    audio-separator divides the input mix by its peak over 0.9 before separating, and a stem
    by its own peak over 0.9 before writing it; the second factor is lost with the stem's peak.
    """
    if abs(float(stem_peak) - SEPARATOR_PEAK) < CAPPED_TOLERANCE:
        return None
    return max(1.0, float(input_peak) / SEPARATOR_PEAK)


def octave_programme_db(stem_source, mix_source, mix_output, rate):
    """The worst octave of the background, in dB, read on the mixes where the background leads; None where none reads."""
    spectra = _spectra(stem_source, mix_source, mix_output, rate)
    if spectra is None:
        return None
    freqs, stem, src, out = spectra
    own = _read(octave_levels(src, out, freqs, background_cells(stem, src), _chance(stem, freqs)))
    if not own:
        return None
    gain = mix_gain_db(src, out, freqs)
    return max(FLOOR_DB, min(own) - float(np.median(own if gain is None else [gain])))


def _read(levels):
    """The octaves that read."""
    return [level for level in levels if level is not None]


def background_cells(stem, src):
    """The source stem's guarded programme cells where the stem holds the mix within `LEAD_DB`."""
    guarded = auditory.programme_cells(stem) & balance_metrics.over_noise(stem, NOISE_MARGIN_DB)
    return guarded & (stem >= src * 10.0 ** (-LEAD_DB / 10.0))


def mix_gain_db(src, out, freqs):
    """The mix's own level change: the median over its readable octaves of the output-over-source dB on its programme cells."""
    over = balance_metrics.over_noise(src, NOISE_MARGIN_DB)
    read = _read(octave_levels(src, out, freqs, auditory.programme_cells(src) & over, _chance(src, freqs)))
    return float(np.median(read)) if read else None


def octave_levels(src, out, freqs, cells, chance):
    """Per octave, the median output-over-source dB over `cells`, or None (few cells, noise-led, or under the share)."""
    found = [_octave(src, out, cells, chance, dsp_metrics.band_bins(freqs, band)) for band in OCTAVES_HZ]
    least = max((energy for _level, energy in filter(None, found)), default=0.0) * 10.0 ** (-SHARE_DB / 10.0)
    return [_over_share(item, least) for item in found]


def _over_share(item, least):
    """An octave's level when it read and its cells hold at least `least` of the source's energy, else None."""
    return item[0] if item and item[1] >= least else None


def _octave(src, out, cells, chance, rows):
    """`(median dB, source energy)` over the octave's cells, or None with too few or noise-led cells."""
    mask = cells[rows]
    count = int(mask.sum())
    if count < MIN_OCTAVE_CELLS or float(chance[rows].sum()) > MAX_NOISE_SHARE * count:
        return None
    src_cells, out_cells = src[rows][mask], out[rows][mask]
    return float(np.median(auditory.cell_ratio_db(src_cells, out_cells))), float(src_cells.sum())


def _chance(power, freqs):
    """Guard (b): per bin, the loud-frame cells expected to pass the margin on noise alone."""
    return balance_metrics.chance_cells(power, freqs, balance_metrics.over_noise(power, NOISE_MARGIN_DB))


def _spectra(stem_source, mix_source, mix_output, rate):
    """`(freqs, stem, src, out)` power on the mix's live frames of the common length, or None when too short or too empty."""
    length = min(len(stem_source), len(mix_source), len(mix_output))
    if length < FRAME:
        return None
    freqs, src = stft_power(mix_source[:length], rate, FRAME, HOP)
    level = src.sum(axis=0)
    live = level > level.max() * LIVE_RANGE
    if int(live.sum()) < MIN_LIVE_FRAMES:
        return None
    stem = stft_power(stem_source[:length], rate, FRAME, HOP)[1]
    out = stft_power(mix_output[:length], rate, FRAME, HOP)[1]
    return freqs, stem[:, live], src[:, live], out[:, live]
