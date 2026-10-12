"""The background's worst octave read on the programme it leads: hiss removal and a level change read nothing, stripping reads.

The stems are synthetic and the separation perfect (the stem is the bed with its hiss), so these
tests hold the reading, not BS-RoFormer; the real separator's checks are the GPU follow-up.
"""

import numpy as np
import pytest

from scripts import quality_degradations as deg
from scripts.restoration_quality import audio_io, stem_metrics, stem_octave
from scripts.restoration_quality.lf_metrics import stft_power

RATE = 44100
SECONDS = 15.0
HISS_DBFS = -45.0


def _chord(rng, seconds=0.4):
    """Three random harmonic notes with one attack and decay."""
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = np.minimum(1.0, t / 0.02) * np.exp(-t / 0.25)
    notes = [
        0.02 / k * np.sin(2 * np.pi * root * k * t + rng.uniform(0, 2 * np.pi))
        for root in rng.uniform(110.0, 440.0, 3)
        for k in range(1, 12)
    ]
    return envelope * np.sum(notes, axis=0)


def _bed(seed=3):
    """A dense, moving music bed: a new chord every 0.4 s."""
    rng = np.random.default_rng(seed)
    return np.concatenate([_chord(rng) for _chord_index in range(int(SECONDS / 0.4) + 1)])[: int(SECONDS * RATE)]


def _hiss(seed=5, dbfs=HISS_DBFS):
    """White noise at `dbfs` RMS for the whole window."""
    return 10.0 ** (dbfs / 20.0) * np.random.default_rng(seed).standard_normal(int(SECONDS * RATE))


def _voice(seed=11):
    """A gated harmonic voice: half a second on, half a second off, so the bed leads the pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SECONDS * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * 150.0 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 30))
    return 0.2 * tone * ((t % 1.0) < 0.5)


def _read(stem, mix_source, mix_output):
    return stem_octave.octave_programme_db(*(np.asarray(x, dtype=np.float32) for x in (stem, mix_source, mix_output)), RATE)


def _raw(source, output):
    """The raw worst octave on the least-squares matched pair, as `stem_metrics._stems` feeds it."""
    src, out, _lag = audio_io.align_pair(np.asarray(source, dtype=np.float32), np.asarray(output, dtype=np.float32))
    return stem_metrics.octave_ratio_db(src, out, RATE)


@pytest.fixture(name="bed")
def fixture_bed():
    """The moving music bed of every test."""
    return _bed()


@pytest.fixture(name="hiss")
def fixture_hiss():
    """White tape hiss at -45 dBFS."""
    return _hiss()


def test_removing_the_hiss_reads_no_loss_where_the_raw_octave_read_stripping(bed, hiss):
    """The oracle restoration (hiss 20 dB down): the raw octave reads the hiss-only top octave as lost, the new one nothing."""
    source, output = bed + hiss, bed + 0.1 * hiss
    assert _raw(source, output) < -10.0
    assert _read(source, source, output) == pytest.approx(0.0, abs=0.3)


def test_a_planted_octave_mute_fires_with_the_hiss_kept_or_removed(bed, hiss):
    """The bed's 500-1000 Hz taken out reads far under the -3 dB gate, whether the restoration kept the hiss or not."""
    gone = deg.band_gain(bed, RATE, (500.0, 1000.0), -60.0)
    assert _read(bed + hiss, bed + hiss, gone + hiss) < -10.0
    assert _read(bed + hiss, bed + hiss, gone + 0.1 * hiss) < -10.0


def _calibration_case(name, level, bed, hiss):
    """The reading on one calibration stems case as `quality_degradations` builds it, the source as its own stem."""
    source, output = deg.apply(name, level, bed, RATE, {deg.STEM_NOISE: hiss}, None)
    return _read(source, source, output)


def test_the_calibrations_planted_mute_reads_deeper_at_every_level(bed, hiss):
    """`stem_octave_muted` at -6 / -12 / -24 dB reads ever lower, under the -3 dB gate from -12 dB."""
    readings = [_calibration_case("stem_octave_muted", level, bed, hiss) for level in deg.STEM_MUTE_LEVELS]
    assert readings[0] > readings[1] > readings[2]
    assert readings[1] < -3.0


def test_the_calibrations_oracle_stays_flat(bed, hiss):
    """`music_hiss_removed` at its severest level stays inside the check's 0.5 dB."""
    assert abs(_calibration_case("music_hiss_removed", deg.STEM_HISS_LEVELS[-1], bed, hiss)) <= deg.STEM_OCTAVE_TOLERANCE_DB


@pytest.mark.parametrize("gain_db", [-6.0, -12.0, -24.0])
def test_a_background_stripped_under_a_voice_reads_its_loss(bed, hiss, gain_db):
    """The voice holds the mix's level, so the bed's loss reads in full (the least-squares stem match cancelled most of it)."""
    voice = _voice()
    stripped = voice + 10.0 ** (gain_db / 20.0) * bed + 0.1 * hiss
    assert _read(bed + hiss, voice + bed + hiss, stripped) == pytest.approx(gain_db, abs=2.0)


def test_a_music_only_programme_turned_down_is_a_level_change(bed, hiss):
    """With no voice the background is the programme: turning all of it down reads 0, not stripping."""
    source = bed + hiss
    assert _read(source, source, 0.25 * source) == pytest.approx(0.0, abs=0.3)


def test_a_sustained_chord_is_the_blind_spot(hiss):
    """A chord that never falls to a floor in the window has no cell over its own p10: unread, stripped or not."""
    t = np.arange(int(SECONDS * RATE)) / RATE
    chord = sum(0.05 / k * np.sin(2 * np.pi * 220.0 * k * t + k) for k in range(1, 25))
    source = chord + hiss
    assert _read(source, source, chord + 0.1 * hiss) is None
    assert _read(source, source, deg.band_gain(chord, RATE, (500.0, 1000.0), -60.0) + hiss) is None


def test_a_window_too_short_or_too_empty_reads_nothing(bed):
    """Under one STFT frame, or under a hundred live frames, there is nothing to read."""
    short = bed[: stem_octave.FRAME - 1]
    assert _read(short, short, short) is None
    sparse = np.zeros(int(1.0 * RATE))
    assert _read(sparse, sparse, sparse) is None


def _programme_power(seed=2):
    """A `(bins, frames)` power whose cells swing far over each bin's floor: programme everywhere it is loud."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(-12.0, 0.0, (64, 400)))


def test_a_cell_the_background_does_not_lead_is_the_voices():
    """The stem must hold the mix within 1 dB: at -3 dB (half the cell is something else) no cell counts, at -0.5 dB they do."""
    stem = _programme_power()
    assert not stem_octave.background_cells(stem, stem / 0.5).any()
    assert stem_octave.background_cells(stem, stem / 10.0 ** (-0.05)).any()


def test_an_octave_far_under_the_loudest_is_left_out():
    """An octave whose cells hold under a 1/10000 share of the loudest octave's energy reads None (R11's share)."""
    freqs = np.linspace(0.0, RATE / 2.0, 1025)
    src = np.full((1025, 400), 1e-12)
    src[(freqs >= 250.0) & (freqs < 500.0)] = 1.0
    src[(freqs >= 2000.0) & (freqs < 4000.0)] = 1e-6
    cells = src > 1e-9
    levels = stem_octave.octave_levels(src, 0.5 * src, freqs, cells, np.zeros(1025))
    assert levels[1] == pytest.approx(-3.0103, abs=1e-3)
    assert levels[4] is None


def test_noise_led_octaves_and_thin_octaves_read_none():
    """Guard (b) over a fifth of the cells, or fewer than MIN_OCTAVE_CELLS cells, leaves the octave unread."""
    freqs = np.linspace(0.0, RATE / 2.0, 1025)
    src = np.ones((1025, 400))
    cells = np.zeros_like(src, dtype=bool)
    cells[(freqs >= 250.0) & (freqs < 500.0)] = True
    cells[(freqs >= 500.0) & (freqs < 1000.0), :1] = True
    chance = np.where((freqs >= 250.0) & (freqs < 500.0), 1e6, 0.0)
    levels = stem_octave.octave_levels(src, src, freqs, cells, chance)
    assert levels[1] is None
    assert levels[2] is None


def test_without_a_readable_mix_octave_the_background_levels_against_itself(bed, hiss, monkeypatch):
    """No mix gain: the worst octave reads against the median of the background's own octaves."""
    monkeypatch.setattr(stem_octave, "mix_gain_db", lambda *_args: None)
    source = bed + hiss
    gone = deg.band_gain(bed, RATE, (500.0, 1000.0), -60.0)
    assert _read(source, source, 0.5 * source) == pytest.approx(0.0, abs=0.3)
    assert _read(source, source, gone + hiss) < -10.0


def _sine_cells(dbfs, bin_index=100, seconds=1.0):
    """The STFT power of a sine at `dbfs` (its amplitude re full scale) centred on one bin."""
    t = np.arange(int(seconds * RATE)) / RATE
    tone = 10.0 ** (dbfs / 20.0) * np.sin(2 * np.pi * bin_index * RATE / stem_octave.FRAME * t)
    return stft_power(tone, RATE, stem_octave.FRAME, stem_octave.HOP)[1][bin_index, 4:-4]


@pytest.mark.parametrize(("dbfs", "over"), [(-89.0, True), (-91.0, False), (0.0, True)])
def test_the_floor_reads_a_cell_as_the_sine_that_fills_it(dbfs, over):
    """A sine centred on a bin reads its own level there: -89 dBFS clears the -90 dBFS floor, -91 does not."""
    assert bool(stem_octave.over_floor(_sine_cells(dbfs)).all()) is over


def test_a_full_scale_sine_puts_the_floors_reference_power_in_its_bin():
    assert float(np.median(_sine_cells(0.0))) == pytest.approx(stem_octave.FULL_SCALE_SINE_POWER, rel=1e-6)


# White noise under the band-limited tape's empty top octave: 31 dB under a -94 dBFS tone's bin.
DITHER_DBFS = -100.0


def _band_limited_tape(tone_dbfs):
    """`(source, output)` of a quiet band-limited tape: the bed and its hiss end at 5 kHz, white noise at DITHER_DBFS above.

    Over that clean top octave, tones at `tone_dbfs` in 8-16 kHz play from 30 to 130 ms into each 0.4 s chord (loud
    frames, never the quiet ones guard (b) counts on), so their bins fall to the noise between chords and the cells
    stand 18 dB or more over the bins' floor: guard (a) passes them at -94 dBFS. The output is the oracle
    restoration (hiss 20 dB down) without the tones.
    """
    t = np.arange(int(SECONDS * RATE)) / RATE
    tones = sum(np.sin(2 * np.pi * hz * t) for hz in (9000.0, 10500.0, 12000.0, 13500.0)) * 10.0 ** (tone_dbfs / 20.0)
    hiss = deg.band_gain(_hiss(), RATE, (5000.0, RATE / 2.0), -120.0)
    dither = 10.0 ** (DITHER_DBFS / 20.0) * np.random.default_rng(13).standard_normal(len(t))
    bed = 0.05 * _bed()
    gate = (t % 0.4 >= 0.03) & (t % 0.4 < 0.13)
    return bed + 0.05 * hiss + dither + tones * gate, bed + 0.005 * hiss + dither


def test_et_els_case_cells_at_minus_94_dbfs_no_longer_read_as_a_loss():
    """Cells at -94 dBFS (et-el's level as the octave investigation gave it) that the output leaves out read no loss."""
    source, output = _band_limited_tape(-94.0)
    assert _read(source, source, output) == pytest.approx(0.0, abs=0.3)


def test_without_the_floor_those_cells_read_as_a_deep_loss(monkeypatch):
    """The same pair with guard (c) off: the empty octave reads about -29 dB, far under the -3 dB gate."""
    monkeypatch.setattr(stem_octave, "CELL_FLOOR_DBFS", -np.inf)
    source, output = _band_limited_tape(-94.0)
    assert _read(source, source, output) < -20.0


def test_a_quiet_background_over_the_floor_still_reads_its_loss():
    """The same tones at -60 dBFS are programme: leaving them out reads about -48 dB."""
    source, output = _band_limited_tape(-60.0)
    assert _read(source, source, output) < -20.0


@pytest.mark.parametrize(
    ("input_peak", "stem_peak", "scale"),
    [(0.5, 0.4, 1.0), (1.8, 0.6, 2.0), (1.8, 0.9, None), (0.7, 0.90003, None)],
)
def test_the_stem_comes_back_to_its_inputs_level_unless_the_separator_capped_it(input_peak, stem_peak, scale):
    """audio-separator caps the input mix and each stem at a 0.9 peak; the input's cap is undone, a stem's cannot be."""
    assert stem_octave.stem_scale(input_peak, stem_peak) == (None if scale is None else pytest.approx(scale))
