"""R11, the programme's low end: a low shelf or a high pass reads in order, a gain, hiss or hum removal reads nothing."""

import functools

import numpy as np
import pytest
import scipy.signal

from scripts import degradations_v3 as v3
from scripts.restoration_quality import auditory, balance_metrics
from scripts.restoration_quality import lf_metrics as lf

RATE = 44100
SECONDS = 15.0
SHELF_GAINS_DB = (-6.0, -3.0, -1.5, 1.5, 3.0, 6.0)
CORNERS_HZ = (40.0, 60.0, 80.0)


@functools.lru_cache(maxsize=None)
def _music(seed=3, f0=45.0, top_hz=3000.0, hiss=1e-3):
    """`(clean, noise)`: harmonics of 45 Hz up to 3 kHz in 0.5 s bursts and a steady hiss under them.

    The 45 Hz series puts four harmonics in 40-200 Hz (45, 90, 135, 180) and none within a
    Hann main lobe of either band edge, so a gain on exactly 40-200 Hz reads at its own size.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(SECONDS * RATE)) / RATE
    count = int(top_hz // f0)
    phases = rng.uniform(0.0, 2 * np.pi, count + 1)
    tone = sum(np.sin(2 * np.pi * f0 * k * t + phases[k]) / np.sqrt(k) for k in range(1, count + 1))
    ramp = np.hanning(int(0.02 * RATE))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    return (0.02 * tone * gate).astype(np.float32), (hiss * rng.standard_normal(len(t))).astype(np.float32)


def _source():
    clean, noise = _music()
    return clean + noise


def _band_gain(mono, gain_db, band=lf.LF_BAND_HZ):
    """`mono` with exactly `band` scaled by `gain_db` (one zero-phase FFT of the whole signal)."""
    return v3.fft_filter(mono, RATE, lambda freqs: np.where((freqs >= band[0]) & (freqs < band[1]), gain_db, 0.0))


def _bandpassed_noise(low, high, level, seed=21):
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    noise = np.random.default_rng(seed).standard_normal(int(SECONDS * RATE))
    return (level * scipy.signal.sosfiltfilt(sos, noise)).astype(np.float32)


def _hum(mains_hz=50.0, level=0.01, harmonics=8):
    """A steady hum series, each harmonic at `level / k`."""
    t = np.arange(int(SECONDS * RATE)) / RATE
    return (sum(level / k * np.sin(2 * np.pi * mains_hz * k * t + 0.3 * k) for k in range(1, harmonics + 1))).astype(np.float32)


def test_identity_and_a_broadband_gain_read_flat():
    """The identity reads 0 and any broadband gain cancels in the programme gain match."""
    source = _source()
    assert abs(lf.lf_programme_db(source, source, RATE)) < 1e-9
    assert abs(lf.lf_programme_db(source, 0.3 * source, RATE)) < 1e-3


def test_a_gain_on_exactly_the_band_reads_at_its_own_size():
    """The low end 6 dB down and nothing else: R11 reads -6 dB, with the mains lines left out or not."""
    source = _source()
    output = _band_gain(source, -6.0)
    assert abs(lf.lf_programme_db(source, output, RATE) + 6.0) < 0.05
    assert abs(lf.lf_programme_db(source, output, RATE, mains_hz=50.0) + 6.0) < 0.05


def test_a_low_shelf_reads_in_order_of_its_gain():
    """FFmpeg's `bass` shelf at 100 Hz: the reading follows its gain in order and sign, inside the shelf's own span."""
    source = _source()
    readings = [lf.lf_programme_db(source, v3.lf_shelf(source, RATE, gain), RATE) for gain in SHELF_GAINS_DB]
    assert all(low < high for low, high in zip(readings, readings[1:]))
    assert all(np.sign(reading) == np.sign(gain) for reading, gain in zip(readings, SHELF_GAINS_DB))
    assert all(abs(reading) < abs(gain) for reading, gain in zip(readings, SHELF_GAINS_DB))


def test_a_high_pass_reads_in_order_of_its_corner():
    """Round C3's dewind corners: 80 Hz thins the low end more than 60 Hz, 60 Hz more than 40 Hz."""
    source = _source()
    readings = [lf.lf_programme_db(source, v3.lf_highpass(source, RATE, corner), RATE) for corner in CORNERS_HZ]
    assert 0.0 > readings[0] > readings[1] > readings[2]
    assert readings[2] < -3.0


def test_oracle_hiss_removal_moves_it_little():
    """White hiss 20 dB down: the band's hiss is a small share of the loud frames' low end."""
    clean, noise = _music(hiss=3e-3)
    for scale in (0.316, 0.1):
        assert abs(lf.lf_programme_db(clean + noise, clean + scale * noise, RATE)) < 0.1


def test_rumble_removed_reads_as_less_low_end_and_rumble_left_reads_none_removed():
    """Noise in 40-80 Hz under the loud frames counts: its removal reads negative, the rumble left reads 0."""
    clean, noise = _music()
    rumble = _bandpassed_noise(40.0, 80.0, 0.3)
    source = clean + noise + rumble
    assert lf.lf_programme_db(source, clean + noise, RATE) < -0.5
    assert abs(lf.lf_programme_db(source, source + 0.0, RATE)) < 1e-9


def test_a_removed_hum_series_reads_nothing_once_its_mains_lines_are_left_out():
    """Dehum is not a thinner bass: with the mains named its lines leave the band, without it their removal reads."""
    clean, noise = _music()
    source = clean + noise + _hum()
    assert abs(lf.lf_programme_db(source, clean + noise, RATE, mains_hz=50.0)) < 0.05
    assert lf.lf_programme_db(source, clean + noise, RATE) < -0.5


def test_a_cut_of_40_to_60_hz_reads_with_the_50_hz_mains_left_out():
    """Round C3's band: 40-60 Hz 20 dB down reads -2.76 dB with a 50 Hz mains named, as without one (-2.78).

    The 8192-sample frame took 39-61 Hz out round 50 Hz and read the same cut -0.00002 dB:
    the 45 Hz fundamental under the cut sat inside the exclusion. At 1.35 Hz bins it sits
    5 Hz from the line, outside its +-2.7 Hz.
    """
    source = _source()
    output = _band_gain(source, -20.0, band=(40.0, 60.0))
    named, unnamed = (lf.lf_programme_db(source, output, RATE, mains) for mains in (50.0, None))
    assert named < -2.0
    assert abs(named - unnamed) < 0.1


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_the_dewind_corners_read_in_order_with_a_mains_left_out(mains):
    """40 / 60 / 80 Hz read in order with either mains named: -0.94 / -2.29 / -3.45 dB at 50 Hz (the old frame -0.06 / -0.27 / -0.71)."""
    source = _source()
    readings = [lf.lf_programme_db(source, v3.lf_highpass(source, RATE, corner), RATE, mains) for corner in CORNERS_HZ]
    assert -0.5 > readings[0] > readings[1] > readings[2]
    assert readings[1] - readings[2] > 0.5


@pytest.mark.parametrize(("rate", "frame"), [(22050, 16384), (44100, 32768), (48000, 32768), (96000, 65536)])
def test_the_band_frame_holds_about_1_35_hz_bins_at_any_rate(rate, frame):
    """The band STFT's frame follows the rate (bins of 1.35-1.47 Hz), its hop a quarter frame; the gain STFT stays 8192 / 2048."""
    assert lf.frame_length(rate) == frame
    assert 1.3 < rate / frame < 1.5
    assert lf.hop_length(rate) == frame // 4
    assert (lf.GAIN_FRAME, lf.GAIN_HOP) == (8192, 2048)


def test_at_96_khz_the_band_keeps_its_bins_round_a_50_hz_mains():
    """At 96 kHz the old frame kept 1 band bin of 14 round 50 Hz; the band frame keeps 95 and reads a band gain at its size."""
    source = scipy.signal.resample_poly(_source(), 320, 147).astype(np.float32)
    output = v3.fft_filter(source, 96000, lambda freqs: np.where((freqs >= 40.0) & (freqs < 200.0), -6.0, 0.0))
    assert int(lf.lf_bins(_stft_freqs(96000), 96000, 50.0).sum()) >= 90
    assert abs(lf.lf_programme_db(source, output, 96000, mains_hz=50.0) + 6.0) < 0.05


def test_a_band_the_mains_lines_empty_reads_none(monkeypatch):
    """Under `MIN_LF_BINS` bins left once the mains lines are out: no band to judge (a 27 Hz exclusion empties it)."""
    source = _source()
    output = v3.lf_shelf(source, RATE, -3.0)
    monkeypatch.setattr(lf, "HUM_HALF_WIDTH_BINS", 20.0)
    assert int(lf.lf_bins(_stft_freqs(), RATE, 50.0).sum()) < lf.MIN_LF_BINS
    assert lf.lf_programme_db(source, output, RATE, mains_hz=50.0) is None
    assert lf.lf_programme_db(source, output, RATE) < 0.0


def _stft_freqs(rate=RATE):
    return np.fft.rfftfreq(lf.frame_length(rate), 1.0 / rate)


def test_without_a_mains_the_band_is_every_bin_of_40_to_200_hz():
    """No mains named (None or 0): nothing leaves the band."""
    freqs = _stft_freqs()
    band = (freqs >= 40.0) & (freqs < 200.0)
    assert np.array_equal(lf.lf_bins(freqs, RATE), band)
    assert np.array_equal(lf.lf_bins(freqs, RATE, 0.0), band)


@pytest.mark.parametrize(("mains", "lines"), [(50.0, (50.0, 100.0, 150.0, 200.0)), (60.0, (60.0, 120.0, 180.0))])
def test_the_band_leaves_out_each_mains_line_within_a_main_lobe(mains, lines):
    """The bins of 40-200 Hz, less exactly those within +-2 bins (2.7 Hz at 44.1 kHz) of a harmonic of the mains named."""
    freqs = _stft_freqs()
    width = lf.HUM_HALF_WIDTH_BINS * RATE / lf.frame_length(RATE)
    assert abs(width - 2.69) < 0.01
    near = np.min(np.abs(freqs[:, np.newaxis] - np.array(lines)[np.newaxis, :]), axis=1) <= width
    assert np.array_equal(lf.lf_bins(freqs, RATE, mains), (freqs >= 40.0) & (freqs < 200.0) & ~near)


def test_a_stereo_pair_reads_its_downmix():
    """A stereo side is downmixed the way the runner's pair is."""
    source = _source()
    output = v3.lf_shelf(source, RATE, -3.0)
    stereo = lf.lf_programme_db(np.stack([source, source], axis=1), np.stack([output, output], axis=1), RATE)
    assert abs(stereo - lf.lf_programme_db(source, output, RATE)) < 1e-9


def test_short_silent_and_briefly_live_windows_read_none():
    """Shorter than one band frame (0.74 s), under -70 dBFS, or under 30 live gain frames (1.4 s): nothing to read."""
    source = _source()
    short = source[: lf.frame_length(RATE) - 1]
    assert lf.lf_programme_db(short, short, RATE) is None
    quiet = (source * 1e-4).astype(np.float32)
    assert lf.lf_programme_db(quiet, quiet, RATE) is None
    assert lf.lf_programme_db(source, np.zeros_like(source), RATE) is None
    brief = source[: int(1.2 * RATE)]
    assert lf.lf_programme_db(brief, brief, RATE) is None


def _dbfs(mono):
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))


def test_a_dead_render_reads_none():
    """An output 64 dB under the source's programme, though over the silence level, is a dead render."""
    source = (_source() * 40.0).astype(np.float32)
    output = (source * 10.0 ** (-64.0 / 20.0)).astype(np.float32)
    assert _dbfs(output) > -70.0
    assert lf.lf_programme_db(source, output, RATE) is None


def test_a_window_without_programme_in_the_gain_band_reads_none():
    """A lone 100 Hz tone over hiss: no programme cell in 300-3000 Hz to match the gain on."""
    t = np.arange(int(SECONDS * RATE)) / RATE
    source = (0.1 * np.sin(2 * np.pi * 100.0 * t) + 1e-4 * np.random.default_rng(4).standard_normal(len(t))).astype(np.float32)
    assert lf.lf_programme_db(source, source, RATE) is None


def test_a_programme_without_a_low_end_reads_none():
    """Programme high-passed at 400 Hz: the band holds under 1/10000 of the core's power, so there is no low end to judge."""
    clean, _noise = _music()
    sos = scipy.signal.butter(8, 400.0, btype="highpass", fs=RATE, output="sos")
    source = scipy.signal.sosfiltfilt(sos, clean).astype(np.float32)
    assert lf.lf_programme_db(source, source, RATE) is None


def test_the_share_compares_the_low_end_with_the_core_on_the_loud_frames():
    """`lf_share_db` is the band's power over the 300-3000 Hz core's, in dB, on the frames given."""
    freqs = np.array([100.0, 1000.0, 5000.0])
    power = np.array([[1.0, 4.0], [10.0, 40.0], [99.0, 99.0]])
    loud = np.array([False, True])
    assert abs(lf.lf_share_db(power, freqs, loud, 4.0) - 10.0 * np.log10(0.1)) < 1e-9


def test_r1s_gain_match_reads_a_broadband_gain_and_none_without_programme():
    """The gain match R11 borrows: a halved output reads -6.02 dB; a core without guarded cells reads None."""
    clean, noise = _music()
    freqs, power = lf.stft_power(clean + noise, RATE, lf.GAIN_FRAME, lf.GAIN_HOP)
    assert abs(lf.matched_gain_db(power, 0.25 * power, freqs) - 10.0 * np.log10(0.25)) < 1e-9
    flat = np.ones_like(power)
    assert not balance_metrics.guarded_cells(flat).any()
    assert lf.matched_gain_db(flat, flat, freqs) is None
    assert auditory.GAIN_BAND_HZ == (300.0, 3000.0)


def test_a_core_with_too_few_guarded_cells_reads_none(monkeypatch):
    """Under 100 cells noise-only cells can own the median: on a realistic-v2 bed they set the gain -19 dB and R11 +18.9."""
    clean, noise = _music()
    freqs, power = lf.stft_power(clean + noise, RATE, lf.GAIN_FRAME, lf.GAIN_HOP)
    count = int(np.count_nonzero(balance_metrics.guarded_cells(power)[lf.core_bins(freqs)]))
    assert count >= lf.MIN_GAIN_CELLS
    monkeypatch.setattr(lf, "MIN_GAIN_CELLS", count + 1)
    assert lf.matched_gain_db(power, power, freqs) is None
    assert lf.lf_programme_db(clean + noise, clean + noise, RATE) is None
