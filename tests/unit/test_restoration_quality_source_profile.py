"""The capture profile reads a source's programme band, codec cut, mains, CRT line and channel state."""

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import source_profile

# 48 kHz puts the whole 15375-15984 Hz line band under Nyquist, as on the captures.
RATE = 48000
SIXTH_OCTAVE = 1.0 / 6.0
# Line drifts as (depth, period s, phase): 2000 ppm moves the line +-31 Hz and the hum
# fundamental +-0.1 Hz; a 16 s period is slow enough for 1 s frames to follow.
STEADY = (0.0, 16.0, 0.0)
WITH_THE_HUM = (2e-3, 16.0, 0.0)
ON_ITS_OWN = (2e-3, 11.0, 1.0)


def _profile(signal, rate=RATE):
    """The capture profile of `signal` as float32."""
    return source_profile.capture_profile(np.asarray(signal, dtype=np.float32), rate)


def _octaves_off(value_hz, reference_hz):
    """How many octaves `value_hz` sits from `reference_hz`."""
    return abs(np.log2(value_hz / reference_hz))


def _cut(signal, cutoff_hz):
    """`signal` with every bin at or above `cutoff_hz` zeroed (a brickwall low-pass)."""
    spectrum = np.fft.rfft(signal)
    spectrum[np.fft.rfftfreq(len(signal), 1.0 / RATE) >= cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, len(signal))


def _times(seconds):
    """Sample times of `seconds` at the test rate."""
    return np.arange(int(seconds * RATE)) / RATE


def _programme(seconds=12.0, cutoff_hz=8000.0, floor=1e-3, seed=1, on_s=0.4):
    """Noise bursts (`on_s` on in every 0.7 s) brickwalled at `cutoff_hz` over a full-band hiss ~40 dB under them."""
    rng = np.random.default_rng(seed)
    t = _times(seconds)
    # A 20 ms ramp, so the gate's own edges add no broadband clicks above the cut.
    ramp = np.hanning(int(0.02 * RATE))
    gate = np.convolve(((t % 0.7) < on_s).astype(np.float64), ramp / ramp.sum(), mode="same")
    return (0.1 * gate * _cut(rng.standard_normal(len(t)), cutoff_hz) + floor * rng.standard_normal(len(t))).astype(np.float32)


def _tone(t, freq_hz, amplitude, wander=0.0):
    """A sine at `freq_hz * (1 + wander)`, `wander` a fraction per sample (a tape-speed error)."""
    return amplitude * np.sin(2 * np.pi * np.cumsum(np.full(len(t), freq_hz) * (1.0 + wander)) / RATE)


def _hum(t, mains_hz, wander=0.0):
    """Mains hum: the fundamental and its next two harmonics, all moved by the same speed error."""
    return _tone(t, mains_hz, 0.05, wander) + _tone(t, 2 * mains_hz, 0.02, wander) + _tone(t, 3 * mains_hz, 0.01, wander)


def _drift(t, depth, period_s, phase):
    """A slow sinusoidal speed drift, `depth` at its peak."""
    return depth * np.sin(2 * np.pi * t / period_s + phase)


def _line_case(line_drift, hum_depth, line_hz=15625.0, seconds=40.0):
    """Hum drifting `hum_depth` on a 16 s period, a CRT line drifting by `line_drift`, over a hiss floor."""
    t = _times(seconds)
    floor = 1e-3 * np.random.default_rng(5).standard_normal(len(t))
    hum = _hum(t, 50.0, _drift(t, hum_depth, 16.0, 0.0))
    return (hum + _tone(t, line_hz, 0.01, _drift(t, *line_drift)) + floor).astype(np.float32)


@pytest.mark.parametrize("cutoff_hz", [5000.0, 8000.0])
def test_a_band_limited_programme_reads_its_bandwidth_within_a_sixth_octave(cutoff_hz):
    """The loud-over-gap excess ends at the programme's cut, not at the hiss's."""
    bandwidth = _profile(_programme(cutoff_hz=cutoff_hz))["prog_bandwidth_hz"]
    assert bandwidth is not None
    assert _octaves_off(bandwidth, cutoff_hz) <= SIXTH_OCTAVE


def test_a_programme_band_edge_over_hiss_is_not_a_brickwall():
    """The hiss above a programme band edge keeps the gap frames from falling off a cliff."""
    assert _profile(_programme(cutoff_hz=8000.0))["brickwall_hz"] is None


def test_a_codec_cut_reads_a_brickwall_and_caps_the_bandwidth():
    """A hiss cut at 16 kHz is a brickwall, and the cap alone keeps loud programme above it out of the bandwidth."""
    # Bursts 0.25 s in 0.7 s, so every gap frame (p15-p40) sits wholly between them.
    bursts = _programme(cutoff_hz=20000.0, floor=0.0, on_s=0.25)
    signal = (bursts + _cut(1e-3 * np.random.default_rng(9).standard_normal(len(bursts)), 16000.0)).astype(np.float32)
    profile = _profile(signal)
    # Hann leakage lifts the first 100-200 Hz above the cut.
    assert 16000.0 <= profile["brickwall_hz"] <= 16300.0
    assert profile["prog_bandwidth_hz"] <= profile["brickwall_hz"]
    assert source_profile.programme_bandwidth_hz(source_profile.long_term_spectra(signal, RATE)) > profile["brickwall_hz"]


def test_a_gentle_roll_off_is_not_a_brickwall():
    """A 12 dB/octave roll-off never falls 30 dB within 500 Hz."""
    sos = scipy.signal.butter(2, 4000.0, fs=RATE, output="sos")
    assert _profile(scipy.signal.sosfilt(sos, _programme(cutoff_hz=20000.0)))["brickwall_hz"] is None


@pytest.mark.parametrize("mains_hz, error", [(50.0, 0.01), (50.0, -0.01), (60.0, 0.005), (60.0, -0.005)])
def test_hum_off_its_nominal_rate_still_reads_its_mains(mains_hz, error):
    """A tape-speed error up to 1 % moves the first three harmonics within their 2-bin search."""
    t = _times(20.0)
    profile = _profile(_hum(t, mains_hz * (1.0 + error)) + 0.03 * np.random.default_rng(2).standard_normal(len(t)))
    assert profile["mains_hz"] == mains_hz
    assert profile["mains_evidence_db"] > 20.0
    assert profile["mains_excess_db"] > 6.0


def test_a_speed_error_on_the_upper_harmonics_lands_in_their_wider_search():
    """A 1 % error moves the 8th harmonic of 50 Hz by 4 Hz, past 2 bins: the 1 % share still holds it (+10000 ppm)."""
    t = _times(20.0)
    hum = sum(_tone(t, harmonic * 50.5, 0.02) for harmonic in (6, 7, 8))
    signal = (hum + 1e-3 * np.random.default_rng(10).standard_normal(len(t))).astype(np.float32)
    ppm, _prominence = source_profile.track_tones(signal, RATE)["hum"][50.0]
    assert np.median(ppm[:, 7]) == pytest.approx(10000.0, abs=100.0)
    assert _profile(signal)["mains_hz"] == 50.0


def _voice_over_hum(seconds=20.0):
    """A weak 50 Hz fundamental under a voice whose 120 Hz pitch lines up with the 60 Hz series, voiced 30 % of the time."""
    t = _times(seconds)
    voiced = (t % 10.0) < 3.0
    voice = voiced * sum(_tone(t, harmonic * 120.0, 0.05) for harmonic in range(1, 5))
    return _tone(t, 50.0, 0.005) + voice + 1e-3 * np.random.default_rng(3).standard_normal(len(t))


def test_a_lone_fundamental_under_louder_programme_is_still_read_by_its_frames():
    """The SOTI case: the summed rule follows the loudest neighbourhoods (the voice), the frames' medians the hum."""
    signal = _voice_over_hum()
    profile = _profile(signal)
    assert source_profile.summed_rule_mains(signal, RATE) == 60.0
    assert profile["mains_hz"] == 50.0
    assert profile["mains_evidence_db"] >= source_profile.MAINS_MIN_EVIDENCE_DB


def _noise_evidence(seed, seconds=15.0):
    """The 50 and 60 Hz harmonic evidence of `seconds` of white noise."""
    noise = 0.05 * np.random.default_rng(seed).standard_normal(int(seconds * RATE))
    tracks = source_profile.track_tones(noise.astype(np.float32), RATE)
    return [source_profile.harmonic_evidence_db(prominence) for _ppm, prominence in tracks["hum"].values()]


def test_noise_alone_never_names_the_mains_on_the_frames():
    """On 24 noise-only 15 s clips neither series reaches the evidence the frames need to decide."""
    evidence = np.array([_noise_evidence(seed) for seed in range(24)])
    assert evidence.max() < source_profile.MAINS_MIN_EVIDENCE_DB


@pytest.mark.parametrize("harmonic", range(1, 9))
def test_both_series_search_equally_many_bins_at_each_harmonic(harmonic):
    """Harmonic k of 50 and of 60 Hz search and floor over as many bins, so noise reads no more on either."""
    (search_50, floor_50), (search_60, floor_60) = (source_profile.hum_windows(harmonic * mains, harmonic, 1.0) for mains in (50.0, 60.0))
    assert search_50[1] - search_50[0] == search_60[1] - search_60[0]
    assert floor_50[1] - floor_50[0] == floor_60[1] - floor_60[0]


def test_without_harmonic_evidence_the_summed_rule_decides():
    """Plain noise gives the frames too little evidence, so `measure_hum._mains_by_evidence` picks."""
    noise = (0.05 * np.random.default_rng(4).standard_normal(int(20.0 * RATE))).astype(np.float32)
    profile = _profile(noise)
    assert profile["mains_hz"] == source_profile.summed_rule_mains(noise.astype(np.float64), RATE)
    assert profile["mains_evidence_db"] < source_profile.MAINS_MIN_EVIDENCE_DB


def test_fewer_than_ten_frames_never_decide():
    """A median over fewer than ten 1 s frames is not evidence, however high it reads (noise reaches 7 dB on 5 s)."""
    assert not source_profile.frames_decide(50.0, source_profile.MIN_TRACK_FRAMES - 1)
    assert source_profile.frames_decide(source_profile.MAINS_MIN_EVIDENCE_DB, source_profile.MIN_TRACK_FRAMES)


def test_a_long_source_is_read_on_eight_evenly_spaced_blocks():
    """200 s are read as eight 15 s blocks, the first at the start and the last at the end."""
    rate = 1000
    excerpt = source_profile.mains_excerpt(np.arange(200 * rate), rate)
    block = 15 * rate
    assert len(excerpt) == 8 * block
    assert np.array_equal(excerpt[:block], np.arange(block))
    assert np.array_equal(excerpt[-block:], np.arange(200 * rate - block, 200 * rate))


def test_a_steady_line_over_drifting_hum_is_the_playback_chain():
    """A line that holds still while the recorded hum drifts did not come off the tape."""
    profile = _profile(_line_case(STEADY, 2e-3))
    assert profile["line_origin"] == "playback_chain"
    assert abs(profile["line_hz"] - 15625.0) < 0.05
    assert profile["line_sd_ppm"] < 5.0
    assert profile["line_hum_corr"] is None


def test_a_corpus_length_clip_reads_its_line_but_no_correlation():
    """A 15 s corpus clip has enough frames for a line, too few for a correlation."""
    profile = _profile(_line_case(STEADY, 2e-3, seconds=15.0))
    assert abs(profile["line_hz"] - 15625.0) < 0.05
    assert profile["line_hum_corr"] is None


def test_a_line_drifting_with_the_hum_is_speed_locked():
    """A line moved by the same speed drift as the hum follows the transport."""
    profile = _profile(_line_case(WITH_THE_HUM, 2e-3))
    assert profile["line_origin"] == "speed_locked"
    assert profile["line_hum_corr"] > 0.9
    assert profile["line_sd_ppm"] > 1000.0


def test_a_line_drifting_on_its_own_is_unresolved():
    """A line that wanders but not with the hum is neither speed-locked nor steady."""
    profile = _profile(_line_case(ON_ITS_OWN, 2e-3))
    assert profile["line_origin"] == "unresolved"
    assert profile["line_hum_corr"] < source_profile.LINE_LOCK_CORR


def test_an_ntsc_line_reads_against_its_own_rate():
    """An NTSC line (15734.264 Hz) is measured against 15734 Hz, not 15625 Hz."""
    profile = _profile(_line_case(STEADY, 0.0, line_hz=15734.264))
    assert abs(profile["line_ppm"] - 16.8) < 0.5
    assert profile["line_origin"] == "playback_chain"


def test_frames_where_a_stronger_neighbour_wins_are_dropped_from_the_line():
    """The SOTI case: frames where a sideband 60 Hz away wins are not part of the line."""
    t = _times(40.0)
    # Every seventh second a tone three times the line's level sits 60 Hz above it.
    bursts = ((t.astype(int) % 7) == 3).astype(np.float64)
    profile = _profile(_line_case(STEADY, 0.0) + 0.03 * bursts * _tone(t, 15685.0, 1.0))
    assert abs(profile["line_hz"] - 15625.0) < 0.05
    assert profile["line_sd_ppm"] < 5.0


@pytest.mark.parametrize(
    "signal, rate",
    [
        (_hum(_times(30.0), 50.0), RATE),
        (_tone(_times(30.0), 15400.0, 0.01), RATE),
        (_line_case(STEADY, 0.0, seconds=8.0), RATE),
        (np.zeros(30 * 22050), 22050),
    ],
    ids=["no-line", "off-rate-peak", "too-few-frames", "line-above-nyquist"],
)
def test_no_line_reads_none(signal, rate):
    """No tone in the band, a peak 1.4 % off both rates, under ten frames, or a band above Nyquist: no line."""
    profile = _profile(signal, rate)
    assert {key: profile[key] for key in source_profile.NO_LINE} == source_profile.NO_LINE


def test_a_line_that_wanders_too_fast_to_track_is_no_line():
    """The Vaccin case: a peak hopping ~100 Hz from second to second is not a line."""
    t = _times(30.0)
    hops = 15625.0 + 150.0 * np.sin(np.floor(t) * 2.4)
    signal = 0.01 * np.sin(2 * np.pi * np.cumsum(hops) / RATE) + 1e-3 * np.random.default_rng(6).standard_normal(len(t))
    assert _profile(signal)["line_hz"] is None


def test_grandke_interpolation_is_exact_on_a_hann_windowed_tone():
    """A tone 0.3 bin off a bin centre reads 0.3 bin off (10 Hz bins, 1003 Hz)."""
    frame = 4800
    t = np.arange(frame) / RATE
    power = (np.abs(np.fft.rfft(np.sin(2 * np.pi * 1003.0 * t) * np.hanning(frame))) ** 2)[np.newaxis, :]
    peak = np.argmax(power, axis=1)
    assert abs(peak[0] + source_profile.grandke_offset(power, peak)[0] - 100.3) < 0.01


def test_a_peak_on_the_band_edge_keeps_its_bin():
    """A peak with only one neighbour is not interpolated."""
    band = np.array([[5.0, 1.0, 0.5], [0.5, 1.0, 5.0]])
    assert list(source_profile.grandke_offset(band, np.array([0, 2]))) == [0.0, 0.0]


def test_a_lone_qualifying_band_is_not_a_programme_band():
    """One band alone is not a sustained excess; two are, and the brickwall cap removes them."""
    loud = np.arange(100) % 2 == 0
    band_power = np.ones((100, 3))
    band_power[loud, 1] = 10.0
    spectra = {"gap": ~loud, "loud": loud, "band_power": band_power, "band_bins": np.ones(3), "centres": np.array([1000.0, 2000.0, 4000.0])}
    assert source_profile.programme_bandwidth_hz(spectra) is None
    band_power[loud, 0] = 10.0
    assert source_profile.programme_bandwidth_hz(spectra) == 2000.0
    assert source_profile.programme_bandwidth_hz(spectra, top_hz=1500.0) is None


def test_constant_noise_has_no_programme_band():
    """Stationary noise has no loud frames that differ from its gaps."""
    assert _profile(0.01 * np.random.default_rng(7).standard_normal(10 * RATE))["prog_bandwidth_hz"] is None


def test_empty_tracks_read_no_line_and_no_evidence():
    """Under one tracking frame there is no line and no harmonic evidence."""
    tracks = source_profile.track_tones(np.zeros(100, dtype=np.float32), RATE)
    hum = source_profile.hum_series(*tracks["hum"][50.0])
    assert source_profile.line_profile(tracks, hum) == source_profile.NO_LINE
    assert source_profile.harmonic_evidence_db(tracks["hum"][60.0][1]) == 0.0


def test_an_l_minus_r_pair_is_inverted_and_profiled_on_the_difference():
    """An L/-R pair reads r -1 and is profiled on (L - R) / 2, which a plain downmix would cancel."""
    programme = _programme()
    profile = _profile(np.stack([programme, -programme], axis=1))
    state = profile["channel_state"]
    assert state["correlation"] == pytest.approx(-1.0)
    assert state["inverted"]
    assert not state["dual_mono"]
    assert _octaves_off(profile["prog_bandwidth_hz"], 8000.0) <= SIXTH_OCTAVE


def test_identical_channels_are_dual_mono():
    """Identical channels are dual mono, fully correlated and level."""
    programme = _programme(seconds=4.0)
    state = source_profile.channel_state(np.stack([programme, programme], axis=1), RATE)
    assert state["dual_mono"]
    assert not state["inverted"]
    assert state["correlation"] == pytest.approx(1.0)
    assert state["energy_ratio_db"] == pytest.approx(0.0)


@pytest.mark.parametrize("live, dead", [(0, "right"), (1, "left")])
def test_a_dead_channel_is_named_and_the_live_one_profiled(live, dead):
    """A side ~70 dB down is dead (the app's cut is 25 dB), and the profile reads the live side alone."""
    programme = _programme()
    audio = np.zeros((len(programme), 2), dtype=np.float32)
    audio[:, live] = programme
    audio[:, 1 - live] = 1e-4 * np.random.default_rng(8).standard_normal(len(programme))
    profile = _profile(audio)
    assert profile["channel_state"]["dead_channel"] == dead
    assert not profile["channel_state"]["inverted"]
    assert _octaves_off(profile["prog_bandwidth_hz"], 8000.0) <= SIXTH_OCTAVE


def test_a_level_mismatch_reads_its_ratio_and_keeps_both_channels():
    """A 6 dB mismatch is a level difference, not a dead channel."""
    programme = _programme(seconds=4.0)
    state = source_profile.channel_state(np.stack([programme, 0.5 * programme], axis=1), RATE)
    assert state["energy_ratio_db"] == pytest.approx(6.02, abs=0.01)
    assert state["dead_channel"] is None
    assert state["correlation"] == pytest.approx(1.0)


def test_a_non_finite_sample_reads_as_silence():
    """One NaN in a pair that differs by 0.1 neither kills the channel nor hides the difference."""
    programme = _programme(seconds=4.0)
    audio = np.stack([programme, programme + 0.1], axis=1)
    audio[100, 0] = np.nan
    state = source_profile.channel_state(audio, RATE)
    assert state["dead_channel"] is None
    assert not state["dual_mono"]
    assert np.isfinite([state["correlation"], state["energy_ratio_db"]]).all()


def test_silent_channels_have_no_ratio_or_correlation():
    """Two silent channels give no ratio and no correlation."""
    state = source_profile.channel_state(np.zeros((RATE, 2), dtype=np.float32), RATE)
    assert state["energy_ratio_db"] is None
    assert state["correlation"] is None
    assert state["dual_mono"]
    assert not state["inverted"]


def test_mono_input_is_one_channel_in_either_shape():
    """A flat array and a one-column array give the same profile."""
    programme = _programme()
    flat = _profile(programme)
    assert flat["channel_state"] == source_profile.MONO_STATE
    assert flat == _profile(programme[:, np.newaxis])


def test_short_audio_reads_none_but_keeps_its_channel_state():
    """Under 1 s nothing is read; under 60 profile frames (2.5 s) there are no classes."""
    programme = _programme(seconds=1.5)
    short = _profile(np.stack([programme, programme], axis=1))
    assert _profile(programme[: RATE // 2])["mains_hz"] is None
    assert short["prog_bandwidth_hz"] is None
    assert short["brickwall_hz"] is None
    assert short["channel_state"]["dual_mono"]
