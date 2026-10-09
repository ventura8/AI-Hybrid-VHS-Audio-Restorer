"""R1, the programme-cell balance: a shelf or a tilt reads at its size, a gain or hiss removal reads nothing."""

import numpy as np
import scipy.signal

from scripts.restoration_quality import balance_metrics as balance

RATE = 44100
SECONDS = 15.0


def _programme(seed=3, hiss=1e-3, f0=150.0, top_hz=15000.0):
    """`(clean, noise)`: harmonic bursts up to `top_hz` (0.5 s on, 0.5 s off) and a steady hiss under them.

    The shared `_voice` helpers stop at 6-9 kHz; R1 reads up to 16 kHz, so the harmonics here run to 15 kHz.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(SECONDS * RATE)) / RATE
    count = int(top_hz // f0)
    phases = rng.uniform(0.0, 2 * np.pi, count + 1)
    voice = sum(np.sin(2 * np.pi * f0 * k * t + phases[k]) / np.sqrt(k) for k in range(1, count + 1))
    ramp = np.hanning(int(0.02 * RATE))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    return (0.02 * voice * gate).astype(np.float32), (hiss * rng.standard_normal(len(t))).astype(np.float32)


def _source():
    clean, noise = _programme()
    return clean + noise


def _shelf_coefficients(gain_db, corner_hz, q=0.707):
    """RBJ high shelf, the shape of ffmpeg's `treble`."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * corner_hz / RATE
    cos, lift = np.cos(w0), 2.0 * np.sqrt(a) * np.sin(w0) / (2.0 * q)
    b = [a * ((a + 1) + (a - 1) * cos + lift), -2 * a * ((a - 1) + (a + 1) * cos), a * ((a + 1) + (a - 1) * cos - lift)]
    den = [(a + 1) - (a - 1) * cos + lift, 2 * ((a - 1) - (a + 1) * cos), (a + 1) - (a - 1) * cos - lift]
    return np.array(b) / den[0], np.array(den) / den[0]


def _shelved(mono, gain_db=2.0, corner_hz=7500.0):
    b, a = _shelf_coefficients(gain_db, corner_hz)
    return scipy.signal.lfilter(b, a, mono).astype(np.float32)


def _shelf_response_db(freqs, gain_db=2.0, corner_hz=7500.0):
    b, a = _shelf_coefficients(gain_db, corner_hz)
    _w, response = scipy.signal.freqz(b, a, worN=freqs, fs=RATE)
    return 20.0 * np.log10(np.abs(response))


def _tilted(mono, db_per_octave, from_hz=1000.0):
    """A zero-phase tilt of `db_per_octave` above `from_hz`, flat below."""
    spectrum = np.fft.rfft(np.asarray(mono, dtype=np.float64))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / RATE)
    gain_db = db_per_octave * np.log2(np.maximum(freqs, from_hz) / from_hz)
    return np.fft.irfft(spectrum * 10.0 ** (gain_db / 20.0), len(mono)).astype(np.float32)


def _bursty_hf_noise(length, level=1e-3, seed=11):
    """Noise above 6 kHz switched on in 10 % of 10 ms blocks: a heavy-tailed bin no per-cell margin can stop."""
    rng = np.random.default_rng(seed)
    block = int(0.01 * RATE)
    gate = np.repeat(np.where(rng.random(-(-length // block)) < 0.1, 1.0, 1e-3), block)[:length]
    sos = scipy.signal.butter(8, 6000.0, btype="highpass", fs=RATE, output="sos")
    return (level * scipy.signal.sosfiltfilt(sos, rng.standard_normal(length)) * gate).astype(np.float32)


def _bursty_pair(level=1e-3, wander_db=0.0, wander_hz=0.3):
    """Programme to 5 kHz over hiss and HF bursts; the whole noise wanders +-`wander_db` at `wander_hz`."""
    clean, hiss = _programme(hiss=1e-4, top_hz=5000.0)
    wander = 10.0 ** (wander_db * np.sin(2 * np.pi * wander_hz * np.arange(len(clean)) / RATE + 0.7) / 20.0)
    noise = ((hiss + _bursty_hf_noise(len(clean), level=level)) * wander).astype(np.float32)
    return clean + noise, clean + 0.1 * noise


def _bandpass(noise, low, high):
    """A zero-phase 4th-order Butterworth band-pass."""
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, noise)


def _voice_with_esses(seconds=SECONDS, seed=7):
    """The sibilance tests' speech: 0.3 s vowels, 120 ms 's' bursts, 30 % pauses (60 and 120 ms) over a faint hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t + phases[k]) / k for k in range(1, 30))
    vowel_gate = ((t % 0.6) < 0.3).astype(np.float64)
    ess_gate = ((t % 0.6) >= 0.36) & ((t % 0.6) < 0.48)
    ramp = np.hanning(int(0.01 * RATE))
    vowel_gate = np.convolve(vowel_gate, ramp / ramp.sum(), mode="same")
    ess_gate = np.convolve(ess_gate.astype(np.float64), ramp / ramp.sum(), mode="same")
    ess = 0.5 * _bandpass(rng.standard_normal(len(t)), 4000.0, 10000.0) + 0.35 * _bandpass(rng.standard_normal(len(t)), 1000.0, 4000.0)
    return (0.1 * vowel * vowel_gate + 0.1 * ess * ess_gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32), ess_gate > 0.5


def _dulled(source, ess_gate, depth_db=12.0):
    """The 6-12 kHz top of the 's' attenuated inside the bursts only."""
    top = _bandpass(np.asarray(source, dtype=np.float64), 6000.0, 12000.0)
    return (source - (1.0 - 10 ** (-depth_db / 20.0)) * top * ess_gate).astype(np.float32)


def _silenced(audio, seconds):
    """`audio` with its first `seconds` set to digital zero, as capture padding leaves it."""
    padded = np.array(audio, dtype=np.float32)
    padded[: int(seconds * RATE)] = 0.0
    return padded


def _moves_under(readings, limit):
    """Every reading is None or moved by less than `limit`."""
    return all(value is None or abs(value) < limit for value in readings.values())


def test_identity_and_a_broadband_gain_read_flat():
    source = _source()
    assert all(abs(value) < 1e-6 for value in balance.balance_readings(source, source, RATE).values())
    assert all(abs(value) < 1e-3 for value in balance.balance_readings(source, 0.3 * source, RATE).values())


def test_an_air_shelf_reads_its_own_response_in_every_band():
    source = _source()
    centres, levels = balance.band_profile(source, _shelved(source), RATE)
    readable = np.isfinite(levels)
    assert readable.sum() >= 30
    assert np.max(np.abs(levels[readable] - _shelf_response_db(centres[readable]))) < 0.1


def test_an_air_shelf_lifts_air_leaves_presence_and_tilts_up():
    source = _source()
    readings = balance.balance_readings(source, _shelved(source), RATE)
    centres, _levels = balance.band_profile(source, source, RATE)
    expected_air = float(np.mean(_shelf_response_db(centres[centres >= 5000.0])))
    assert abs(readings["balance_air_db"] - expected_air) < 0.15
    assert readings["balance_air_db"] > 1.0
    assert abs(readings["balance_presence_db"]) < 0.2
    assert readings["balance_tilt_db_oct"] > 0.3


def test_a_tilt_above_1_khz_reads_its_slope():
    source = _source()
    readings = balance.balance_readings(source, _tilted(source, -1.0), RATE)
    assert abs(readings["balance_tilt_db_oct"] + 1.0) < 0.05
    assert readings["balance_air_db"] < readings["balance_presence_db"] < 0.0


def test_oracle_hiss_removal_moves_no_reading():
    clean, noise = _programme(hiss=3e-3)
    for scale in (0.316, 0.1):
        readings = balance.balance_readings(clean + noise, clean + scale * noise, RATE)
        assert all(abs(value) < 0.3 for value in readings.values())


def test_leading_digital_silence_moves_no_reading():
    """Capture padding fills the quiet frames with zeros; the live-frame trim keeps hiss cells out."""
    clean, noise = _programme(hiss=3e-3)
    source, output = _silenced(clean + noise, 8.0), _silenced(clean + 0.1 * noise, 8.0)
    readings = balance.balance_readings(source, output, RATE)
    assert _moves_under(readings, 0.3)
    assert readings["balance_presence_db"] is not None


def test_without_the_noise_margin_hiss_between_sparse_harmonics_reads_as_a_change(monkeypatch):
    """Guard (a) alone: with guard (b) off, hiss cells between 600 Hz harmonics pass the bare 12 dB floor rule."""
    clean, noise = _programme(hiss=3e-3, f0=600.0)
    source, output = clean + noise, clean + 0.1 * noise
    monkeypatch.setattr(balance, "MAX_NOISE_SHARE", np.inf)
    guarded = balance.balance_readings(source, output, RATE)
    assert _moves_under(guarded, 0.3)
    assert guarded["balance_presence_db"] is not None
    monkeypatch.setattr(balance, "NOISE_MARGIN_DB", 0.0)
    monkeypatch.setattr(balance, "NOISE_MEAN_OVER_FLOOR", 1.0)
    assert balance.balance_readings(source, output, RATE)["balance_presence_db"] < -3.0


def test_speech_with_short_pauses_reads_its_tilt_and_presence():
    """Pauses under 30 % of the frames: the 6 dB cut keeps vowel onsets out of the quiet frames, so speech bands stay readable."""
    voice, _ess_gate = _voice_with_esses()
    readings = balance.balance_readings(voice, _tilted(voice, -1.0), RATE)
    assert abs(readings["balance_tilt_db_oct"] + 1.0) < 0.1
    assert readings["balance_presence_db"] < -0.5
    assert abs(balance.balance_readings(voice, voice, RATE)["balance_presence_db"]) < 1e-9


def test_without_the_6_db_cut_vowel_onsets_read_speech_bands_as_noise_led(monkeypatch):
    """The quietest 30 % alone hold vowel onsets and decays: their harmonics count as noise and the tilt is lost."""
    voice, _ess_gate = _voice_with_esses()
    monkeypatch.setattr(balance, "QUIET_FRAME_SPAN_DB", np.inf)
    assert balance.balance_readings(voice, _tilted(voice, -1.0), RATE)["balance_tilt_db_oct"] is None


def test_speech_reads_voiced_loud_frames_only_so_a_dulled_s_reads_nothing():
    """No 's' frame is a loud frame: R1 cannot see the 's', R2 and the above-band reading must."""
    voice, ess_gate = _voice_with_esses()
    assert _moves_under(balance.balance_readings(voice, _dulled(voice, ess_gate), RATE), 0.05)


def test_a_muted_output_reads_none(monkeypatch):
    """A dead render reads None, by its level or, with the level check off, by the programme gain floor."""
    source = _source()
    dither = (1e-7 * np.random.default_rng(1).standard_normal(len(source))).astype(np.float32)
    for output in (np.zeros_like(source), dither):
        assert set(balance.balance_readings(source, output, RATE).values()) == {None}
    monkeypatch.setattr(balance, "MIN_LEVEL_DBFS", -500.0)
    assert set(balance.balance_readings(source, np.zeros_like(source), RATE).values()) == {None}


def test_a_stereo_pair_reads_its_downmix():
    """A 2-D side is downmixed the way the runner's pair is."""
    source = _source()
    stereo = balance.balance_readings(np.stack([source, source], axis=1), np.stack([_shelved(source)] * 2, axis=1), RATE)
    assert stereo == balance.balance_readings(source, _shelved(source), RATE)


def test_bands_led_by_bursty_noise_drop_out():
    source, output = _bursty_pair()
    centres, levels = balance.band_profile(source, output, RATE)
    assert np.all(np.isnan(levels[centres >= 6500.0]))
    readings = balance.balance_readings(source, output, RATE)
    assert readings["balance_air_db"] is None
    assert abs(readings["balance_tilt_db_oct"]) < 0.1


def test_without_the_noise_share_guard_bursty_noise_would_read_as_dull(monkeypatch):
    source, output = _bursty_pair()
    monkeypatch.setattr(balance, "MAX_NOISE_SHARE", np.inf)
    assert balance.balance_readings(source, output, RATE)["balance_air_db"] < -10.0


def test_a_wandering_noise_floor_keeps_the_quietest_30_percent(monkeypatch):
    """A flat noise-led core keeps the whole 30 %: the quietest 10 % alone sit in the floor's troughs and under-count."""
    source, output = _bursty_pair(level=3e-3, wander_db=3.0)
    assert balance.balance_readings(source, output, RATE)["balance_air_db"] is None
    monkeypatch.setattr(balance, "QUIET_FRAME_PERCENTILE", 10.0)
    assert balance.balance_readings(source, output, RATE)["balance_air_db"] < -10.0


def test_the_bandwidth_clips_the_bands_read():
    source = _source()
    centres, _levels = balance.band_profile(source, _shelved(source), RATE, bandwidth_hz=8000.0)
    assert centres.max() < 8000.0
    full = balance.balance_readings(source, _shelved(source), RATE)["balance_air_db"]
    assert 0.0 < balance.balance_readings(source, _shelved(source), RATE, bandwidth_hz=8000.0)["balance_air_db"] < full
    assert set(balance.balance_readings(source, source, RATE, bandwidth_hz=50.0).values()) == {None}


def test_short_silent_and_quiet_windows_read_none():
    source = _source()
    silence = np.zeros(2 * RATE, dtype=np.float32)
    for src in (source[:1000], silence, source * np.float32(1e-4), _silenced(source, 14.0)):
        assert set(balance.balance_readings(src, src, RATE).values()) == {None}


def test_a_window_without_programme_in_the_gain_band_reads_none():
    clean, noise = _programme(f0=5000.0)
    source = clean + noise
    assert balance.band_profile(source, source, RATE) is None


def test_lag_aligned_applies_the_lag_without_gain():
    raw = np.arange(10.0)
    output = balance.lag_aligned(raw, np.concatenate([[0.0, 0.0], 2.0 * raw]), 2)[1]
    assert list(output) == list(2.0 * raw)
    source = balance.lag_aligned(np.concatenate([[9.0], raw]), raw, -1)[0]
    assert list(source) == list(raw)
    assert len(balance.lag_aligned(raw, raw[:7], 0)[0]) == 7


def test_programme_below_1_2_khz_reads_body_only():
    clean, noise = _programme(top_hz=1200.0)
    readings = balance.balance_readings(clean + noise, _tilted(clean + noise, -1.0), RATE)
    assert readings["balance_tilt_db_oct"] is None
    assert readings["balance_presence_db"] is None
    assert readings["balance_air_db"] is None
    assert abs(readings["balance_body_db"]) < 0.3


def test_the_band_layout_under_a_cap_starts_each_reading_above_its_span():
    """The last band centres under the cap: air's 5 kHz span has no band under a 5.2 kHz cap, its first under 5292 Hz."""
    assert [balance.bands_under("balance_air_db", cap) for cap in (5000.0, 5200.0, 5300.0, 7127.0)] == [0, 0, 1, 3]
    assert balance.bands_under("balance_presence_db", 2100.0) == 0
    assert balance.bands_under("balance_tilt_db_oct", 1100.0) < 3
    assert balance.bands_under("balance_air_db", None, 22050) == balance.bands_under("balance_air_db", 11025.0) == 7


def test_the_tilt_is_readable_from_its_third_band_only():
    """The tilt needs three bands, which the layout carries from a 1408 Hz cap: 1400 Hz reads no tilt, 1414.2 Hz does."""
    assert not balance.readable_under("balance_tilt_db_oct", 1400.0)
    assert balance.readable_under("balance_tilt_db_oct", 1414.2)


def test_the_first_readable_cap_is_where_the_layout_carries_the_reading():
    """To 1 Hz: air 5292, presence 2125, the tilt's third band 1408; body has a band from the 100 Hz floor up."""
    found = {name: balance.readable_from_hz(name) for name in balance.READINGS}
    assert {name: round(cap) for name, cap in found.items()} == {
        "balance_air_db": 5292,
        "balance_presence_db": 2125,
        "balance_tilt_db_oct": 1408,
        "balance_body_db": 101,
    }
    found.pop("balance_body_db")
    assert all(balance.readable_under(name, cap) and not balance.readable_under(name, cap - 1.0) for name, cap in found.items())
