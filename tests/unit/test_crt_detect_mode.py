"""crt_detect_mode: the scan's CRT line search on the head ('head', the default) or on the whole file ('welch').

'head' takes one FFT of the first 32768 samples and calls a line at 1.7x the band's median magnitude,
which noise alone clears. The verdicts and graphs pinned below were recorded on these fixtures with
HEAD e429683's modules/filters.py: the default key must rebuild them exactly, so every output keeps
its bytes. 'welch' averages up to 256 segments over the whole file and calls a line when the band's
strongest bin stands 10 dB over the band and 6 dB over both its sides, each at least 21 dB over the
square root of the segments averaged, and is no tooth of a comb (modules/crt_line.py).
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

import modules.config as cfg
from modules import auto_scanner, crt_line, filters
from scripts import tune_restoration as tune

RATE = 44100
SECONDS = 10.0
PAL, NTSC = 15625.0, 15734.0
PAL_NOTCH = "bandreject=f=15625.0:width_type=q:w=30"
NTSC_NOTCH = "bandreject=f=15734.0:width_type=q:w=30"
HANN_ENBW_BINS = 1.5
PAL_HUM = "bandreject=f=50.0:width_type=q:w=15,bandreject=f=100.0:width_type=q:w=15,"
# HEAD e429683: the head search's verdict on each fixture and the pre-conditioning graph a whole scan builds.
HEAD = {
    "hiss_only": (NTSC, NTSC_NOTCH),
    "pal_line": (PAL, PAL_HUM + PAL_NOTCH),
    "late_pal_line": (NTSC, NTSC_NOTCH),
    "wandering_line": (NTSC, NTSC_NOTCH),
    "clean": (PAL, PAL_HUM + PAL_NOTCH),
}
# What the whole-file search reads: no line on hiss or on a clean programme, the PAL line wherever it sits.
WELCH = {"hiss_only": 0.0, "pal_line": PAL, "late_pal_line": PAL, "wandering_line": PAL, "clean": 0.0}


def _programme(seed=1):
    """Speech-like bursts low-passed at 6 kHz: nothing of their own in the line band."""
    rng = np.random.default_rng(seed)
    count = int(SECONDS * RATE)
    gate = (np.arange(count) / RATE % 1.0) < 0.6
    sos = scipy.signal.butter(8, 6000.0, fs=RATE, output="sos")
    return 0.1 * scipy.signal.sosfilt(sos, rng.standard_normal(count) * gate)


def _hiss(level, seed=2):
    return level * np.random.default_rng(seed).standard_normal(int(SECONDS * RATE))


def _tone(hz, amplitude, start_s=0.0):
    time = np.arange(int(SECONDS * RATE)) / RATE
    return amplitude * np.sin(2 * np.pi * hz * time) * (time >= start_s)


def _wandering(amplitude):
    """A line gliding from 15700 Hz down to 15625 Hz over the first 4 s, then steady: the head hears it as NTSC."""
    time = np.arange(int(SECONDS * RATE)) / RATE
    freq = np.where(time < 4.0, 15700.0 - 75.0 * time / 4.0, PAL)
    return amplitude * np.sin(2 * np.pi * np.cumsum(freq) / RATE)


FIXTURES = {
    "hiss_only": lambda: _programme() + _hiss(0.003),
    "pal_line": lambda: _programme() + _hiss(0.003) + _tone(PAL, 0.002),
    "late_pal_line": lambda: _programme() + _hiss(0.0003) + _tone(PAL, 0.002, start_s=1.0),
    "wandering_line": lambda: _programme() + _hiss(0.0003) + _wandering(0.002),
    "clean": _programme,
}


@pytest.fixture(name="welch")
def _welch():
    """The whole-file search, with the scan's log captured."""
    with patch("modules.filters.CRT_DETECT_MODE", "welch"), patch("modules.filters.log_msg") as log:
        yield log


def _scan_graph(tmp_path, name):
    """The pre-conditioning graph a whole scan of the fixture (as a dual-mono float WAV) builds."""
    wav = tmp_path / f"{name}.wav"
    mono = FIXTURES[name]()
    sf.write(str(wav), np.stack([mono, mono], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    with patch("modules.auto_scanner.log_msg"):
        strategy = auto_scanner.scan_and_decide_restoration_strategy(wav)
    return filters._precondition_filter_from_config(strategy["precondition_filters"])


def test_the_key_defaults_to_head():
    """The resolved config, its module constant and the scan's copy all hold every release's search."""
    assert cfg.CONFIG["crt_detect_mode"] == "head"
    assert cfg.CRT_DETECT_MODE == "head"
    assert filters.CRT_DETECT_MODE == "head"


def _loaded(user):
    """The configuration load_config resolves from a config.yaml holding `user`, and the warning printer."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user
    with (
        patch("modules.config.Path.exists", return_value=True),
        patch.object(cfg, "yaml", mock_yaml),
        patch("builtins.open", MagicMock()),
        patch("builtins.print") as warn,
    ):
        conf, _source = cfg.load_config()
    return conf, warn


@pytest.mark.parametrize(
    "user, expected, warned",
    [({}, "head", False), ({"crt_detect_mode": " Welch "}, "welch", False), ({"crt_detect_mode": "whole"}, "head", True)],
)
def test_the_key_is_a_choice(user, expected, warned):
    """An older config.yaml gets 'head' quietly; case and spaces are forgiven; an unknown value warns and keeps 'head'."""
    conf, warn = _loaded({"process_mode": "cathar", **user})
    assert conf["crt_detect_mode"] == expected
    assert warn.called is warned


def test_the_repository_config_and_the_tuning_driver_know_the_key():
    """config.yaml ships 'head'; a tuning grid may set the key, and an unknown value is a problem there."""
    text = (Path(cfg.__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    assert '\ncrt_detect_mode: "head"\n' in text
    fields = tune.known_config_fields()
    assert fields["crt_detect_mode"][0] == {"head", "welch"}
    assert tune._problem("crt_detect_mode", "whole", fields["crt_detect_mode"])


@pytest.mark.parametrize("name", sorted(HEAD))
def test_the_default_verdict_is_heads(name):
    """The default key reads each fixture as HEAD e429683 did, noise and a clean programme included."""
    assert filters._detect_crt_flyback_notch(FIXTURES[name](), RATE) == HEAD[name][0]


@pytest.mark.parametrize("name", sorted(HEAD))
def test_a_scan_with_the_default_key_builds_heads_graph(tmp_path, name):
    """The whole scan builds HEAD e429683's pre-conditioning graph, byte for byte."""
    assert _scan_graph(tmp_path, name) == HEAD[name][1]


@pytest.mark.parametrize("name", sorted(WELCH))
def test_the_whole_file_search_reads_the_line_wherever_it_sits(welch, name):
    """No line on hiss or a clean programme; the PAL line found when it starts late or wanders through NTSC's half."""
    assert filters._detect_crt_flyback_notch(FIXTURES[name](), RATE) == WELCH[name]
    welch.assert_called_once()


def test_the_whole_file_search_logs_what_it_rests_on(welch):
    filters._detect_crt_flyback_notch(FIXTURES["late_pal_line"](), RATE)
    line = str(welch.call_args.args[0])
    assert line.startswith("    [Auto-Scan] CRT line (whole file): 15625 Hz line over 25 segments (strongest bin 1562")


@pytest.mark.usefixtures("welch")
@pytest.mark.parametrize(("name", "kept", "dropped"), [("hiss_only", "", NTSC_NOTCH), ("late_pal_line", PAL_NOTCH, NTSC_NOTCH)])
def test_a_scan_with_the_whole_file_search_notches_only_a_line(tmp_path, name, kept, dropped):
    graph = _scan_graph(tmp_path, name)
    assert kept in graph
    assert dropped not in graph


@pytest.mark.parametrize(("mode", "line", "notch"), [("head", NTSC, 0.0), ("welch", 0.0, 50.0)])
def test_a_noise_verdict_no_longer_takes_the_hum_notch_away(mode, line, notch):
    """A hissy PAL tape with 50 Hz hum: the head reads the hiss as NTSC and holds the mains search to 60/120 Hz."""
    mono = _programme() + _hiss(0.003) + _tone(50.0, 0.01)
    with patch("modules.filters.CRT_DETECT_MODE", mode), patch("modules.filters.log_msg"):
        verdict = filters._detect_crt_flyback_notch(mono, RATE)
    assert (verdict, filters._detect_mains_buzz_notch(mono, RATE, verdict)) == (line, notch)


def _pink(rng, count):
    spectrum = np.fft.rfft(rng.standard_normal(count)) / np.sqrt(np.maximum(np.arange(count // 2 + 1), 1))
    return np.fft.irfft(spectrum, count)


def _verdict(mono):
    return crt_line.verdict(crt_line.search(mono, RATE), (PAL, NTSC))


@pytest.mark.parametrize("seconds", [0.8, 2.0, 15.0])
def test_noise_makes_no_line_at_any_length(seconds):
    """White and pink noise on one, four and 39 segments: no line."""
    rng = np.random.default_rng(3)
    count = int(seconds * RATE)
    assert [_verdict(noise) for noise in (rng.standard_normal(count), _pink(rng, count))] == [0.0, 0.0]


def test_15_s_of_noise_stands_under_3_db():
    """The band's strongest bin of 15 s of white noise stands under 3 dB: the 10 dB test keeps a wide margin."""
    found = crt_line.search(np.random.default_rng(4).standard_normal(15 * RATE), RATE)
    assert found.prominence_db < 3.0


def test_a_codec_cut_inside_the_band_is_no_line():
    """Noise cut at 15.5 kHz: the strongest bin sits on the cut, far over the band's median, but no higher than its lower side."""
    spectrum = np.fft.rfft(_hiss(0.01))
    spectrum[np.fft.rfftfreq(int(SECONDS * RATE), 1.0 / RATE) > 15500.0] = 0.0
    found = crt_line.search(np.fft.irfft(spectrum, int(SECONDS * RATE)), RATE)
    assert found.prominence_db > crt_line.MIN_PROMINENCE_DB
    assert crt_line.verdict(found, (PAL, NTSC)) == 0.0


def _lines(frequencies, over_db, count):
    """Steady tones, each `over_db` over the density white hiss of mean square 1e-4 puts in one 32768-point bin."""
    amplitude = np.sqrt(4.0 * 1e-4 * HANN_ENBW_BINS / crt_line.SEGMENT * 10.0 ** (over_db / 10.0))
    time = np.arange(count) / RATE
    return sum(amplitude * np.sin(2 * np.pi * hz * time) for hz in frequencies)


def _tones(frequencies, over_db, seconds=15.0, seed=6):
    """White hiss of mean square 1e-4 and the tones `_lines` gives."""
    count = int(seconds * RATE)
    return np.random.default_rng(seed).standard_normal(count) * 0.01 + _lines(frequencies, over_db, count)


@pytest.mark.parametrize(("over_db", "expected"), [(8.0, 0.0), (14.0, PAL)])
def test_a_line_needs_10_db_over_the_band(over_db, expected):
    """A line at the rate 8 dB over the hiss is not called (the documented limit: a weak line waits for the denoise); 14 dB is."""
    assert _verdict(_tones([PAL], over_db)) == expected


def test_a_comb_is_no_line():
    """Tones every 43.07 Hz (44100 / 1024) across the band, 20 dB out: each tooth stands, but the comb has a dozen."""
    found = crt_line.search(_tones(np.arange(15460.8, 15900.0, RATE / 1024), 20.0), RATE)
    assert found.prominence_db > crt_line.MIN_PROMINENCE_DB
    assert found.teeth >= crt_line.COMB_TEETH
    assert crt_line.verdict(found, (PAL, NTSC)) == 0.0


def test_a_line_with_its_field_sidebands_is_a_line():
    """A PAL line with sidebands 25 and 50 Hz either side at 6 dB less: four teeth at most, still a line."""
    mono = _tones([PAL], 20.0) + _lines([PAL - 50.0, PAL - 25.0, PAL + 25.0, PAL + 50.0], 14.0, 15 * RATE)
    found = crt_line.search(mono, RATE)
    assert found.teeth < crt_line.COMB_TEETH
    assert crt_line.verdict(found, (PAL, NTSC)) == PAL


def test_a_whine_spread_by_flutter_is_still_a_line():
    """A recorded whine swung +-55 Hz at 0.5 Hz (realistic-v2's transport classes): its sides 108-215 Hz away stay clear."""
    count = 15 * RATE
    time = np.arange(count) / RATE
    phase = 2 * np.pi * PAL * time + (55.0 / 0.5) * np.sin(2 * np.pi * 0.5 * time)
    mono = np.random.default_rng(9).standard_normal(count) * 0.001 + 0.01 * np.sin(phase)
    assert _verdict(mono) == PAL


@pytest.mark.parametrize(("seconds", "expected"), [(0.8, 0.0), (15.0, PAL)])
def test_one_segment_asks_for_21_db(seconds, expected):
    """One segment's noise reaches 12.5 dB: a line 15 dB over the hiss is no line on 0.8 s and a line on 15 s."""
    assert _verdict(_tones([PAL], 15.0, seconds=seconds, seed=7)) == expected


@pytest.mark.parametrize(("segments", "expected"), [(1, 21.0), (4, 10.5), (49, 3.0)])
def test_the_noise_floor_falls_with_the_root_of_the_segments(segments, expected):
    assert crt_line.noise_db(segments) == pytest.approx(expected)


@pytest.mark.parametrize(("count", "rate"), [(crt_line.SEGMENT - 1, RATE), (crt_line.SEGMENT, 30000)])
def test_audio_the_search_cannot_read_has_no_search(count, rate):
    """Shorter than a segment, or a rate whose band lies over Nyquist: no search and no line."""
    found = crt_line.search(np.ones(count), rate)
    assert found is None
    assert crt_line.verdict(found, (PAL, NTSC)) == 0.0


@pytest.mark.parametrize(("peak", "sign"), [(90, 1), (10, -1)])
def test_a_side_off_the_spectrum_reads_minus_infinity(peak, sign):
    assert crt_line._side_db(np.zeros(100), peak, sign) == -math.inf


def test_a_long_tape_reads_at_most_256_segments():
    """Three minutes hold 322 half-overlapping segments; the search spreads 256 evenly over them."""
    with patch("modules.crt_line.np.fft.rfft", wraps=np.fft.rfft) as fft:
        _power, segments = crt_line.averaged_power(np.zeros(180 * RATE))
    assert fft.call_count == crt_line.MAX_SEGMENTS
    assert segments == crt_line.MAX_SEGMENTS


@pytest.mark.parametrize(
    ("found", "rate_hz", "expected"),
    [
        (None, 0.0, "no line (shorter than one 32768-sample segment, or the band over Nyquist)"),
        (
            crt_line.Search(39, 15625.0, 22.94, 21.01, 0),
            PAL,
            "15625 Hz line over 39 segments (strongest bin 15625.0 Hz, 22.9 dB over the band, 21.0 dB over its sides, 0 comb teeth)",
        ),
        (
            crt_line.Search(39, 15703.1, 82.8, 68.8, 5),
            0.0,
            "no line over 39 segments (strongest bin 15703.1 Hz, 82.8 dB over the band, 68.8 dB over its sides, 5 comb teeth)",
        ),
    ],
)
def test_the_log_line_says_what_the_search_gave(found, rate_hz, expected):
    assert crt_line.describe(found, rate_hz) == expected
