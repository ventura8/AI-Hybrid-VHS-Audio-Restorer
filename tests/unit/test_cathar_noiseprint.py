"""cathar's noise probe (single window, length rule, stitched pauses) and the multiband de-esser guard."""

from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import modules.cathar as cathar


def test_cathar_deesser_single_band_keeps_a_negative_threshold(tmp_path):
    """With one band the threshold is a HF/broadband ratio, where -24 dB is cathar's default."""
    with patch("modules.cathar._run_cathar_step") as mock_step:
        cathar._cathar_deesser_step(tmp_path / "in.wav", tmp_path, bands=1, freq=4000, threshold=-24.0)
    assert mock_step.call_args[0][0] == ["deesser", "--bands", "1", "-f", "4000", "--threshold=-24.0"]


def test_cathar_deesser_multiband_refuses_a_non_positive_threshold(tmp_path):
    """Multiband threshold is dB above each band's running average.

    A negative value engages the stage on every frame: measured -28 dB from 4-12 kHz on a real
    dialogue tape, speech that sounds under water. The stage warns and uses the 6 dB cathar
    suggests instead of shipping a muffled track.
    """
    with (
        patch("modules.cathar._run_cathar_step") as mock_step,
        patch("modules.cathar.log_msg") as mock_log,
    ):
        cathar._cathar_deesser_step(tmp_path / "in.wav", tmp_path, bands=3, freq=4000, threshold=-24.0)
    assert mock_step.call_args[0][0] == ["deesser", "--bands", "3", "-f", "4000", "--threshold=6.0"]
    assert mock_log.call_args.kwargs.get("is_error") is True
    assert "4000 Hz" in mock_log.call_args[0][0]


def _read_mono_samples_with(fake_wav, samples):
    """Runs _read_mono_samples against a stubbed soundfile reader."""
    mock_info = MagicMock()
    mock_info.frames = 1000
    mock_info.samplerate = 44100
    with patch("modules.cathar.sf.info", return_value=mock_info):
        with patch("modules.cathar.sf.read", return_value=(samples, 44100)):
            return cathar._read_mono_samples(fake_wav)


def test_read_mono_samples_downmixes_stereo(tmp_path):
    """Verifies _read_mono_samples downmixes stereo to bounded float32 mono."""
    import numpy as np

    mono, sr, source_start = _read_mono_samples_with(tmp_path / "test.wav", np.ones((500, 2), dtype=np.float32))
    assert mono.shape == (500,)
    assert sr == 44100
    assert source_start == 0


def test_read_mono_samples_passes_mono_through(tmp_path):
    """Verifies _read_mono_samples leaves already-mono input unchanged."""
    import numpy as np

    mono, sr, source_start = _read_mono_samples_with(tmp_path / "test.wav", np.ones((500,), dtype=np.float32))
    assert mono.shape == (500,)
    assert sr == 44100
    assert source_start == 0


def test_noiseprint_helpers_and_error_handling(tmp_path):
    """Tests _extract_noiseprint_slice, _execute_noiseprint, and noiseprint error handling."""
    slice_wav = tmp_path / "slice.wav"
    out_json = tmp_path / "out.np.json"

    with (
        patch("modules.cathar._find_quiet_window", return_value=1.5),
        patch("modules.cathar.run_command_with_progress") as mock_run,
        patch("modules.cathar.is_valid_audio", return_value=True),
    ):
        assert cathar._extract_noiseprint_slice(tmp_path / "in.wav", slice_wav, 0.75) is True
        mock_run.assert_called_once()

    with patch("modules.cathar.run_command_with_progress") as mock_run:
        out_json.write_text("{}")
        assert cathar._execute_noiseprint(slice_wav, out_json) == out_json
        mock_run.assert_called_once()

    # Exception inside _cathar_noiseprint_step logs warning and returns None
    in_wav = tmp_path / "in_exc.wav"
    with patch("modules.cathar._extract_noiseprint_slice", side_effect=RuntimeError("Extraction crashed")):
        assert cathar._cathar_noiseprint_step(in_wav, tmp_path) is None


def test_probe_duration_keeps_short_material_on_the_shipped_window(tmp_path):
    """A 15 s corpus clip and a 60 s excerpt keep the 0.75 s cathar shipped with, exactly, so their output stays bit-identical."""
    assert cathar._probe_duration_s(15.0, tmp_path / "x.wav", cap_s=4.0) == 0.75
    assert cathar._probe_duration_s(15.032, tmp_path / "x.wav", cap_s=4.0) == 0.75
    assert cathar._probe_duration_s(60.0, tmp_path / "x.wav", cap_s=4.0) == 0.75
    assert cathar._probe_duration_s(79.9, tmp_path / "x.wav", cap_s=4.0) == 0.75


def test_probe_duration_gives_a_tape_the_full_cap(tmp_path):
    """From 20x the window (80 s for a 4 s cap) the tape gets the full cap; a cap at or under 0.75 s is honoured as is."""
    assert cathar._probe_duration_s(80.0, tmp_path / "x.wav", cap_s=4.0) == 4.0
    assert cathar._probe_duration_s(134.0, tmp_path / "x.wav", cap_s=4.0) == 4.0
    assert cathar._probe_duration_s(7200.0, tmp_path / "x.wav", cap_s=4.0) == 4.0
    assert cathar._probe_duration_s(134.0, tmp_path / "x.wav", cap_s=0.5) == 0.5


def test_probe_duration_falls_back_to_the_header_then_the_floor(tmp_path):
    with patch("modules.cathar.sf.info", return_value=type("I", (), {"duration": 200.0})()):
        assert cathar._probe_duration_s(None, tmp_path / "x.wav", cap_s=4.0) == 4.0
    with patch("modules.cathar.sf.info", side_effect=OSError("no header")):
        assert cathar._probe_duration_s(0, tmp_path / "x.wav", cap_s=4.0) == 0.75


def test_pipeline_learns_the_noise_print_from_a_material_bounded_window(tmp_path):
    """The pipeline hands the noiseprint step the bounded window, not the raw cap."""
    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_precondition_pass", return_value=tmp_path / "a.wav"),
        patch("modules.cathar._cathar_repair_pass", return_value=tmp_path / "b.wav"),
        patch("modules.cathar._cathar_noiseprint_step", return_value=tmp_path / "np.json") as mock_np,
        patch("modules.cathar._cathar_denoise_step", return_value=tmp_path / "c.wav"),
        patch("modules.cathar._cathar_polish_pass", return_value=tmp_path / "d.wav"),
        patch.object(cathar, "CATHAR_ENABLE_NOISEPRINT", True),
    ):
        cathar.filter_cathar_vhs_pipeline(tmp_path / "in.wav", tmp_path, total_duration=15.0)
    assert mock_np.call_args.kwargs["duration_s"] == 0.75
    assert mock_np.call_args.kwargs["stitched"] is False


def _tape_with_pauses(path, seconds=40.0, rate=8000):
    """Hiss throughout, with 1.5 s speech-like bursts every 5 s; the pauses between them are noise."""
    rng = np.random.default_rng(7)
    total = int(seconds * rate)
    hiss = rng.normal(0.0, 0.01, total).astype(np.float32)
    t = np.arange(total) / rate
    burst = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    gate = ((t % 5.0) < 1.5).astype(np.float32)
    sf.write(str(path), np.stack([hiss + burst * gate] * 2, axis=1), rate, subtype="FLOAT")
    return gate


def _stitched_probe(tmp_path):
    tape = tmp_path / "tape.wav"
    gate = _tape_with_pauses(tape)
    probe = tmp_path / "probe.wav"
    with patch("modules.cathar.log_msg") as mock_log:
        assert cathar._extract_stitched_noiseprint(tape, probe, 6.0) is True
    stitched, rate = sf.read(str(probe), dtype="float32", always_2d=True)
    return gate, stitched, rate, mock_log.call_args[0][0]


def test_stitched_noise_print_has_eight_crossfaded_windows(tmp_path):
    """Eight 0.75 s windows joined with 10 ms crossfades, both channels kept."""
    _gate, stitched, rate, message = _stitched_probe(tmp_path)
    window, crossfade = int(0.75 * rate), int(0.01 * rate)
    assert len(stitched) == 8 * window - 7 * crossfade
    assert stitched.shape[1] == 2
    assert "stitched from 8 pauses" in message


def _probe_starts(message):
    return [float(s) for s in message.split(" at ")[1].rstrip(" s").split(", ")]


def test_stitched_noise_print_is_quiet_and_made_of_eight_distinct_windows(tmp_path):
    """A single long window on dialogue would carry speech; the stitched print stays at pause level."""
    _gate, stitched, _rate, message = _stitched_probe(tmp_path)
    assert float(np.sqrt(np.mean(stitched**2))) < 0.02
    assert len(set(_probe_starts(message))) == 8


def test_stitched_noise_print_windows_all_sit_inside_pauses(tmp_path):
    """Every window comes from a quiet gap, never from a burst."""
    gate, _stitched, rate, message = _stitched_probe(tmp_path)
    window = int(0.75 * rate)
    begins = [int(start * rate) for start in _probe_starts(message)]
    assert all(gate[begin:][:window].max() == 0.0 for begin in begins)


def test_stitched_noise_print_refuses_an_empty_file(tmp_path):
    """Nothing to learn from is a bypass, not a crash."""
    tape = tmp_path / "empty.wav"
    sf.write(str(tape), np.zeros((100, 2), dtype=np.float32), 8000, subtype="FLOAT")
    assert cathar._extract_stitched_noiseprint(tape, tmp_path / "probe.wav", 6.0) is False


def test_noiseprint_step_uses_the_stitched_probe_when_asked(tmp_path):
    """`stitched=True` routes to the stitched extractor and still runs cathar's noiseprint."""
    out_dir = tmp_path / "work"
    out_dir.mkdir()
    with (
        patch("modules.cathar._extract_stitched_noiseprint", return_value=True) as mock_stitched,
        patch("modules.cathar._extract_noiseprint_slice") as mock_single,
        patch("modules.cathar._execute_noiseprint", return_value=out_dir / "noise_in.np.json") as mock_exec,
    ):
        result = cathar._cathar_noiseprint_step(tmp_path / "in.wav", out_dir, duration_s=6.0, stitched=True)
    assert result == out_dir / "noise_in.np.json"
    assert mock_stitched.call_args[0][2] == 6.0
    mock_single.assert_not_called()
    mock_exec.assert_called_once()
