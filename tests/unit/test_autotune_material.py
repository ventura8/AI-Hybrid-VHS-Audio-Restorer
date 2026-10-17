"""Inertness from the tapes' material: music-only knobs left out when the incumbent's render log reads every tape as speech."""

from types import SimpleNamespace

from scripts import autotune_material as material
from scripts import autotune_restoration as at

SPEECH = [0.0004, 0.0005, 0.0055, 0.0011]
CATHAR_MUSIC = set(material.MUSIC_ONLY["cathar"][1])
APL_MUSIC = set(material.MUSIC_ONLY["apl"][1])


def _log(path, readings, extra=""):
    lines = [f"\x1b[K    - Held Partials   : {value:.4f} (music holds 0.06 and up)" for value in readings]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["scan", *lines, extra]), encoding="utf-8")
    return path


def test_the_readings_come_from_the_scanner_lines_of_the_render_log(tmp_path):
    """One reading per scanned tape; a log with fewer readings than tapes, or none at all, decides nothing."""
    log = _log(tmp_path / "run.log", SPEECH)
    assert material.held_partials(log, 4) == SPEECH
    assert material.held_partials(log, 5) is None
    assert material.held_partials(tmp_path / "missing.log", 1) is None


def test_speech_tapes_make_every_music_only_knob_dead():
    """All four Tata readings sit far under the 0.05 floor: cathar's music keys and APL's music model and stem path go."""
    assert material.material_inert("cathar", {}, SPEECH) == CATHAR_MUSIC
    assert material.material_inert("apl", {"apl_music_persistence_min": 0.05}, SPEECH) == APL_MUSIC


def test_one_music_tape_a_reading_at_the_floor_or_no_readings_keep_every_knob_live():
    """A tape over the floor, a tape within the rounding margin of it, a missing log or an unknown engine hide nothing."""
    assert material.material_inert("cathar", {}, [*SPEECH, 0.12]) == set()
    assert material.material_inert("cathar", {}, [*SPEECH, 0.0498]) == set()
    assert material.material_inert("cathar", {}, None) == set()
    assert material.material_inert("ffmpeg", {}, SPEECH) == set()


def test_a_floor_move_is_dead_only_when_no_tape_crosses_it():
    """0.05 to 0.08 leaves every speech tape below both; to 0.004 the 0.0055 tape would turn music."""
    incumbent = {"cathar_music_persistence_min": 0.05}
    assert material.dead_floor_move("cathar", "cathar_music_persistence_min", 0.08, incumbent, SPEECH)
    assert not material.dead_floor_move("cathar", "cathar_music_persistence_min", 0.004, incumbent, SPEECH)
    assert not material.dead_floor_move("cathar", "cathar_alpha", 0.08, incumbent, SPEECH)
    assert not material.dead_floor_move("cathar", "cathar_music_persistence_min", 0.08, incumbent, None)


def test_the_note_names_the_loudest_reading_and_the_floor():
    """The log line appears only when the music knobs were left out."""
    assert material.speech_note("cathar", {}, SPEECH) == (
        "material: every tape reads as speech (held partials at most 0.0055, music floor 0.05); music knobs left out"
    )
    assert material.speech_note("cathar", {}, [0.2]) == ""


def test_the_loop_proposes_no_music_move_on_speech_tapes():
    """With the readings the music keys go, and of the floor's moves only 0.005 stays: the 0.0055 tape would cross it."""
    with_readings = at.neighbour_moves("cathar", {}, None, SPEECH)
    without = {knob for knob, _ in at.neighbour_moves("cathar", {})}
    assert not {knob for knob, _ in with_readings} & CATHAR_MUSIC
    assert [value for knob, value in with_readings if knob == "cathar_music_persistence_min"] == [0.005]
    # The split-band factor is dead anyway with the split band off (INERT_WHEN).
    assert CATHAR_MUSIC - {"cathar_music_alpha_high"} <= without


def _round(tmp_path, monkeypatch):
    """A cathar round over four tapes whose incumbent log reads speech; returns what was rendered and proposed."""
    incumbent = {"cathar_alpha": 1.0}
    _log(tmp_path / "cands" / at.candidate_id(incumbent) / "run.log", SPEECH)
    seen = SimpleNamespace(rendered=[], proposed=[])
    monkeypatch.setattr(at, "run_candidate", lambda _engine, overrides, *_args: seen.rendered.append(overrides))
    monkeypatch.setattr(at, "propose", lambda *args: seen.proposed.append(args) or {})
    state = {"incumbent": incumbent, "rounds": [], "candidates": {}}
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, durations=None)
    at.run_round(args, "cathar", {f"t{i}": f"t{i}.mov" for i in range(4)}, tmp_path, state, {"m": "up"}, 1)
    return seen


def test_a_round_renders_the_incumbent_and_proposes_with_its_readings(tmp_path, monkeypatch):
    """The incumbent renders first (here a stand-in) and its log's readings reach propose."""
    seen = _round(tmp_path, monkeypatch)
    assert (seen.rendered, seen.proposed[0][3]) == ([{"cathar_alpha": 1.0}], SPEECH)


def test_a_round_says_when_the_music_knobs_were_left_out(tmp_path, monkeypatch, capsys):
    """The material note is printed with the round."""
    _round(tmp_path, monkeypatch)
    assert "every tape reads as speech" in capsys.readouterr().out
