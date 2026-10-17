"""Inertness from the tapes' material for the self-driving loop: music-only knobs on tapes the app reads as speech.

Both engines key their music settings on one scanner reading, each tape's held partials
(`profile.tonal_persistence`, `modules/tonal_persistence.material_is_music`): a tape at or
above `cathar_music_persistence_min` takes cathar's music profile, and one at or above
`apl_music_persistence_min` takes `apl_music_neural_model` and, with the switch on, the stem
path and its background floor. The scanner prints the reading for every tape it scans
(`modules/auto_scanner.py`: "Held Partials   : 0.0004"), so the incumbent's own render log says
which tapes are music. When none is, every music-only knob is dead on these tapes and each of
its moves only renders the incumbent's audio again (round C4: the 11 cathar music keys on the
four speech tapes, about 3 minutes of render each, every one hashed inert afterwards). A move
of the music floor itself is dead when no tape crosses it. The reading is printed to four
decimals, so a tape within `MARGIN` of a floor keeps every move live, and a log with fewer
readings than tapes (a render cut short) decides nothing.
"""

import re
from pathlib import Path

HELD_RE = re.compile(r"Held Partials\s*:\s*(\d+(?:\.\d+)?)")
# Four printed decimals round by at most 0.00005; ten times that keeps a tape near a floor live.
MARGIN = 0.0005
# Both floors' default in modules/config.py, what an incumbent without the key runs at.
DEFAULT_FLOOR = 0.05
# engine: (the knob holding the music floor, the knobs read only on music material).
MUSIC_ONLY = {
    "cathar": (
        "cathar_music_persistence_min",
        (
            "cathar_music_alpha",
            "cathar_music_enable_noiseprint",
            "cathar_music_enable_coherent",
            "cathar_music_enable_deplosive",
            "cathar_music_enable_deesser",
            "cathar_music_expander_depth_db",
            "cathar_music_crt_notch_q",
            "cathar_music_alpha_high",
            "cathar_music_beta",
            "cathar_music_enable_repair",
            "cathar_music_dewind_cutoff",
        ),
    ),
    # processing._neural_stage picks apl_music_neural_model on music only; apl_stems.wanted runs the stem path
    # (and its background floor) on music only.
    "apl": ("apl_music_persistence_min", ("apl_music_neural_model", "apl_music_stem_path", "apl_music_bg_floor_db")),
}


def held_partials(run_log, n_tapes):
    """The held-partials readings a render log holds, or None when it holds fewer than `n_tapes` (or is missing)."""
    path = Path(run_log)
    if not path.is_file():
        return None
    values = [float(value) for value in HELD_RE.findall(path.read_text(encoding="utf-8", errors="replace"))]
    return values if len(values) >= n_tapes else None


def _side(reading, floor):
    """1 at or above the floor, 0 below it, None within MARGIN of it."""
    if abs(reading - float(floor)) <= MARGIN:
        return None
    return int(reading >= float(floor))


def _floor(engine, incumbent):
    value = incumbent.get(MUSIC_ONLY[engine][0])
    return DEFAULT_FLOOR if value is None else value


def all_speech(readings, floor):
    """Whether every reading stands clear below `floor`."""
    return readings is not None and all(_side(reading, floor) == 0 for reading in readings)


def material_inert(engine, incumbent, readings):
    """The music-only knobs of `engine` that are dead when no tape reaches the incumbent's music floor."""
    if engine not in MUSIC_ONLY or not all_speech(readings, _floor(engine, incumbent)):
        return set()
    return set(MUSIC_ONLY[engine][1])


def dead_floor_move(engine, knob, value, incumbent, readings):
    """Whether moving the music floor to `value` leaves every tape, clear of both floors, on the side it was."""
    if readings is None or engine not in MUSIC_ONLY or knob != MUSIC_ONLY[engine][0]:
        return False
    current = _floor(engine, incumbent)
    return all(_stays(reading, current, value) for reading in readings)


def _stays(reading, current, value):
    """Whether a tape clear of the current floor lands on the same side of `value`."""
    side = _side(reading, current)
    return side is not None and side == _side(reading, value)


def speech_note(engine, incumbent, readings):
    """The log line saying the music knobs were left out, or "" when they were not."""
    if not material_inert(engine, incumbent, readings):
        return ""
    floor = _floor(engine, incumbent)
    return f"material: every tape reads as speech (held partials at most {max(readings):.4f}, music floor {floor:g}); music knobs left out"
