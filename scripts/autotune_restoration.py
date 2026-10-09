"""Self-driving fine-tuning of one engine on whole tapes, judged by the output-quality harness.

Each round proposes one-knob moves from the incumbent settings (the neighbours of every
knob's current value, plus the combination of every move that helped on its own), restores
every tape with each candidate in a fresh interpreter, keeps only the audio, scores it with
`scripts/restoration_quality` (all tapes of one candidate in parallel) and ranks candidates
against the incumbent with the grid's `ranking` metrics. A candidate replaces the incumbent
when its mean rank across tapes is better, it wins on at least half the tapes and it fails
no more hard gates than the incumbent. The loop stops when no candidate qualifies (a
plateau); `--rounds` (default 20) is only a safety cap. The listener is asked at the plateau.

An ear v3 grid (`tune_grids/tata_v3.yaml`, `music_v3.yaml`) adds three guards to that verdict
(plan 1.5, `scripts/autotune_guards.py`); a v1 or v2 grid has none of these sections and is
judged as before. `reversals:` refuses, before it is rendered, a move that heads past a ledger
boundary the way the user rejected (`linear_air_gain_db` up to 2.0 or above: round 3 heard
+1 dB over +2 dB, all at 7500 Hz, so another `linear_air_freq_hz` is judged as the 7500 Hz shelf
lifting 4-8 kHz as much, `autotune_guards.ledger_air_gain`). `audibility:` makes a candidate
`auditory.compare_files` cannot tell from the incumbent on any tape a tie, never scored and
never accepted. `vetoes:` keeps a candidate
from winning when a learned judge's median moved past `floor_multiple` x its benign floor
(`--noise-floors`, a `scripts/reward_noise_floor.py` report) on some tape; a veto without a
floor stops the loop before the first render.

Inert candidates are found twice. Before rendering, `INERT_WHEN` leaves out the knobs the
incumbent's switches make dead (a de-esser threshold with the de-esser off, the air gain with
the shelf off, the split-band factors under the Wiener denoiser), and `DURATION_BOUND` leaves
out the moves the tapes' own length makes dead (cathar's print length on tapes too short to
use it). After rendering, every candidate's decoded audio is hashed per tape
(`audio_fingerprint`); a candidate that sounds exactly like the incumbent, or like a candidate
proposed before it, on every tape is marked inert, logged in `log.md` as a knob-table finding
and never scored. Inert renders cost about 30% of a round before (ear v3 design, section 5.2);
a round then took about 4 h for ~21 APL candidates on four tapes: ~205 s of render per
candidate, the rest scoring (design, section 7).

The hash (`exact_audio_sha256`) is the loop's own and exact: the samples as stored (float as
float, integer PCM as int32) with the rate, channel count and sample type, read in blocks of
`HASH_BLOCK_FRAMES`. The harness's `auditory.audio_sha256` is not used here: it hashes
`audio_io.load_audio`, which removes each channel's DC offset, so two renders that differ by
an offset alone would read as one and an audible-or-not question would be answered "inert"
(design 5.2 asks for a byte-identical match); and it decodes the whole file at once, about
1.3 GB of float32 per hour of stereo tape per copy and three copies on the way, while the
loop hashes whole tapes beside two scorers of ~4 GB each.

Knob-table facts the tables below rest on (2026-10-09, read from the code paths):

- `apl_tonal_flatness_max` is live with the subtraction stage off: the plosive tamer
  (`plosive_tamer._skip_reason`) and the hum canceller's series length
  (`hum_cancel._series_length`) read it. The tuned finals carry 0.01, the shipped default is
  0.035 and the Tata tapes read 0.022, so tuned APL ran the tamer and the 40-harmonic hum
  series while shipped APL does not.
- `apl_music_persistence_min` is live with the stem path off: it decides which tapes take
  `apl_music_neural_model` (`processing._neural_stage`).
- `apl_noiseprint_tonal_s` is live with the subtraction stage off once the stem path is on:
  it is the probe of the background suppressor (`apl_stems._background_pass`), which runs
  while `apl_music_bg_floor_db` is below 0.
- `cathar_noiseprint_duration_s` only applies to a tape at least
  `cathar.NOISEPRINT_MIN_MATERIAL_RATIO` (20) times longer than it; a shorter tape learns from
  `cathar.NOISEPRINT_SHORT_S` (0.75 s) whatever the knob says (`cathar._probe_duration_s`).
  4.5 s needs 90 s of tape and 6 s needs 120 s, so on a set of short clips the knob is dead.
  The tapes' lengths come from ffprobe; a tape within `DURATION_MARGIN_S` of the switch, or
  one ffprobe cannot read, keeps every move live.
- `env:AI_RESTORE_CATHAR_BIN [None, 0.7.6]` rendered the same binary twice once the
  installers provisioned 0.7.6; a binary knob returns only for a new build (round C0).
- `apl_neural_model [None, ROFORMER, ROFORMER_AGGR]` listed one setting twice once the listener
  round made the Mel-RoFormer the app's default (`DEFAULT_APL_NEURAL_MODEL`): from a start
  without the key every round proposed the default under its file name, a render the hash
  then found inert, and never the aggressive model. The list is `[ROFORMER, ROFORMER_AGGR]`,
  seeded from the app's resolved value, so its one move is the live alternative. A list that
  holds None and the app's own default names one setting twice.
- The cathar music profile's de-esser switch, expander depth and CRT notch width were not
  knobs, so the music rounds could not move them.
- The air shelf's corner and the polish expander's attack and decay were hard-coded in
  `modules/filters.py` (7500 Hz, 0.04 / 0.18 s); since 2026-10-09 they are
  `linear_air_freq_hz`, `expander_attack_s` and `expander_decay_s`, whose defaults build the
  same filter strings. The shelf is APL's alone (cathar's polish runs without air); the
  expander timing is shared, since cathar's mode runs the same polish expander
  (`modes/cathar.py` -> `processing._polish_full_audio_step`).

`--stage-cache DIR` (plan 1.5) points every candidate's render at one neural-stage cache
(`AI_RESTORE_STAGE_CACHE`, `modules/stage_cache.py`; `--stage-cache-gb` caps it, 50 GB by
default): an auto_pure_linear candidate whose knobs all act after the neural denoiser (air,
sibilant guard, pause floor, expanders, loudness range) replays the incumbent's chain and model
output instead of rendering them. Each candidate's `timing.json` records the hits and misses
its `run.log` reports. cathar never uses the cache. Off by default.

Not built yet (plan 1.5, design 5.1): inertness from each tape's material. The `cathar_music_*`
keys and `apl_music_neural_model` are dead on a tape the scanner reads below the music
persistence floor (`modules/tonal_persistence.py`), and cathar's speech keys on a clip that
takes the music profile; telling them apart before the render needs the scanner's tonal
persistence per tape (the R0 / scanner reading), which the loop does not read yet. Until it
does, a speech-only tape set pays a render for each of those moves every round (~205 s per APL
candidate on four tapes) and only the hash keeps them from being scored.

usage:
  autotune_restoration.py --engine cathar|apl --tapes tapes.json [--rounds 20] [--out experiments/autotune]
                          [--grid scripts/tune_grids/tata_v1.yaml] [--families dsp,stems,speech,mos]
                          [--start '{"cathar_alpha": 2.0}' | --start-file final.json] [--language ro] [--gates gates.json]
                          [--parallel 2] [--noise-floors experiments/reward/noise_floor_benign.json]
                          [--stage-cache experiments/autotune/stage_cache] [--stage-cache-gb 50]

`tapes.json` maps a slug to a video path. Everything is resumable: candidates are keyed by
their settings, and a scored candidate is never run or scored twice.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Imported by name once the repository is on sys.path, so the script runs from any directory.
tune = import_module("scripts.tune_restoration")
audio_io = import_module("scripts.restoration_quality.audio_io")
guards_mod = import_module("scripts.autotune_guards")
cathar = import_module("modules.cathar")
stage_cache = import_module("modules.stage_cache")

ENV_PREFIX = "env:"
ROFORMER = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
ROFORMER_AGGR = "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"
SCORER_PARALLEL = 2
LOG_MD = "log.md"
HASH_SIDECAR_SUFFIX = ".audio_sha256.json"
HASH_BLOCK_FRAMES = 1 << 20
# Recorded in every sidecar: a sidecar written by another hash function (the harness's DC-blind one) is never trusted.
HASH_NAME = "exact-pcm"
# ffprobe's container length and the length the app reads may differ by a fraction of a second.
DURATION_MARGIN_S = 1.0
# Model-name knobs are bare file names; `tune_restoration.known_config_fields` types only apl_neural_model.
MODEL_KNOBS = ("apl_neural_model", "apl_music_neural_model")

# Knobs both engines share: the polish expander (the stage that turns a denoised pause into
# dead air), the mux's loudness-range target (loudnorm drops to dynamic mode above it and rides
# the gain between words) and the CRT notch width.
SHARED_KNOBS = {
    "enable_dynamic_expander": [True, False],
    "expander_depth_db": [4.0, 7.0, 12.0],
    "expander_knee_offset_db": [0.0, 4.0, 8.0],
    # The expander's compand timing (round A3, pause texture), hard-coded 0.04 / 0.18 s until 2026-10-09.
    "expander_attack_s": [0.02, 0.04, 0.08],
    "expander_decay_s": [0.12, 0.18, 0.3],
    "loudnorm_target_lra": [11.0, 20.0, 40.0],
    "crt_notch_q": [30.0, 60.0, 120.0],
    # The pause floor keeper (modules/pause_floor.py): the source's own pause texture put back
    # this many dB under its level, so a pause never collapses to dead air.
    "enable_pause_floor": [False, True],
    "pause_floor_fill_db": [8.0, 12.0, 18.0, 24.0],
}

# Ordered candidate values per knob; None means "the app's default" (the override is dropped).
KNOBS = {
    "cathar": {
        # 0.5 and 1.0, and no print at all, exist for music: on a music-dominant clip the print
        # learned from the music's own quietest window removes no noise and only shaves the top
        # octave (alpha 1.5: -14 dB at 8-16 kHz, quiet frames +1.3 dB; alpha 0.5: -1.6 dB).
        "cathar_alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "cathar_enable_noiseprint": [True, False],
        "cathar_beta": [0.005, 0.01, 0.02],
        "cathar_deesser_threshold": [6.0, 9.0, 12.0],
        # Round C2 (plan Part 2): where the multiband de-esser starts and how many bands it splits into.
        "cathar_deesser_freq": [4000, 5000, 6000],
        "cathar_deesser_bands": [1, 3],
        "cathar_enable_deesser": [True, False],
        "cathar_noiseprint_duration_s": [4.5, 6.0],
        "cathar_enable_coherent": [True, False],
        "cathar_enable_enhance": [True, False],
        "cathar_enable_deplosive": [True, False],
        "cathar_repair_strength": [2, 4],
        # The music profile (modules/cathar.py) overrides cathar_alpha, the print, the coherent
        # path, the deplosive and the de-esser on every clip whose held partials reach the floor,
        # and sets its own expander depth and CRT notch width, so on a music clip only these keys
        # move; the rounds before they were knobs moved nothing there.
        "cathar_music_alpha": [0.25, 0.5, 1.0],
        "cathar_music_enable_noiseprint": [False, True],
        "cathar_music_enable_coherent": [False, True],
        "cathar_music_enable_deplosive": [False, True],
        "cathar_music_enable_deesser": [False, True],
        "cathar_music_expander_depth_db": [4.0, 7.0, 12.0],
        "cathar_music_crt_notch_q": [30.0, 60.0, 120.0],
        "cathar_music_persistence_min": [0.005, 0.02, 0.05],
        # Split-band subtraction (modules/split_band.py): the highs get their own factor.
        "cathar_split_band_hz": [0, 4000, 6000, 8000],
        "cathar_alpha_high": [0.5, 1.0, 1.5, 2.0, 3.0],
        "cathar_music_alpha_high": [0.25, 0.5, 1.0],
        **SHARED_KNOBS,
    },
    # On the Tata tapes the scanner reads a spectral flatness of 0.022 (< apl_tonal_flatness_max 0.035),
    # so APL takes its tonal path: apl_spectral_alpha_tonal and apl_noiseprint_tonal_s are the live
    # knobs there (apl_spectral_alpha and apl_noiseprint_duration_s produced bit-identical audio),
    # and the flatness threshold itself decides whether an interview is treated as tonal at all, in
    # the subtraction stage, the plosive tamer and the hum canceller alike. The tonal probe length
    # also feeds the stem path's background suppressor (apl_stems._background_pass).
    "apl": {
        "apl_spectral_alpha_tonal": [1.5, 2.0, 2.5, 3.0],
        "apl_noiseprint_tonal_s": [2.5, 4.0, 6.0],
        "apl_tonal_flatness_max": [0.01, 0.035],
        "enable_linear_air": [True, False],
        # Round A1 (brightness, plan Part 2: 0 / 0.5 / 1.0 / 1.5 dB): the shelf's gain around the by-ear +1 dB
        # (0 is the enable_linear_air False neighbour; +2 dB, which round 3 rejected, is not proposed, so a v1 / v2
        # grid without the reversal cannot step to it) and its corner, hard-coded at 7500 Hz until 2026-10-09. The
        # v3 grids' reversal judges the pair as one shelf (autotune_guards.ledger_air_gain): +1.5 dB at 6000 Hz is refused.
        "linear_air_gain_db": [0.5, 1.0, 1.5],
        "linear_air_freq_hz": [6000.0, 7500.0, 9000.0],
        # The app's default model is ROFORMER (seeded from the app), so the one move is the aggressive model.
        "apl_neural_model": [ROFORMER, ROFORMER_AGGR],
        # The model on music (empty, the default, follows the chain's own choice).
        "apl_music_neural_model": [None, ROFORMER],
        "apl_enable_learned_blend": [True, False],
        "apl_enable_spectral_denoise": [True, False],
        "apl_use_native_suppress": [False, True],
        # The native suppressor's gain floor: what a pause keeps once the subtraction slot is on.
        "apl_suppress_gain_floor_db": [-30.0, -20.0, -12.0],
        # The sibilant guard (modules/sibilant_guard.py): the 's' keeps its body under the neural stage.
        # The listener round stopped at mix 0.8, the top of its grid, and the user still heard APL's
        # 's' thin (2026-10-05); the harness traces the thinning to the 1-4 kHz body, below the
        # guard's 4 kHz crossover, so the crossover is a knob and the mix reaches 1.0.
        "apl_enable_sibilant_guard": [False, True],
        "apl_sibilant_mix": [0.3, 0.5, 0.8, 0.9, 1.0],
        "apl_sibilant_guard_hz": [2000, 2500, 3000, 4000, 5000],
        # The fricative detector's high-band share: at 0.5 the guard found 12.5% of the harness's
        # fricatives on Vaccin, so mix and crossover moves barely touched the audio (-70..-100 dBFS).
        "apl_sibilant_hf_share_min": [0.2, 0.3, 0.4, 0.5],
        # This mode's own expander depth (the shared expander_depth_db is cathar's on speech).
        "apl_expander_depth_db": [7.0, 12.0, 18.0],
        # The stem path on music (modules/apl_stems.py); 0.005 admits Gaudeamus (held partials 0.009).
        "apl_music_stem_path": [False, True],
        "apl_music_persistence_min": [0.005, 0.02, 0.05],
        "apl_music_bg_floor_db": [-20.0, -10.0, -5.0, 0.0],
        **SHARED_KNOBS,
    },
}
# Every setting a grid's reversal may name: a key no engine tunes would never fire, so it is refused as a typo.
ALL_KNOBS = frozenset(knob for table in KNOBS.values() for knob in table)
ENGINES = {
    "cathar": {"process_mode": "cathar", "suffix": "_Cathar_Cleaned"},
    "apl": {"process_mode": "auto_pure_linear", "suffix": "_PureLinear_Cleaned"},
}
PYTHON = str(REPO / ".venv" / "Scripts" / "python.exe") if (REPO / ".venv" / "Scripts" / "python.exe").exists() else sys.executable


# ----------------------------------------------------------------------------- candidates


def _without_defaults(overrides):
    """The overrides that set something: a None value means the app's own default."""
    return {k: v for k, v in overrides.items() if v is not None}


def candidate_id(overrides):
    """A stable short id for a set of overrides (None values dropped first)."""
    clean = _without_defaults(overrides)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]


def split_overrides(overrides):
    """`(config_overrides, env)`: `env:` keys become environment for the app's children."""
    chosen = _without_defaults(overrides)
    config = {k: v for k, v in chosen.items() if not k.startswith(ENV_PREFIX)}
    env = {k.removeprefix(ENV_PREFIX): v for k, v in chosen.items() if k.startswith(ENV_PREFIX)}
    return config, env


def validation_problems(config):
    """Problems with config overrides against the app's typed settings, model-name knobs included."""
    fields = {**tune.known_config_fields(), **{name: (str, None, None) for name in MODEL_KNOBS}}
    return tune.validate_overrides(config, fields)


def seed_defaults(engine, start, out_dir):
    """`start` completed with the app's own value for every numeric knob it does not name.

    A knob whose candidates include None (the app's default) needs no seeding; the others
    are read back from a child interpreter so the incumbent is the real starting point and
    a neighbour never re-runs the default under another name.
    """
    seeded = dict(start)
    missing = _unseeded(engine, seeded)
    if not missing:
        return seeded
    resolved = _app_defaults(engine, out_dir)
    for knob in missing:
        match = _matching_value(resolved.get(knob), KNOBS[engine][knob])
        if match is not None:
            seeded[knob] = match
    return seeded


def _unseeded(engine, seeded):
    """The knobs `seeded` leaves at the app's default (unnamed or None) whose candidates do not list that default as None.

    A saved state from before a table change may hold None for such a knob (the old
    `apl_neural_model` list offered it); it is seeded like a knob it does not name.
    """
    return [knob for knob, values in KNOBS[engine].items() if seeded.get(knob) is None and None not in values]


def _app_defaults(engine, out_dir):
    """CONFIG as the app resolves it for this engine with no override."""
    probe = out_dir / "defaults"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / tune.CONFIG_YAML).write_text(
        tune.materialise_config((REPO / tune.CONFIG_YAML).read_text(encoding="utf-8"), ENGINES[engine]["process_mode"], {}),
        encoding="utf-8",
    )
    return tune.resolved_config(probe, PYTHON)


def _matching_value(seen, values):
    """The candidate value the app resolved to, or None."""
    for value in values:
        same = str(seen).lower() == str(value).lower() if isinstance(value, bool) else tune._close(seen, value)
        if same:
            return value
    return None


# Knobs that cannot change the audio while every switch in `when` holds the value given there:
# candidates moving them render byte-identical output and only cost a render and a scoring pass
# each. The switches stay knobs, so the loop can still turn the stage back on and then move
# them. Values absent from the incumbent read as the app's defaults (SWITCH_DEFAULTS: what the
# repository's config.yaml resolves to). An entry is written only from every code path that
# reads the knob; a knob dead on some material only (a music key on a speech tape) is left to
# the audio hash until the loop reads the material (see the module docstring), so a live knob
# is never hidden.
# Read only inside spectral_denoise, behind apl_enable_spectral_denoise. apl_noiseprint_tonal_s is
# not here: the stem path's background suppressor reads it too (its own entry below).
SUBTRACTION_KNOBS = (
    "apl_spectral_alpha_tonal",
    "apl_enable_learned_blend",
    "apl_use_native_suppress",
    "apl_suppress_gain_floor_db",
)
_CATHAR_MUSIC_KEYS = (
    "cathar_music_alpha",
    "cathar_music_enable_noiseprint",
    "cathar_music_enable_coherent",
    "cathar_music_enable_deplosive",
    "cathar_music_enable_deesser",
    "cathar_music_expander_depth_db",
    "cathar_music_crt_notch_q",
    "cathar_music_persistence_min",
    "cathar_music_alpha_high",
)
# The polish expander's knee and compand timing (filters._build_full_audio_expander_filter), shared by both engines.
EXPANDER_SHAPE_KNOBS = ("expander_knee_offset_db", "expander_attack_s", "expander_decay_s")
INERT_WHEN = {
    "apl": (
        ({"apl_enable_spectral_denoise": False}, SUBTRACTION_KNOBS),
        # The tonal probe feeds the subtraction stage and the stem path's background suppressor, which
        # runs only on the stem path and only with a floor below 0 (apl_stems._background_pass).
        ({"apl_enable_spectral_denoise": False, "apl_music_stem_path": False}, ("apl_noiseprint_tonal_s",)),
        ({"apl_enable_spectral_denoise": False, "apl_music_bg_floor_db": 0.0}, ("apl_noiseprint_tonal_s",)),
        # Only the native suppressor reads its gain floor (spectral_denoise._native_suppress).
        ({"apl_use_native_suppress": False}, ("apl_suppress_gain_floor_db",)),
        # apl_music_persistence_min is not here: it also picks the tapes that take apl_music_neural_model.
        ({"apl_music_stem_path": False}, ("apl_music_bg_floor_db",)),
        ({"apl_enable_sibilant_guard": False}, ("apl_sibilant_mix", "apl_sibilant_guard_hz", "apl_sibilant_hf_share_min")),
        ({"enable_pause_floor": False}, ("pause_floor_fill_db",)),
        # filters._append_expander_stage builds no compand with the expander off: its depth, knee and timing are dead.
        ({"enable_dynamic_expander": False}, ("apl_expander_depth_db",) + EXPANDER_SHAPE_KNOBS),
        # filters._build_linear_air_filter returns no shelf with the air off, whatever the gain or corner.
        ({"enable_linear_air": False}, ("linear_air_gain_db", "linear_air_freq_hz")),
    ),
    "cathar": (
        ({"cathar_split_band_hz": 0}, ("cathar_alpha_high", "cathar_music_alpha_high")),
        # `denoise --wiener` takes no alpha or beta, and the split band needs a factor to split.
        (
            {"cathar_denoise_method": "wiener"},
            ("cathar_alpha", "cathar_beta", "cathar_music_alpha", "cathar_split_band_hz", "cathar_alpha_high", "cathar_music_alpha_high"),
        ),
        # Speech takes cathar_enable_deesser, music cathar_music_enable_deesser: dead only with both off.
        (
            {"cathar_enable_deesser": False, "cathar_music_enable_deesser": False},
            ("cathar_deesser_threshold", "cathar_deesser_freq", "cathar_deesser_bands"),
        ),
        ({"cathar_enable_noiseprint": False, "cathar_music_enable_noiseprint": False}, ("cathar_noiseprint_duration_s",)),
        ({"cathar_music_profile": False}, _CATHAR_MUSIC_KEYS),
        ({"enable_pause_floor": False}, ("pause_floor_fill_db",)),
        ({"enable_dynamic_expander": False}, ("expander_depth_db", "cathar_music_expander_depth_db") + EXPANDER_SHAPE_KNOBS),
    ),
}
# Knobs an engine never reads: auto_pure_linear takes apl_expander_depth_db, not the shared depth.
NEVER_READ = {"apl": ("expander_depth_db",), "cathar": ()}
# The repository's config.yaml over modules/config.py's defaults, which is what a candidate starts from.
SWITCH_DEFAULTS = {
    "apl_enable_spectral_denoise": False,
    "apl_use_native_suppress": False,
    "apl_music_stem_path": False,
    "apl_music_bg_floor_db": -10.0,
    "apl_enable_sibilant_guard": True,
    "enable_pause_floor": True,
    "enable_dynamic_expander": True,
    "enable_linear_air": True,
    "cathar_split_band_hz": 0,
    "cathar_denoise_method": "spectral",
    "cathar_enable_deesser": True,
    "cathar_music_enable_deesser": False,
    "cathar_enable_noiseprint": True,
    "cathar_music_enable_noiseprint": False,
    "cathar_music_profile": True,
}


def _switch_value(incumbent, switch):
    value = incumbent.get(switch)
    return SWITCH_DEFAULTS[switch] if value is None else value


def inert_knobs(engine, incumbent, durations=None):
    """The knobs that cannot move this engine's audio under the incumbent's switches and, given `durations`, on these tapes."""
    inert = set(NEVER_READ.get(engine, ()))
    for when, knobs in INERT_WHEN.get(engine, ()):
        if all(_switch_value(incumbent, switch) == value for switch, value in when.items()):
            inert.update(knobs)
    inert.update(_dead_on_these_tapes(engine, incumbent, durations))
    return inert


def cathar_print_s(value, duration_s):
    """Seconds of tape cathar learns its print from with `cathar_noiseprint_duration_s` at `value` on a tape this long.

    `cathar._probe_duration_s`'s rule, read from its own constants. None when the length is
    unknown or within DURATION_MARGIN_S of the switch, so a tape that cannot tell never makes
    a move dead.
    """
    switch_s = float(value) * cathar.NOISEPRINT_MIN_MATERIAL_RATIO
    if duration_s is None or abs(duration_s - switch_s) < DURATION_MARGIN_S:
        return None
    applies = duration_s >= switch_s or float(value) <= cathar.NOISEPRINT_SHORT_S
    return float(value) if applies else cathar.NOISEPRINT_SHORT_S


# Knobs whose effect depends on the tape's length: the setting each value gives a tape of a given length.
DURATION_BOUND = {"cathar_noiseprint_duration_s": cathar_print_s}


def _settings_per_tape(knob, value, durations):
    return tuple(DURATION_BOUND[knob](value, seconds) for seconds in durations.values())


def dead_move(knob, value, incumbent, durations):
    """Whether moving `knob` to `value` leaves every tape on the setting the incumbent's value already gives it."""
    current = incumbent.get(knob)
    if knob not in DURATION_BOUND or current is None or not durations:
        return False
    settings = _settings_per_tape(knob, value, durations)
    return None not in settings and settings == _settings_per_tape(knob, current, durations)


def _every_move_dead(knob, values, incumbent, durations):
    return all(dead_move(knob, v, incumbent, durations) for v in values if v != incumbent.get(knob))


def _dead_on_these_tapes(engine, incumbent, durations):
    """The length-bound knobs of this engine none of whose other values changes the setting of any tape."""
    bound = {knob: values for knob, values in KNOBS[engine].items() if knob in DURATION_BOUND}
    return {knob for knob, values in bound.items() if _every_move_dead(knob, values, incumbent, durations)}


def tape_durations(tapes):
    """`{slug: seconds}` read by ffprobe; None for a tape it cannot read, which keeps every length-bound move live."""
    return {slug: tune.probe_duration(tape) for slug, tape in tapes.items()}


def _neighbours(values, current):
    """The values next to `current` in the ordered list, or every other value when it is not one of them."""
    if current in values:
        index = values.index(current)
        low, high = max(index - 1, 0), index + 2
        values = values[low:high]
    return [v for v in values if v != current]


def neighbour_moves(engine, incumbent, durations=None):
    """`[(knob, value)]`: the values adjacent to each knob's current value (or every value when unset).

    Knobs the incumbent's switches make inert are left out (see INERT_WHEN), and so is a move
    the tapes' lengths make dead (see DURATION_BOUND; `durations` is `{slug: seconds}`).
    """
    moves = []
    inert = inert_knobs(engine, incumbent, durations)
    for knob, values in KNOBS[engine].items():
        if knob not in inert:
            moves += [(knob, v) for v in _neighbours(values, incumbent.get(knob)) if not dead_move(knob, v, incumbent, durations)]
    return moves


def propose(engine, incumbent, durations=None):
    """`{id: overrides}` for every single-knob neighbour of the incumbent that can change the audio on these tapes."""
    proposals = {}
    for knob, value in neighbour_moves(engine, incumbent, durations):
        overrides = {**incumbent, knob: value}
        proposals[candidate_id(overrides)] = overrides
    proposals.pop(candidate_id(incumbent), None)
    return proposals


def _moves(candidate, incumbent):
    """The settings where `candidate` differs from the incumbent."""
    return {k: v for k, v in candidate.items() if v != incumbent.get(k)}


def combine(incumbent, winners):
    """One candidate carrying every move that beat the incumbent on its own (two or more of them)."""
    if len(winners) < 2:
        return {}
    overrides = dict(incumbent)
    for candidate in winners:
        overrides.update(_moves(candidate, incumbent))
    cid = candidate_id(overrides)
    return {cid: overrides} if cid != candidate_id(incumbent) else {}


# ----------------------------------------------------------------------------- running


def run_candidate(engine, overrides, tapes, out_dir):
    """Restores every tape with `overrides` and keeps the audio as WAV under `out_dir/<id>/`; returns the WAV paths."""
    cid = candidate_id(overrides)
    cand_dir = out_dir / "cands" / cid
    wavs = {slug: cand_dir / f"{slug}.wav" for slug in tapes}
    if all(w.exists() for w in wavs.values()):
        return wavs
    spec = {**ENGINES[engine], "env": split_overrides(overrides)[1]}
    extra_env = _prepare_candidate(cid, cand_dir, spec, overrides)
    started = time.time()
    _restore(cand_dir, tapes, extra_env)
    _collect_outputs(cid, tapes, spec["suffix"], wavs)
    timing = {"wall_s": time.time() - started, "stage_cache": stage_cache_counts(cand_dir / "run.log")}
    (cand_dir / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    return wavs


def stage_cache_counts(run_log):
    """The neural-stage cache's hits and misses a candidate's run log reports (both 0 with the cache off)."""
    try:
        text = Path(run_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"hits": 0, "misses": 0}
    return {"hits": text.count("[Stage Cache] Hit "), "misses": text.count("[Stage Cache] Miss ")}


def enable_stage_cache(folder, cap_gb=None):
    """Points every candidate's render at one neural-stage cache; the children inherit it. None leaves it as it is (off)."""
    if folder is None:
        return None
    root = Path(folder).resolve()
    os.environ[stage_cache.ENV_VAR] = str(root)
    if cap_gb is not None:
        os.environ[stage_cache.MAX_GB_ENV] = f"{cap_gb:g}"
    print(f"stage cache: {root} (cap {stage_cache.max_bytes() / stage_cache.GB:g} GB)", flush=True)
    return root


def _prepare_candidate(cid, cand_dir, spec, overrides):
    """Writes the candidate's config, checks the app honours every override; returns the engine's extra environment."""
    config = {"batch_jobs": 1, **split_overrides(overrides)[0]}  # one candidate at a time; the app's own batch stays sequential
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "overrides.json").write_text(json.dumps(overrides, indent=1), encoding="utf-8")
    problems = validation_problems(config)
    if problems:
        raise SystemExit(f"{cid}: {problems}")
    (cand_dir / tune.CONFIG_YAML).write_text(
        tune.materialise_config((REPO / tune.CONFIG_YAML).read_text(encoding="utf-8"), spec["process_mode"], config), encoding="utf-8"
    )
    extra_env = tune.engine_env(spec)
    reverted = tune.assert_overrides_resolved(config, tune.resolved_config(cand_dir, PYTHON, extra_env))
    if reverted:
        raise SystemExit(f"{cid}: the app did not honour {reverted}")
    return extra_env


def _restore(cand_dir, tapes, extra_env):
    """Runs the app on every tape in a fresh interpreter launched in the candidate's folder."""
    with open(cand_dir / "run.log", "w", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON, str(REPO / "restore_audio_hybrid.py"), *(str(Path(p)) for p in tapes.values())],
            cwd=str(cand_dir),
            env={**os.environ, **extra_env, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=tune.RUN_TIMEOUT_PER_EXCERPT_S * 4 * max(1, len(tapes)),
            check=False,
        )


def _collect_outputs(cid, tapes, suffix, wavs):
    """Moves each restored tape's audio into the candidate's WAV and deletes the restored video beside the tape."""
    for slug, tape in tapes.items():
        produced = Path(tape).with_name(f"{Path(tape).stem}{suffix}{Path(tape).suffix}")
        if not produced.exists():
            raise SystemExit(f"{cid}: no output for {slug} ({produced.name})")
        shutil.move(str(audio_io.extract_wav(produced, wavs[slug].parent)), str(wavs[slug]))
        produced.unlink()


# ----------------------------------------------------------------------------- inert detection


def exact_audio_sha256(path):
    """SHA-256 of a file's decoded samples as stored, after its rate, channel count and sample type.

    Two files that decode to the same samples hash alike whatever their headers or tags say,
    and any other two do not: float files are read as stored and integer PCM as int32, so no
    two distinct sample values collapse into one, and nothing is subtracted (a DC offset is a
    difference). Read in blocks so an hour of tape never sits in memory at once.
    """
    digest = hashlib.sha256()
    with sf.SoundFile(str(path)) as audio:
        dtype = {"FLOAT": "float32", "DOUBLE": "float64"}.get(audio.subtype, "int32")
        digest.update(f"{audio.samplerate}|{audio.channels}|{dtype}|".encode("ascii"))
        for block in audio.blocks(blocksize=HASH_BLOCK_FRAMES, dtype=dtype, always_2d=True):
            digest.update(block.tobytes())
    return digest.hexdigest()


def _cached_audio_sha256(wav):
    """The WAV's decoded-audio hash, kept beside it and reused while the file and the hash function are unchanged."""
    key = f"{audio_io.file_key(wav)}|{HASH_NAME}"
    sidecar = wav.with_name(wav.stem + HASH_SIDECAR_SUFFIX)
    cached = guards_mod.read_json(sidecar)
    if cached.get("key") == key and isinstance(cached.get("sha256"), str):
        return cached["sha256"]
    digest = exact_audio_sha256(wav)
    sidecar.write_text(json.dumps({"key": key, "sha256": digest}), encoding="utf-8")
    return digest


def audio_fingerprint(out_dir, cid, tapes):
    """One decoded-audio hash per tape, in tape order: what the candidate sounds like, whatever its settings."""
    return tuple(_cached_audio_sha256(guards_mod.candidate_wav(out_dir, cid, slug)) for slug in tapes)


def find_inert(fingerprints, incumbent_id):
    """`{cid: same_as}` for every candidate whose audio repeats the incumbent's, or an earlier candidate's, on every tape.

    `fingerprints` maps each candidate id (the incumbent's among them) to its `audio_fingerprint`,
    in proposal order. The incumbent is never inert; of candidates that sound alike, the first
    proposed is scored and the others name it.
    """
    owners = {fingerprints[incumbent_id]: incumbent_id}
    inert = {}
    for cid, fingerprint in fingerprints.items():
        owner = owners.setdefault(fingerprint, cid)
        if owner != cid:
            inert[cid] = owner
    return inert


def _aside_verdict(**why):
    """The verdict of a candidate set aside before scoring: no rank, never qualifies; `why` says what set it aside."""
    return {"mean_rank": None, "wins": 0, "failures": 0.0, "flags": 0.0, "per_tape": {}, "qualifies": False, **why}


def _inert_verdict(same_as):
    return _aside_verdict(inert=same_as)


def _without(candidates, left_out):
    """`candidates` less the ids in `left_out`."""
    return {cid: overrides for cid, overrides in candidates.items() if cid not in left_out}


def _split_inert(everything, incumbent_id, tapes, out_dir):
    """`(live, inert)`: the candidates worth scoring and `{cid: same_as}` for the ones that sound like another."""
    inert = find_inert({cid: audio_fingerprint(out_dir, cid, tapes) for cid in everything}, incumbent_id)
    return _without(everything, inert), inert


def _with_set_aside(verdicts, inert, ties, refused):
    """The judged verdicts plus one for every candidate set aside before scoring: inert, a tie or refused."""
    verdicts.update({cid: _inert_verdict(same_as) for cid, same_as in inert.items()})
    verdicts.update({cid: _aside_verdict(tie=True) for cid in ties})
    verdicts.update({cid: _aside_verdict(refused=why) for cid, why in refused.items()})
    return verdicts


# ----------------------------------------------------------------------------- scoring


def score_round(candidates, tapes, out_dir, families, gates, language, parallel=SCORER_PARALLEL):
    """Scores every candidate on every tape, at most `parallel` tape processes at once; returns `{cid: {slug: flat}}`.

    A scorer holds about 4 GB of RAM with every model resident; two engines tuning at
    once with four scorers each ran a 62 GB machine out of memory, so the default is two.
    """
    (out_dir / "rounds").mkdir(parents=True, exist_ok=True)
    pending = _unscored(candidates, tapes, out_dir)
    step = max(1, parallel)
    for start in range(0, len(pending), step):
        stop = start + step
        _score_batch(pending[start:stop], out_dir, (families, gates, language))
    return _load_scores(candidates, tapes, out_dir)


def _score_batch(batch, out_dir, scorer_options):
    """One scorer per `(slug, tape, cids)` of the batch, all launched before the first is waited for."""
    procs = [(slug, cids, _launch_scorer(slug, tape, cids, out_dir, *scorer_options)) for slug, tape, cids in batch]
    for slug, cids, (log, proc) in procs:
        _harvest(slug, cids, log, proc, out_dir)


def _load_scores(candidates, tapes, out_dir):
    """`{cid: {slug: flat}}` read back from the score files."""
    return {cid: {slug: json.loads(_score_path(out_dir, cid, slug).read_text(encoding="utf-8")) for slug in tapes} for cid in candidates}


def _unscored_ids(candidates, out_dir, slug):
    return [cid for cid in candidates if not _score_path(out_dir, cid, slug).exists()]


def _unscored(candidates, tapes, out_dir):
    """`[(slug, tape, cids)]` for every tape that still has candidates without a score on it."""
    pending = [(slug, tape, _unscored_ids(candidates, out_dir, slug)) for slug, tape in tapes.items()]
    return [row for row in pending if row[2]]


def _harvest(slug, cids, log, proc, out_dir):
    proc.wait()
    log.close()
    if proc.returncode != 0:
        raise SystemExit(f"scoring {slug} failed (see {out_dir / 'rounds' / f'score_{slug}.log'})")
    report = json.loads((out_dir / "rounds" / f"score_{slug}.json").read_text(encoding="utf-8"))
    for cid in cids:
        flat = tune.runner.aggregates_for_tuning(report, cid)
        _score_path(out_dir, cid, slug).write_text(json.dumps(flat, indent=1), encoding="utf-8")
        _drop_cache_entries(out_dir, cid, slug)


def _drop_cache_entries(out_dir, cid, slug):
    """The scorer's cache entries for a scored output (its resampled arrays, its DC-free copy), which nothing reads again.

    A candidate output is scored once; its entries (three rates, ~1 GB per hour of tape) only
    fill the disk afterwards: two loops on full tapes grew 250 GB of them in three days and
    ran the machine out of space. The source side of the cache stays, every round reuses it.
    """
    wav = out_dir / "cands" / cid / f"{slug}.wav"
    if not wav.exists():
        return
    for entry in (out_dir / "cache").glob(f"{audio_io.file_key(wav)}_*"):
        entry.unlink()


def _score_path(out_dir, cid, slug):
    return out_dir / "cands" / cid / f"{slug}.score.json"


LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


def _existing_path(value, what):
    """A command-line path resolved and checked to exist; a value shaped like an option is refused."""
    text = str(value)
    if text.startswith("-"):
        raise SystemExit(f"{what} looks like an option, not a path: {text}")
    resolved = Path(text).resolve()
    if not resolved.exists():
        raise SystemExit(f"{what} does not exist: {resolved}")
    return resolved


def _scorer_arguments(slug, tape, cids, out_dir, families, gates, language):
    """The scorer's argument list from checked inputs: the tape and the gates must exist, the language is a code."""
    if not LANGUAGE_RE.match(language):
        raise SystemExit(f"--language must be a two- or three-letter code: {language!r}")
    unknown = set(families) - set(tune.runner.ALL_FAMILIES)
    if unknown:
        raise SystemExit(f"unknown metric families: {sorted(unknown)}")
    out_dir = _existing_path(out_dir, "--out")
    args = [PYTHON, str(REPO / "scripts" / "validate_restoration.py"), str(_existing_path(tape, "tape"))]
    args += [f"{cid}={out_dir / 'cands' / cid / f'{slug}.wav'}" for cid in cids]
    args += ["--metrics", ",".join(families), "--language", language, "--cache-dir", str(out_dir / "cache")]
    args += ["--report", str(out_dir / "rounds" / f"score_{slug}.json")]
    if gates:
        args += ["--gates", str(_existing_path(gates, "--gates"))]
    return args


def _scorer_env(tape):
    """The scorer's environment: the tape's folder joins AI_RESTORE_DATA_ROOTS, since the scorer confines its paths."""
    roots = [entry for entry in os.environ.get("AI_RESTORE_DATA_ROOTS", "").split(os.pathsep) if entry]
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "AI_RESTORE_DATA_ROOTS": os.pathsep.join([*roots, str(Path(tape).resolve().parent)])}


def _launch_scorer(slug, tape, cids, out_dir, families, gates, language):
    args = _scorer_arguments(slug, tape, cids, out_dir, families, gates, language)
    log = open(out_dir / "rounds" / f"score_{slug}.log", "a", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=_scorer_env(tape))
    return log, proc


# ----------------------------------------------------------------------------- judging


def rank_on_tape(flats, ranking):
    """`{cid: rank_score}` on one tape from flat aggregates, with the grid's directions."""
    summaries = {cid: {"metrics": {key: {"median": value} for key, value in flat.items()}} for cid, flat in flats.items()}
    return tune.rank_variants(summaries, ranking)


def _ranks_per_tape(scores, ranking):
    """`{slug: {cid: rank_score}}` over every tape the candidates were scored on."""
    tapes = list(next(iter(scores.values())))
    return {slug: rank_on_tape({cid: scores[cid][slug] for cid in scores}, ranking) for slug in tapes}


def judge(scores, incumbent_id, ranking, vetoes=None):
    """`{cid: {"mean_rank", "wins", "failures", "per_tape", "qualifies"}}` against the incumbent across tapes.

    With `vetoes` (`{reading: reward.Veto}`, a v3 grid's learned judges) a candidate whose
    median moved past a veto on some tape also carries `vetoed` and cannot qualify.
    """
    per_tape = _ranks_per_tape(scores, ranking)
    verdicts = {cid: _verdict(cid, scores[cid], per_tape, incumbent_id) for cid in scores}
    base = verdicts[incumbent_id]
    for cid, verdict in verdicts.items():
        guards_mod.mark_vetoes(verdict, scores[cid], scores[incumbent_id], vetoes)
        verdict["qualifies"] = cid != incumbent_id and _beats(verdict, base, len(per_tape)) and guards_mod.allowed(verdict, base)
    return verdicts


def _total(tape_scores, key):
    return sum(flat.get(key, 0.0) for flat in tape_scores.values())


def _wins(ranks, per_tape, incumbent_id):
    return sum(1 for slug, rank in ranks.items() if rank is not None and rank < per_tape[slug][incumbent_id])


def _verdict(cid, tape_scores, per_tape, incumbent_id):
    ranks = {slug: per_tape[slug][cid] for slug in tape_scores}
    known = [r for r in ranks.values() if r is not None]
    return {
        "mean_rank": float(np.mean(known)) if known else None,
        "wins": _wins(ranks, per_tape, incumbent_id),
        "failures": _total(tape_scores, "gates.hard_failures"),
        "flags": _total(tape_scores, "gates.listener_flags"),
        "per_tape": ranks,
    }


def _no_worse(verdict, base):
    return all(verdict.get(key, 0.0) <= base.get(key, 0.0) for key in ("failures", "flags"))


def _beats(verdict, base, n_tapes):
    """Better mean rank, ahead on at least half the tapes, no more hard failures and no more listener flags."""
    if None in (verdict["mean_rank"], base["mean_rank"]):
        return False
    better = verdict["mean_rank"] < base["mean_rank"] and verdict["wins"] * 2 >= n_tapes
    return better and _no_worse(verdict, base)


def describe(overrides, incumbent):
    """The overrides as JSON, only the keys that differ from the incumbent's when any do."""
    changed = {k: v for k, v in overrides.items() if incumbent.get(k) != v}
    if not overrides:
        return "defaults"
    return json.dumps(changed if changed else overrides, default=str)


# ----------------------------------------------------------------------------- driver


def load_state(out_dir, start):
    """The run's saved state, or a fresh one starting from `start`."""
    path = out_dir / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"incumbent": start, "rounds": [], "candidates": {}}


def save_state(out_dir, state):
    """Writes the run's state, so a stopped run resumes where it was."""
    (out_dir / "state.json").write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")


def run_round(args, engine, tapes, out_dir, state, ranking, number):
    """One round: neighbours, runs, scores, the combination of the winners, the verdict; True when the incumbent moved."""
    incumbent = state["incumbent"]
    proposals = propose(engine, incumbent, getattr(args, "durations", None))
    if not proposals:
        return False
    everything = {candidate_id(incumbent): incumbent, **proposals}
    print(f"round {number}: incumbent {candidate_id(incumbent)} {describe(incumbent, {})}; {len(proposals)} candidates", flush=True)
    verdicts = _run_and_judge(args, engine, everything, tapes, out_dir, state, ranking, incumbent)
    combo = combine(incumbent, [everything[cid] for cid, v in verdicts.items() if v["qualifies"]])
    if combo:
        everything.update(combo)
        verdicts = _run_and_judge(args, engine, everything, tapes, out_dir, state, ranking, incumbent)
    accepted = _settle(state, number, incumbent, everything, verdicts)
    _report_round(out_dir, state["rounds"][-1], incumbent, everything, len(tapes))
    save_state(out_dir, state)
    return accepted is not None


def _ordered(verdicts):
    """`[(cid, verdict)]` best mean rank first; candidates without a rank (inert ones included) last."""
    return sorted(verdicts.items(), key=lambda kv: (kv[1]["mean_rank"] is None, kv[1]["mean_rank"] or 0.0))


def _settle(state, number, incumbent, everything, verdicts):
    """Records the round and moves the incumbent to the best qualifying candidate; returns its id or None."""
    accepted = next((cid for cid, v in _ordered(verdicts) if v["qualifies"]), None)
    state["rounds"].append({"round": number, "incumbent": candidate_id(incumbent), "verdicts": verdicts, "accepted": accepted})
    if accepted is not None:
        state["incumbent"] = everything[accepted]
    return accepted


def _render(engine, candidates, tapes, out_dir, incumbent):
    for cid, overrides in candidates.items():
        print(f"  running {cid} {describe(overrides, incumbent)}", flush=True)
        run_candidate(engine, overrides, tapes, out_dir)


def _run_and_judge(args, engine, everything, tapes, out_dir, state, ranking, incumbent):
    """Renders every candidate no ledger boundary refuses, sets the inert and inaudible ones aside unscored, judges the rest.

    `args.guards` holds the grid's v3 guards; without it (a v1 or v2 grid) nothing is refused,
    tied or vetoed.
    """
    guards = getattr(args, "guards", guards_mod.NO_GUARDS)
    state["candidates"].update(everything)
    save_state(out_dir, state)
    refused = guards_mod.refusals(everything, incumbent, guards.reversals)
    rendered = _without(everything, refused)
    _render(engine, rendered, tapes, out_dir, incumbent)
    incumbent_id = candidate_id(incumbent)
    live, inert = _split_inert(rendered, incumbent_id, tapes, out_dir)
    ties = guards_mod.find_ties(live, incumbent_id, tapes, out_dir, guards.audibility_offset_db)
    scores = score_round(_without(live, ties), tapes, out_dir, args.families, args.gates, args.language, args.parallel)
    verdicts = judge(scores, incumbent_id, ranking, guards.vetoes)
    return _with_set_aside(verdicts, inert, ties, refused)


def _cells(verdict, n_tapes):
    """The rank, wins, failures, flags and verdict cells of one candidate's row."""
    aside = guards_mod.aside_cell(verdict)
    if aside:
        return f"- | - | - | - | {aside}"
    rank = "-" if verdict["mean_rank"] is None else f"{verdict['mean_rank']:.2f}"
    qualifies = guards_mod.qualifies_cell(verdict)
    return f"{rank} | {verdict['wins']}/{n_tapes} | {verdict['failures']:.0f} | {verdict.get('flags', 0.0):.0f} | {qualifies}"


def _outcome(accepted, everything, incumbent):
    if accepted is None:
        return "no candidate qualified: plateau"
    return f"accepted `{accepted}` -> {describe(everything[accepted], incumbent)}"


def _inert_findings(ordered, everything, incumbent):
    """One knob-table finding per inert candidate: its move rendered audio another candidate already had."""
    inert = [(cid, v["inert"]) for cid, v in ordered if v.get("inert")]
    return [
        f"knob-table finding: `{cid}` {describe(everything[cid], incumbent)} sounds the same as `{same_as}` on every tape; not scored"
        for cid, same_as in inert
    ]


def _report_round(out_dir, record, incumbent, everything, n_tapes):
    """Appends the round's table, its outcome, its knob-table findings and its guards' findings to log.md and prints them."""
    ordered = _ordered(record["verdicts"])
    lines = [f"## round {record['round']}", "", f"incumbent `{candidate_id(incumbent)}` {describe(incumbent, {})}", ""]
    lines += ["| candidate | change | mean rank | wins | hard failures | flags | qualifies |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {cid} | {describe(everything[cid], incumbent)} | {_cells(v, n_tapes)} |" for cid, v in ordered]
    moved = guards_mod.guard_findings(ordered, lambda cid: describe(everything[cid], incumbent))
    findings = [*_inert_findings(ordered, everything, incumbent), *moved]
    lines += ["", _outcome(record["accepted"], everything, incumbent), *findings, ""]
    with open(out_dir / LOG_MD, "a", encoding="utf-8") as log:
        log.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=list(KNOBS), required=True)
    parser.add_argument("--tapes", type=Path, required=True, help="JSON {slug: video path}")
    parser.add_argument("--out", type=Path, default=Path("experiments/autotune"))
    parser.add_argument("--grid", type=Path, default=Path("scripts/tune_grids/tata_v2.yaml"))
    # The loop stops at a plateau (user: "stop only when no more improvements are possible"); the cap is a safety net.
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--families", default="dsp,stems,speech,mos")
    parser.add_argument("--start", default="{}", help="JSON overrides to start from")
    parser.add_argument("--start-file", type=Path, default=None, help="a final.json of an earlier run to start from (wins over --start)")
    parser.add_argument("--language", default="ro")
    parser.add_argument("--parallel", type=int, default=SCORER_PARALLEL, help="tape scorers running at once (about 4 GB RAM each)")
    parser.add_argument("--gates", type=Path, default=tune.DEFAULT_GATES if tune.DEFAULT_GATES.exists() else None)
    parser.add_argument(
        "--noise-floors",
        type=Path,
        default=guards_mod.DEFAULT_NOISE_FLOORS if guards_mod.DEFAULT_NOISE_FLOORS.exists() else None,
        help="a scripts/reward_noise_floor.py report: the benign floors a v3 grid's learned vetoes are measured in",
    )
    parser.add_argument(
        "--stage-cache", type=Path, default=None, help="a folder for the neural-stage cache (AI_RESTORE_STAGE_CACHE); off by default"
    )
    parser.add_argument("--stage-cache-gb", type=float, default=None, help="the stage cache's cap in GB (default 50)")
    args = parser.parse_args(argv)
    args.families = tuple(args.families.split(","))
    return args


def _noise_floors(path):
    """`{reading: floor}` from a `scripts/reward_noise_floor.py` report, or {} without one."""
    return guards_mod.noise_floors(_existing_path(path, "--noise-floors") if path else None)


def _resume(args, out_dir):
    """The saved state, or a fresh one from the start settings; the incumbent seeded with every knob's real value."""
    start = json.loads(args.start_file.read_text(encoding="utf-8")) if args.start_file else json.loads(args.start)
    state = load_state(out_dir, seed_defaults(args.engine, start, out_dir))
    # A knob added to the table after the run began is seeded too, so its first moves are neighbours of the real value.
    state["incumbent"] = seed_defaults(args.engine, state["incumbent"], out_dir)
    return state


def main(argv=None):
    """Runs rounds until a plateau or the round cap, then writes the incumbent to `final.json`."""
    args = _parse_args(argv)
    enable_stage_cache(args.stage_cache, args.stage_cache_gb)
    tapes = {slug: str(Path(p)) for slug, p in json.loads(args.tapes.read_text(encoding="utf-8")).items()}
    out_dir = args.out / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = tune.load_grid(args.grid)
    ranking = grid["ranking"]
    args.guards = guards_mod.loop_guards(grid, _noise_floors(args.noise_floors), ALL_KNOBS)
    print(f"guards: {guards_mod.describe_guards(args.guards)}", flush=True)
    args.durations = tape_durations(tapes)
    print(f"tape lengths (s): {json.dumps(args.durations)}", flush=True)
    state = _resume(args, out_dir)
    if not (out_dir / LOG_MD).exists():
        (out_dir / LOG_MD).write_text(f"# autotune {args.engine}\n\ntapes: {', '.join(tapes)}\n\n", encoding="utf-8")
    number = len(state["rounds"]) + 1
    while number <= args.rounds and run_round(args, args.engine, tapes, out_dir, state, ranking, number):
        number += 1
    final = state["incumbent"]
    print(f"final {args.engine}: {candidate_id(final)} {describe(final, {})}", flush=True)
    (out_dir / "final.json").write_text(json.dumps(final, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
