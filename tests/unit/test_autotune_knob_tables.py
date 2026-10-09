"""The tuner's knob tables (ear v3, plan 1.5): the live knobs, and the moves the switches or the tapes' lengths make dead."""

from modules import config as app_config
from scripts import autotune_restoration as at


def _knobs(engine, incumbent, durations=None):
    return {knob for knob, _ in at.neighbour_moves(engine, incumbent, durations)}


def test_the_flatness_threshold_stays_a_knob_with_the_subtraction_stage_off():
    """The plosive tamer and the hum canceller read apl_tonal_flatness_max whatever the subtraction switch says."""
    assert "apl_tonal_flatness_max" in _knobs("apl", {"apl_enable_spectral_denoise": False})
    assert "apl_tonal_flatness_max" not in at.SUBTRACTION_KNOBS


def test_the_music_persistence_stays_a_knob_with_the_stem_path_off():
    """It also picks the tapes that take apl_music_neural_model, so only the stem path's background floor is inert."""
    inert = at.inert_knobs("apl", {"apl_music_stem_path": False})
    assert "apl_music_persistence_min" not in inert
    assert "apl_music_bg_floor_db" in inert


def test_the_cathar_binary_is_no_longer_a_knob():
    """[None, 0.7.6] rendered the installed 0.7.6 binary twice."""
    assert not [knob for knob in at.KNOBS["cathar"] if knob.startswith(at.ENV_PREFIX)]


def test_the_music_profile_settings_are_knobs():
    """The music rounds could not move the profile's de-esser, expander depth, notch width or APL's music model."""
    cathar = at.KNOBS["cathar"]
    assert cathar["cathar_music_enable_deesser"] == [False, True]
    assert cathar["cathar_music_expander_depth_db"] == [4.0, 7.0, 12.0]
    assert cathar["cathar_music_crt_notch_q"] == [30.0, 60.0, 120.0]
    assert at.KNOBS["apl"]["apl_music_neural_model"] == [None, at.ROFORMER]


def _every_override():
    return [{knob: value} for table in at.KNOBS.values() for knob, values in table.items() for value in values if value is not None]


def test_every_knob_value_is_a_setting_the_app_accepts():
    """A value the app refuses stops a round half-way (run_candidate raises), so the whole table is checked."""
    assert [override for override in _every_override() if at.validation_problems(override)] == []


def test_an_unset_switch_reads_as_the_value_the_app_resolves():
    """SWITCH_DEFAULTS is what a candidate without the key runs with: config.yaml over modules/config.py."""
    assert {switch: app_config.CONFIG.get(switch) for switch in at.SWITCH_DEFAULTS} == at.SWITCH_DEFAULTS


def test_every_switch_of_the_inert_table_has_a_default():
    """inert_knobs looks every switch up in SWITCH_DEFAULTS when the incumbent does not name it."""
    switches = {switch for entries in at.INERT_WHEN.values() for when, _ in entries for switch in when}
    assert switches == set(at.SWITCH_DEFAULTS)


def test_at_the_shipped_defaults_only_the_dead_stages_are_left_out():
    """cathar loses only the split-band factors; APL the subtraction stage, the stem path's floor and the shared depth."""
    assert at.inert_knobs("cathar", {}) == {"cathar_alpha_high", "cathar_music_alpha_high"}
    assert at.inert_knobs("apl", {}) == {*at.SUBTRACTION_KNOBS, "apl_noiseprint_tonal_s", "apl_music_bg_floor_db", "expander_depth_db"}


def test_the_deesser_threshold_is_inert_only_when_no_material_runs_the_deesser():
    """Speech runs cathar_enable_deesser and music cathar_music_enable_deesser."""
    threshold = "cathar_deesser_threshold"
    assert threshold in at.inert_knobs("cathar", {"cathar_enable_deesser": False})
    assert threshold not in at.inert_knobs("cathar", {"cathar_enable_deesser": False, "cathar_music_enable_deesser": True})
    assert threshold in _knobs("cathar", {})


def test_the_air_gain_is_inert_with_the_shelf_off():
    """filters._build_linear_air_filter builds no shelf with the air off, whatever the gain."""
    assert "linear_air_gain_db" in at.inert_knobs("apl", {"enable_linear_air": False})
    assert "linear_air_gain_db" in _knobs("apl", {})


def test_the_wiener_denoiser_freezes_every_subtraction_factor():
    """`denoise --wiener` takes no alpha or beta, and the split band only runs under spectral subtraction."""
    split = {"cathar_split_band_hz": 4000}
    factors = {"cathar_alpha_high", "cathar_alpha", "cathar_beta"}
    assert factors | {"cathar_split_band_hz"} <= at.inert_knobs("cathar", {**split, "cathar_denoise_method": "wiener"})
    assert not at.inert_knobs("cathar", split) & factors


def test_the_print_length_is_inert_when_no_material_learns_a_print():
    """Speech learns a print under cathar_enable_noiseprint, music under cathar_music_enable_noiseprint."""
    length = "cathar_noiseprint_duration_s"
    assert length in at.inert_knobs("cathar", {"cathar_enable_noiseprint": False})
    assert length not in at.inert_knobs("cathar", {"cathar_enable_noiseprint": False, "cathar_music_enable_noiseprint": True})


def test_the_suppressor_floor_is_inert_without_the_native_suppressor():
    """Only spectral_denoise._native_suppress reads apl_suppress_gain_floor_db."""
    stage_on = {"apl_enable_spectral_denoise": True}
    assert "apl_suppress_gain_floor_db" in at.inert_knobs("apl", stage_on)
    assert "apl_suppress_gain_floor_db" not in at.inert_knobs("apl", {**stage_on, "apl_use_native_suppress": True})


def test_the_music_keys_are_inert_with_the_profile_or_the_expander_off():
    """No tape takes the music profile when it is off; the music expander depth needs the expander."""
    music = {"cathar_music_enable_deesser", "cathar_music_crt_notch_q", "cathar_music_persistence_min"}
    assert music <= at.inert_knobs("cathar", {"cathar_music_profile": False})
    assert "cathar_music_expander_depth_db" in at.inert_knobs("cathar", {"enable_dynamic_expander": False})


def test_the_tonal_probe_stays_a_knob_once_the_stem_path_runs_its_background_suppressor():
    """apl_stems._background_pass reads apl_noiseprint_tonal_s as its probe while the floor is below 0."""
    probe = "apl_noiseprint_tonal_s"
    assert probe not in at.SUBTRACTION_KNOBS
    assert probe not in at.inert_knobs("apl", {"apl_music_stem_path": True})
    assert probe in at.inert_knobs("apl", {"apl_music_stem_path": True, "apl_music_bg_floor_db": 0.0})
    assert probe in _knobs("apl", {"apl_enable_spectral_denoise": True})


# ----------------------------------------------------------------------------- the tapes' lengths (design 5.1)

PRINT = "cathar_noiseprint_duration_s"


def test_the_print_length_follows_cathars_rule_with_a_margin_around_the_switch():
    """4.5 s applies from 90 s of tape (20 times), a shorter tape learns from 0.75 s; near 90 s or unknown cannot tell."""
    assert at.cathar_print_s(4.5, 200.0) == 4.5
    assert at.cathar_print_s(4.5, 60.0) == at.cathar.NOISEPRINT_SHORT_S == 0.75
    assert at.cathar_print_s(4.5, 90.4) is None
    assert at.cathar_print_s(4.5, None) is None


def test_the_print_length_rule_reads_the_ratio_from_cathar(monkeypatch):
    """The ratio is cathar's own constant, not a copy: at 10 times, 4.5 s applies to a 60 s tape."""
    monkeypatch.setattr(at.cathar, "NOISEPRINT_MIN_MATERIAL_RATIO", 10.0)
    assert at.cathar_print_s(4.5, 60.0) == 4.5


def test_the_print_length_is_dead_when_every_tape_is_too_short_for_any_of_its_values():
    """60 s and 80 s are under 20 x 4.5 s and 20 x 6 s: both values learn from 0.75 s on both tapes."""
    incumbent, short = {PRINT: 4.5}, {"a": 60.0, "b": 80.0}
    assert PRINT in at.inert_knobs("cathar", incumbent, short)
    assert PRINT not in _knobs("cathar", incumbent, short)
    assert not [cid for cid, overrides in at.propose("cathar", incumbent, short).items() if overrides[PRINT] != 4.5]


def test_one_tape_long_enough_for_one_value_keeps_the_print_length_live():
    """On 100 s 4.5 s applies and 6 s falls back to 0.75 s, so the move changes that tape."""
    incumbent = {PRINT: 4.5}
    assert at.dead_move(PRINT, 6.0, incumbent, {"a": 60.0}) is True
    assert at.dead_move(PRINT, 6.0, incumbent, {"a": 60.0, "b": 100.0}) is False
    assert PRINT in _knobs("cathar", incumbent, {"a": 60.0, "b": 100.0})


UNDECIDED_LENGTHS = [
    ({PRINT: 4.5}, {"a": 60.0, "b": 90.4}),
    ({PRINT: 4.5}, {"a": 60.0, "b": None}),
    ({PRINT: 4.5}, None),
    ({}, {"a": 60.0}),
]


def test_a_move_the_lengths_cannot_decide_stays_live():
    """A tape near the switch, a tape ffprobe could not read, no lengths at all and an unseeded incumbent keep the move."""
    assert [PRINT in _knobs("cathar", incumbent, lengths) for incumbent, lengths in UNDECIDED_LENGTHS] == [True] * 4
    assert not at.dead_move("cathar_alpha", 2.0, {"cathar_alpha": 1.5}, {"a": 60.0})


def test_the_tape_lengths_are_read_by_ffprobe_per_tape(monkeypatch):
    """None is what tune.probe_duration returns for a tape it cannot read."""
    lengths = {"a.mov": 61.0, "b.mov": None}
    monkeypatch.setattr(at.tune, "probe_duration", lambda path: lengths[path])
    assert at.tape_durations({"a": "a.mov", "b": "b.mov"}) == {"a": 61.0, "b": None}


# ----------------------------------------------------------------------------- one setting, one value


def test_the_neural_model_list_offers_the_aggressive_model_and_never_the_default_twice(tmp_path, monkeypatch):
    """From the shipped defaults the incumbent is seeded with the app's model, and its one move is the aggressive one."""
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: dict(app_config.CONFIG))
    incumbent = at.seed_defaults("apl", {}, tmp_path)
    assert incumbent["apl_neural_model"] == app_config.DEFAULT_APL_NEURAL_MODEL == at.ROFORMER
    moves = {value for knob, value in at.neighbour_moves("apl", incumbent) if knob == "apl_neural_model"}
    assert moves == {at.ROFORMER_AGGR}


def test_a_saved_none_for_a_knob_without_a_default_entry_is_seeded(tmp_path, monkeypatch):
    """A state saved under the old [None, ROFORMER, ROFORMER_AGGR] list holds None; it is read as the app's value."""
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: {"apl_neural_model": at.ROFORMER})
    assert "apl_neural_model" in at._unseeded("apl", {"apl_neural_model": None})
    assert at.seed_defaults("apl", {"apl_neural_model": None}, tmp_path)["apl_neural_model"] == at.ROFORMER


# ----------------------------------------------------------------------------- rounds A1 (brightness) and A3 (pause texture)

TIMING = ("expander_attack_s", "expander_decay_s")


def test_the_brightness_and_pause_texture_knobs_hold_the_rounds_values():
    """A1 moves APL's shelf gain (plan: 0 / 0.5 / 1.0 / 1.5 dB, 0 the shelf off) and corner; A3 the shared expander timing."""
    apl = at.KNOBS["apl"]
    assert (apl["linear_air_gain_db"], apl["linear_air_freq_hz"]) == ([0.5, 1.0, 1.5], [6000.0, 7500.0, 9000.0])
    for engine in ("apl", "cathar"):
        assert [at.KNOBS[engine][knob] for knob in TIMING] == [[0.02, 0.04, 0.08], [0.12, 0.18, 0.3]]
    assert "linear_air_freq_hz" not in at.KNOBS["cathar"]


def test_the_air_corner_is_inert_with_the_shelf_off():
    """filters._build_linear_air_filter builds no shelf with the air off, whatever the corner."""
    assert "linear_air_freq_hz" in at.inert_knobs("apl", {"enable_linear_air": False})
    assert "linear_air_freq_hz" in _knobs("apl", {})


def test_the_expander_timing_is_inert_with_the_expander_off_in_both_engines():
    """filters._append_expander_stage builds no compand with the expander off; on, both engines move its timing."""
    for engine in ("apl", "cathar"):
        assert set(TIMING) <= at.inert_knobs(engine, {"enable_dynamic_expander": False})
        assert set(TIMING) <= _knobs(engine, {})


NEW_KNOBS = ("linear_air_gain_db", "linear_air_freq_hz", *TIMING)


def _moves_by_knob(incumbent):
    """`{knob: {value, ...}}` of the neighbour moves APL proposes for the knobs rounds A1 and A3 added or widened."""
    moves = {knob: set() for knob in NEW_KNOBS}
    for knob, value in at.neighbour_moves("apl", incumbent):
        moves.get(knob, set()).add(value)
    return moves


def test_from_the_shipped_defaults_each_new_knob_steps_to_its_neighbours(tmp_path, monkeypatch):
    """Seeded from the app (+1 dB at 7500 Hz, 0.04 / 0.18 s), each knob proposes the values beside the shipped one."""
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: dict(app_config.CONFIG))
    incumbent = at.seed_defaults("apl", {}, tmp_path)
    assert [incumbent[knob] for knob in NEW_KNOBS] == [1.0, 7500.0, 0.04, 0.18]
    assert _moves_by_knob(incumbent) == {
        "linear_air_gain_db": {0.5, 1.5},
        "linear_air_freq_hz": {6000.0, 9000.0},
        "expander_attack_s": {0.02, 0.08},
        "expander_decay_s": {0.12, 0.3},
    }


def test_no_grid_lets_the_loop_step_to_the_shelf_round_three_rejected():
    """+2 dB is not in the table, so a v1 / v2 grid (no reversal) cannot reach it either: +1.5 dB steps only back to +1."""
    assert 2.0 not in at.KNOBS["apl"]["linear_air_gain_db"]
    gains = [value for knob, value in at.neighbour_moves("apl", {"linear_air_gain_db": 1.5}) if knob == "linear_air_gain_db"]
    assert gains == [1.0]
