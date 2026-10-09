"""The stage cache's post-neural allowlist, frozen against the code: every read of an allowlisted key is after the cached point.

A key on `POST_NEURAL_CONFIG_KEYS` is left out of the cache key, so a read of it inside the
cached stages would let a changed setting replay a stale render. These tests scan every module
for each key's constant (as a name, an attribute, a `_get_config_val("X")` string) and its raw
key string, and check every read against the read-site table below: a new read anywhere else
fails here first. `test_stage_cache_reach.py` checks the other half: nothing the cached stages
can reach reads one of these keys.
"""

import ast
import re
from pathlib import Path

from modules import stage_cache_key

MODULES = Path(stage_cache_key.__file__).resolve().parent
SKIPPED = {"config.py", "stage_cache.py", "stage_cache_key.py"}
WHOLE = "*"

# key -> (constant in modules/config.py, the sites allowed to read it: "<module>:<qualified function>" or "<module>:*").
READ_SITES = {
    "enable_linear_air": ("ENABLE_LINEAR_AIR", {"filters.py:_build_linear_air_filter"}),
    "linear_air_gain_db": ("LINEAR_AIR_GAIN_DB", {"filters.py:_build_linear_air_filter"}),
    # The polish expander; processing's background expander runs only in hybrid and auto_pure.
    "enable_dynamic_expander": ("ENABLE_DYNAMIC_EXPANDER", {"filters.py:_append_expander_stage", "processing.py:_expand_background_step"}),
    # The raw string is also the name of the post-neural stage setting the step takes (POST_NEURAL_STAGES).
    "expander_depth_db": (
        "EXPANDER_DEPTH_DB",
        {
            "filters.py:_build_full_audio_expander_filter",
            "processing.py:<module>",
            "processing.py:_denoise_and_polish_full_audio_step",
        },
    ),
    "expander_knee_offset_db": ("EXPANDER_KNEE_OFFSET_DB", {"filters.py:_build_full_audio_expander_filter"}),
    "apl_expander_depth_db": ("APL_EXPANDER_DEPTH_DB", {"modes/auto_pure_linear.py:AutoPureLinearMode.execute.denoise_step"}),
    "apl_enable_sibilant_guard": ("APL_ENABLE_SIBILANT_GUARD", {"sibilant_guard.py:*"}),
    "apl_sibilant_mix": ("APL_SIBILANT_MIX", {"sibilant_guard.py:*"}),
    "apl_sibilant_guard_hz": ("APL_SIBILANT_GUARD_HZ", {"sibilant_guard.py:*"}),
    "apl_sibilant_hf_share_min": ("APL_SIBILANT_HF_SHARE_MIN", {"sibilant_guard.py:*"}),
    "enable_pause_floor": ("ENABLE_PAUSE_FLOOR", {"pause_floor.py:*"}),
    "pause_floor_fill_db": ("PAUSE_FLOOR_FILL_DB", {"pause_floor.py:*"}),
    "pause_floor_quiet_percentile": ("PAUSE_FLOOR_QUIET_PERCENTILE", {"pause_floor.py:*"}),
    # processing's <module> reads are the re-exports at the end of the file.
    "enable_loudnorm": ("ENABLE_LOUDNORM", {"mastering.py:*", "processing.py:<module>"}),
    "loudnorm_target_lra": ("LOUDNORM_TARGET_LRA", {"mastering.py:*", "processing.py:<module>"}),
    "loudnorm_linear_fallback": ("LOUDNORM_LINEAR_FALLBACK", {"mastering.py:*"}),
    "preserve_original_audio_track": (
        "PRESERVE_ORIGINAL_AUDIO_TRACK",
        {"processing.py:_final_mix_output_command", "processing.py:_build_single_audio_mux_command"},
    ),
    "dtw_resolution": ("DTW_RESOLUTION", {"sync.py:*"}),
    "apl_music_bg_floor_db": ("APL_MUSIC_BG_FLOOR_DB", {"apl_stems.py:_background_pass"}),
}

POST_CACHE_IMPORTERS = {"processing.py", "modes/cathar.py"}


def _module_files():
    for path in sorted(MODULES.rglob("*.py")):
        name = path.relative_to(MODULES).as_posix()
        if name not in SKIPPED:
            yield name, ast.parse(path.read_text(encoding="utf-8"))


class _Reads(ast.NodeVisitor):
    """Every load of a watched name, attribute or string, with the qualified function it sits in."""

    def __init__(self, module, watched):
        self.module, self.watched, self.stack, self.found = module, watched, [], []

    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def _hit(self, token):
        if token in self.watched:
            self.found.append((self.watched[token], f"{self.module}:{'.'.join(self.stack) or '<module>'}"))

    def visit_Name(self, node):
        self._hit(node.id)

    def visit_Attribute(self, node):
        self._hit(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self._hit(node.value)


def _all_reads():
    watched = {constant: key for key, (constant, _sites) in READ_SITES.items()}
    watched.update({key: key for key in READ_SITES})
    reads = []
    for module, tree in _module_files():
        visitor = _Reads(module, watched)
        visitor.visit(tree)
        reads.extend(visitor.found)
    return reads


def _allowed(site, sites):
    module = site.split(":", 1)[0]
    return site in sites or f"{module}:{WHOLE}" in sites


def test_the_table_is_the_allowlist():
    assert set(READ_SITES) == set(stage_cache_key.POST_NEURAL_CONFIG_KEYS)


def test_each_constant_is_the_one_config_reads_its_key_into():
    text = (MODULES / "config.py").read_text(encoding="utf-8")
    for key, (constant, _sites) in READ_SITES.items():
        assert re.search(rf'^{constant} = .*CONFIG(\.get\(|\[)"{key}"', text, re.M), (key, constant)


def test_every_read_of_an_allowlisted_key_is_at_a_listed_site():
    stray = [(key, site) for key, site in _all_reads() if not _allowed(site, READ_SITES[key][1])]
    assert not stray


def _sites_reading(found, key, module):
    return {site for read_key, site in found if read_key == key and site.startswith(f"{module}:")}


def _still_read(found, key, site):
    """Whether a listed site still reads its key: the function itself, or any part of a whole-module site."""
    module, function = site.split(":", 1)
    hits = _sites_reading(found, key, module)
    return bool(hits) if function == WHOLE else site in hits


def test_every_listed_site_still_reads_its_key():
    """A row whose read moved or went away is stale: the table must follow the code."""
    found = set(_all_reads())
    stale = [(key, site) for key, (_constant, sites) in READ_SITES.items() for site in sites if not _still_read(found, key, site)]
    assert not stale


def _import_names(node):
    """What one import brings in: each name, and for `from X import`, X's last part."""
    names = [alias.name.rsplit(".", 1)[-1] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        names.append(node.module.rsplit(".", 1)[-1])
    return names


def _imported_modules(tree):
    """The sibling module names a module imports, relative imports resolved to bare names."""
    imports = (node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
    return {name for node in imports for name in _import_names(node)}


def test_the_post_cache_modules_are_imported_only_where_they_run_after_the_cached_point():
    post = {Path(name).stem for name in stage_cache_key.POST_CACHE_MODULES}
    importers = {module for module, tree in _module_files() if post & _imported_modules(tree)}
    entry = ast.parse((MODULES.parent / "restore_audio_hybrid.py").read_text(encoding="utf-8"))
    assert (importers <= POST_CACHE_IMPORTERS, post & _imported_modules(entry)) == (True, set())


def _names_assets(node):
    """A string naming the `assets` folder, alone or as one part of a path."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and "assets" in re.split(r"[\\/]", node.value)


def test_the_blend_weights_are_the_only_asset_a_module_reads():
    readers = {module for module, tree in _module_files() if any(_names_assets(node) for node in ast.walk(tree))}
    assert readers == {"blend_weights.py"}
