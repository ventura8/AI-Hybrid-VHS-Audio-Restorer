"""What the cached stages can reach across every module: no post-neural setting, stage or module is among it.

`test_stage_cache_allowlist.py` checks each read of an allowlisted key against a table of read
sites. This file closes the other half: it follows every reference (a call, a function handed
on as a value, a lambda's body, an alias like `_neural_denoise_dir = _denoise_cache.x`, a lazy
import inside a function, a class and every method in it) from `processing._neural_output`
and `processing._run_neural_model` across all of `modules/`, and checks that nothing reached
reads an allowlisted key, is a listed post-neural read site or a post-neural stage, or lives in
a post-cache module. The graph over-approximates (a reference counts as a call), so a pass
holds for the code as it stands, and a new path from the cached stages to a post-neural
setting fails here.
"""

import ast
from pathlib import Path

from modules import processing, stage_cache_key
from tests.unit.test_stage_cache_allowlist import MODULES, READ_SITES, WHOLE, _Reads

ROOTS = (("processing.py", "_neural_output"), ("processing.py", "_run_neural_model"))
# Stages the cached point is known to run: the graph must reach each of them, or it is blind.
MUST_REACH = {
    ("processing.py", "_pre_denoise_surgical_step"),
    ("filters.py", "build_pre_denoise_surgical_filter"),
    ("apl_chain.py", "stage_plan"),
    ("physical_repair.py", "apply_when_needed"),
    ("impulse_repair.py", "depop"),
    ("hum_cancel.py", "apply_when_needed"),
    ("tone_cancel.py", "apply_when_needed"),
    ("plosive_tamer.py", "apply_when_needed"),
    ("spectral_denoise.py", "apply_when_needed"),
    ("cathar.py", "_cathar_noiseprint_step"),
    ("cathar.py", "_cathar_decrackle_step"),
    ("blend_weights.py", "apply_blend"),
    ("processing.py", "_denoise_full_audio_step"),
    ("denoise_chunking.py", "run"),
}
# The stages after the cached point and the mux, by module: none of them may be reached.
POST_NEURAL_FUNCTIONS = {
    ("processing.py", "_denoise_and_polish_full_audio_step"),
    ("processing.py", "_post_neural_stages"),
    ("processing.py", "_polish_full_audio_step"),
    ("processing.py", "_post_denoise_cleanup_step"),
    ("processing.py", "_expand_background_step"),
    ("processing.py", "_final_mux_single_audio_step"),
    ("processing.py", "_final_mix_step"),
    ("processing.py", "_align_and_mix_stems"),
    ("filters.py", "build_full_audio_polish_filter"),
    ("apl_stems.py", "execute"),
    ("apl_stems.py", "_background_pass"),
}
# The names the producer handed to the cache may close over: the step's own arguments, never its post-neural `stages`.
PRODUCER_NAMES = {"_neural_output", "original_wav", "audio_dir", "total_duration", "denoise_model", "strategy", "flags"}
CACHE_MODULES = {"stage_cache", "stage_cache_key"}
MAX_ALIAS_DEPTH = 8
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
MODULE = "<module>"


def _sources():
    """Relative posix name -> parsed tree of every module source."""
    return {path.relative_to(MODULES).as_posix(): ast.parse(path.read_text(encoding="utf-8")) for path in sorted(MODULES.rglob("*.py"))}


def _scopes(node, prefix=""):
    """(qualified name, node) of every function, method and class under `node`, nested ones included.

    A class counts as one node whose body holds every method, so a reference to a class
    reaches all of its methods.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            name = f"{prefix}{child.name}"
            yield name, child
            yield from _scopes(child, f"{name}.")
        else:
            yield from _scopes(child, prefix)


def _functions_of(tree):
    return dict(_scopes(tree))


def _dotted(name):
    return name.split(".") if name else []


def _import_base(module, node):
    """The package path (posix, no `.py`) a `from ... import` names, resolved against `module`; None outside `modules`."""
    if node.level:
        package = list(Path(module).parent.parts)
        return "/".join([*package[: len(package) - node.level + 1], *_dotted(node.module)])
    parts = _dotted(node.module)
    return "/".join(parts[1:]) if parts[:1] == ["modules"] else None


def _alias_target(base, alias, files):
    """(module file, None) when the import names a module, else (module file, name)."""
    as_module = f"{base}/{alias.name}".lstrip("/") + ".py"
    return (as_module, None) if as_module in files else (f"{base}.py", alias.name)


def _from_imports(module, tree):
    """(package path, node) of every `from ... import` inside `modules`, lazy imports inside functions included."""
    found = ((_import_base(module, node), node) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    return [(base, node) for base, node in found if base is not None]


def _import_aliases(module, tree, files):
    """Local name -> what each `from ... import` in the module binds it to."""
    aliases = {}
    for base, node in _from_imports(module, tree):
        aliases.update({alias.asname or alias.name: _alias_target(base, alias, files) for alias in node.names})
    return aliases


def _binding(node):
    """(name, expression) of a module-level `name = other` or `name = module.attr`, or None."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return None
    target = node.targets[0]
    return (target.id, node.value) if isinstance(target, ast.Name) and isinstance(node.value, (ast.Name, ast.Attribute)) else None


def _assigned_aliases(tree):
    return dict(binding for binding in map(_binding, tree.body) if binding)


class _Graph:
    """Functions and classes by module, and what each reference in a module resolves to."""

    def __init__(self, sources):
        files = set(sources)
        self.functions = {module: _functions_of(tree) for module, tree in sources.items()}
        self.imports = {module: _import_aliases(module, tree, files) for module, tree in sources.items()}
        self.assigned = {module: _assigned_aliases(tree) for module, tree in sources.items()}

    def name(self, module, name, depth=0):
        """(module, function) a bare name in `module` stands for, (MODULE, file) for a module, or None."""
        if depth > MAX_ALIAS_DEPTH or module not in self.functions:
            return None
        if name in self.functions[module]:
            return module, name
        if name in self.assigned[module]:
            return self.expression(module, self.assigned[module][name], depth + 1)
        return self._imported(module, name, depth)

    def _imported(self, module, name, depth):
        target = self.imports[module].get(name)
        if target is None:
            return None
        return (MODULE, target[0]) if target[1] is None else self.name(target[0], target[1], depth + 1)

    def expression(self, module, node, depth=0):
        """What a Name or a `module.attr` Attribute in `module` resolves to (see `name`)."""
        if isinstance(node, ast.Name):
            return self.name(module, node.id, depth)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return self._attribute(module, node, depth)
        return None

    def _attribute(self, module, node, depth):
        base = self.name(module, node.value.id, depth)
        return self.name(base[1], node.attr, depth + 1) if base and base[0] == MODULE else None

    def edges(self, module, function):
        """The functions one function references, its lambdas, nested functions and default values included."""
        targets = {self.expression(module, node) for node in ast.walk(self.functions[module][function])}
        return {target for target in targets if target and target[0] != MODULE}

    def reachable(self, roots):
        seen, todo = set(), list(roots)
        while todo:
            node = todo.pop()
            if node not in seen:
                seen.add(node)
                todo.extend(self.edges(*node) - seen)
        return seen


def _reached():
    sources = _sources()
    return sources, _Graph(sources).reachable(ROOTS)


def test_the_graph_reaches_every_stage_the_cached_point_runs():
    _sources_, reached = _reached()
    assert not MUST_REACH - reached


def _listed_function_sites():
    """The table's read sites that name a function (whole-module sites are the post-cache modules, checked by module)."""
    return {site for _key, (_constant, allowed) in READ_SITES.items() for site in allowed if not site.endswith(f":{WHOLE}")}


def test_no_post_cache_module_post_neural_stage_or_read_site_is_reached():
    _sources_, reached = _reached()
    post_modules = {f"modules/{module}" for module, _function in reached} & stage_cache_key.POST_CACHE_MODULES
    sites = {f"{module}:{function}" for module, function in reached} & _listed_function_sites()
    assert (post_modules, sites, reached & POST_NEURAL_FUNCTIONS) == (set(), set(), set())


def _allowlisted_reads(sources, node):
    """The allowlisted constants and key strings one reached function reads."""
    watched = {constant: key for key, (constant, _sites) in READ_SITES.items()}
    watched.update({key: key for key in READ_SITES})
    visitor = _Reads(node[0], watched)
    visitor.visit(_functions_of(sources[node[0]])[node[1]])
    return visitor.found


def test_nothing_reached_reads_an_allowlisted_key():
    """The direct proof: every function the cached stages can run is free of every allowlisted constant and key string."""
    sources, reached = _reached()
    assert not [read for node in sorted(reached) for read in _allowlisted_reads(sources, node)]


def _step():
    tree = ast.parse((MODULES / "processing.py").read_text(encoding="utf-8"))
    return _functions_of(tree)["_denoise_and_polish_full_audio_step"]


def _through_calls(step):
    return [node for node in ast.walk(step) if isinstance(node, ast.Call) and ast.unparse(node.func) == "_stage_cache.through"]


def test_the_producer_closes_over_the_step_arguments_and_never_its_post_neural_stages():
    (call,) = _through_calls(_step())
    names = {node.id for node in ast.walk(call.args[0].body) if isinstance(node, ast.Name)}
    assert (isinstance(call.args[0], ast.Lambda), names <= PRODUCER_NAMES) == (True, True)


def test_the_flags_the_producer_takes_are_the_seven_neural_switches_in_order():
    (assignment,) = [node for node in ast.walk(_step()) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "flags"]
    values = assignment.value.args[0].args[1]
    assert [element.id for element in values.elts] == list(processing.NEURAL_FLAGS)


def _callers(graph, target):
    """Every function or class whose body references `target`."""
    return {(module, name) for module, functions in graph.functions.items() for name in functions if target in graph.edges(module, name)}


def test_the_polish_builder_and_the_background_pass_have_one_caller_each():
    graph = _Graph(_sources())
    polish = _callers(graph, ("filters.py", "build_full_audio_polish_filter"))
    background = _callers(graph, ("apl_stems.py", "_background_pass"))
    assert (polish, background) == ({("processing.py", "_polish_full_audio_step")}, {("apl_stems.py", "execute")})


def _imports_the_cache(tree):
    names = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    return bool(names & CACHE_MODULES)


def _cache_importers(sources):
    return {module for module, tree in sources.items() if _imports_the_cache(tree)}


def _touches_the_cache(node):
    return node[0].startswith("stage_cache") or node[1] == "_denoise_and_polish_full_audio_step"


def test_cathar_never_reaches_the_cache_and_only_processing_imports_it():
    """cathar's path stays as it was: its mode reaches neither the cache nor the step that calls it."""
    sources = _sources()
    reached = _Graph(sources).reachable([("modes/cathar.py", "CatharMode")])
    touched = [node for node in reached if _touches_the_cache(node)]
    assert (touched, _cache_importers(sources)) == ([], {"processing.py", "stage_cache.py"})


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
# The calls the post-cache modules make at import, each known to set no state the cached stages
# read: sync's optional-import helper and `str` of its import error, mastering's `@dataclass`.
IMPORT_TIME_CALLS = {"_load_optional", "str", "dataclass"}


def _signature(function):
    """What a `def` runs at import: its decorators, default values and annotations."""
    for part in (*function.decorator_list, function.args, function.returns):
        if part is not None:
            yield from ast.walk(part)


def _import_time_nodes(node):
    """Every node that runs when the module is imported: all of it but function and method bodies (class bodies run)."""
    for child in ast.iter_child_nodes(node):
        yield child
        yield from _signature(child) if isinstance(child, FUNCTIONS) else _import_time_nodes(child)


def _stores_into_an_object(node):
    """An assignment into an attribute or an item (`x.flag = 1`, `TABLE[k] = 1`, `x.n += 1`)."""
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    return isinstance(node, (ast.Assign, ast.AugAssign)) and any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in targets)


def _side_effect(node):
    """A call to anything but the known import-time helpers, wherever it sits, or a store into an object."""
    return _stores_into_an_object(node) or (isinstance(node, ast.Call) and ast.unparse(node.func) not in IMPORT_TIME_CALLS)


def _side_effects(tree):
    return [ast.unparse(node) for node in _import_time_nodes(tree) if _side_effect(node)]


def test_the_post_cache_modules_change_nothing_when_imported():
    """Importing them sets no state the cached stages read.

    Their import-time code is also in the code key (`stage_cache_key.import_time_code`), so a
    change to it misses; this keeps it free of calls in the first place, since processing
    imports two of them at start-up and the other two are imported mid-process.
    """
    sources = _sources()
    found = {name: _side_effects(sources[name.removeprefix("modules/")]) for name in stage_cache_key.POST_CACHE_MODULES}
    assert found == dict.fromkeys(stage_cache_key.POST_CACHE_MODULES, [])


IMPORT_TIME_SOURCE = """import x
x.flag = True
x.setup()
TABLE = {}
TABLE['a'] = 1
x.n += 1
N = 2
P = x.set_precision('high')
class K:
    M = x.mode()
    def m(self, d=x.default()):
        x.inner()
@x.wrap()
def f():
    x.inner = 1
T = _load_optional('t')
"""


def test_any_call_at_import_or_a_stored_attribute_is_a_side_effect():
    """Assigned calls, class bodies, defaults and decorators run at import too; function bodies do not."""
    expected = [
        "x.flag = True",
        "x.setup()",
        "TABLE['a'] = 1",
        "x.n += 1",
        "x.set_precision('high')",
        "x.mode()",
        "x.default()",
        "x.wrap()",
    ]
    assert _side_effects(ast.parse(IMPORT_TIME_SOURCE)) == expected
