"""The stage cache's code key: what importing a post-cache module runs, the start-up snapshot and the drift check."""

import os
import time
from types import SimpleNamespace

import pytest

from modules import stage_cache_key as key_mod
from tests.unit.test_stage_cache_key_outside import _fingerprint, _tree


def _same(source, other):
    return key_mod.import_time_code(source) == key_mod.import_time_code(other)


IMPORT_TIME = [
    ("A = 1\n", "A = 2\n"),
    ("A = 1\n", "import torch\nA = 1\n"),
    ("def f(x=1):\n    return x\n", "def f(x=2):\n    return x\n"),
    ("@d\ndef f():\n    return 1\n", "@e\ndef f():\n    return 1\n"),
    ("class C:\n    N = 1\n", "class C:\n    N = 2\n"),
    ("def g():\n    return 1\ndef f():\n    return g()\nX = f()\n", "def g():\n    return 2\ndef f():\n    return g()\nX = f()\n"),
    (
        "class C:\n    def __init__(self):\n        self.n = 1\nI = C()\n",
        "class C:\n    def __init__(self):\n        self.n = 2\nI = C()\n",
    ),
]


@pytest.mark.parametrize(("source", "changed"), IMPORT_TIME)
def test_what_importing_a_module_runs_is_in_its_import_time_code(source, changed):
    """A statement, a default, a decorator, a class body, and a function import calls (and what it calls)."""
    assert not _same(source, changed)


RUN_LATER = [
    ("def f():\n    return 1\n", "def f():\n    return 2\n"),
    ("class C:\n    def m(self):\n        return 1\n", "class C:\n    def m(self):\n        return 2\n"),
    ("def f():\n    '''One.'''\n", "def f():\n    '''Two.'''\n"),
    ("A = 1\n", "\n\nA = 1  # moved\n"),
]


@pytest.mark.parametrize(("source", "changed"), RUN_LATER)
def test_a_body_only_a_later_call_runs_is_not(source, changed):
    assert _same(source, changed)


def test_a_module_that_does_not_parse_cannot_be_keyed():
    with pytest.raises(ValueError):
        key_mod.import_time_code("def broken(:\n")


def _write(root, relative, text):
    (root / relative).write_text(text, encoding="utf-8")
    return _fingerprint(root)


def test_the_code_key_follows_a_post_cache_modules_import_time_code_only(tmp_path):
    root = _tree(tmp_path / "repo")
    base = _write(root, "modules/sibilant_guard.py", "def f():\n    return 1\n")
    body_only = _write(root, "modules/sibilant_guard.py", "def f():\n    return 2\n")
    at_import = _write(root, "modules/sibilant_guard.py", "import torch\ndef f():\n    return 2\n")
    assert (body_only, at_import != base) == (base, True)


class _Process:
    def __init__(self, created):
        self.created = created

    def create_time(self):
        if isinstance(self.created, Exception):
            raise self.created
        return self.created


def _psutil(created):
    return SimpleNamespace(Process=lambda: _Process(created), Error=RuntimeError)


NOW_NS = 5_000_000_000


@pytest.mark.parametrize(
    ("module", "expected"),
    [(None, NOW_NS), (_psutil(2.0), 2_000_000_000), (_psutil(9.0), NOW_NS), (_psutil(RuntimeError("gone")), NOW_NS)],
)
def test_the_process_start_is_psutils_reading_or_now(monkeypatch, module, expected):
    monkeypatch.setattr(key_mod, "psutil", module)
    assert key_mod.process_started_ns(now_ns=NOW_NS) == expected


def test_the_start_falls_back_to_when_the_key_module_was_imported(monkeypatch):
    monkeypatch.setattr(key_mod, "psutil", None)
    assert key_mod.started_ns() == key_mod.IMPORTED_NS


def test_prime_snapshots_the_code_only_with_the_cache_on(tmp_path):
    root = _tree(tmp_path / "repo")
    key_mod.code_fingerprint.cache_clear()
    try:
        off = (key_mod.prime({}, root), key_mod.code_fingerprint.cache_info().currsize)
        on = key_mod.prime({key_mod.CACHE_ENV_VAR: "C:/cache"}, root)
        snapshot = key_mod.code_fingerprint(root)
        (root / "modules" / "a.py").write_text("A = 2\n", encoding="utf-8")
        assert (off, on, key_mod.code_fingerprint(root)) == ((False, 0), True, snapshot)
    finally:
        key_mod.code_fingerprint.cache_clear()


def test_prime_that_cannot_read_the_code_leaves_it_to_the_key(tmp_path):
    assert key_mod.prime({key_mod.CACHE_ENV_VAR: "C:/cache"}, tmp_path) is False


@pytest.fixture(name="steady")
def steady_fixture(tmp_path, monkeypatch):
    """A tree last written before this process started, and the key document of it."""
    root = _tree(tmp_path / "repo")
    before = time.time_ns() - 10**12
    for path in root.rglob("*.py"):
        os.utime(path, ns=(before, before))
    monkeypatch.setattr(key_mod, "started_ns", lambda: before + 1)
    return root, {"code": _fingerprint(root), "runtime": {"distributions": key_mod._distributions_sha256()}}


def test_nothing_moved_is_no_drift(steady):
    root, document = steady
    assert key_mod.drift(document, root) is None


def test_a_source_written_after_the_process_started_is_drift(steady):
    root, document = steady
    os.utime(root / "modules" / "a.py")
    assert "after this process started" in key_mod.drift(document, root)


def test_a_skewed_clock_is_no_edit_but_a_changed_source_still_is(steady):
    root, document = steady
    source = root / "modules" / "a.py"
    future = time.time_ns() + 10**12
    os.utime(source, ns=(future, future))
    unchanged = key_mod.drift(document, root)
    source.write_text("A = 2\n", encoding="utf-8")
    os.utime(source, ns=(future, future))
    assert (unchanged, "sources changed" in key_mod.drift(document, root)) == (None, True)


def test_packages_that_changed_are_drift(steady):
    root, document = steady
    assert "packages changed" in key_mod.drift({**document, "runtime": {"distributions": "0" * 64}}, root)


def test_drift_reads_this_checkout_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr(key_mod, "_code_drift", lambda _document, root, now_ns: seen.append((root, now_ns > 0)))
    monkeypatch.setattr(key_mod, "_package_drift", lambda _document: None)
    key_mod.drift({})
    assert seen == [(key_mod.PROJECT_ROOT, True)]
