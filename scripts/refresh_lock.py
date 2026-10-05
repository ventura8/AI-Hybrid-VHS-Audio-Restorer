#!/usr/bin/env python3
"""Refresh poetry.lock without Poetry's solver: the repo's lock-patch fallback, applied to every package.

    refresh_lock.py [--apply] [--graph pkg==ver ...] [--add name==ver:group ...] [--report experiments/lock_refresh_report.json]

Poetry cannot regenerate this lock: re-resolving the torch cu130 multiple-constraint graph
has run for hours without finishing (see "Verify Poetry Lockfile" in .github/workflows/ci.yml
and the prepare-release skill). This does what that skill describes, mechanically:

- Safe pass (always): for every locked package without a [package.source] (the cu130 torch
  trio and VCS deps keep their pins) take the newest PyPI release that is not a pre-release,
  has non-yanked files, allows Python 3.12, satisfies the pyproject pin when the package is a
  direct dependency, satisfies every constraint the other locked packages put on it, and has
  the same requires_dist / requires_python as the locked version. Rewrite `version` and
  `files`. A newer release whose metadata differs is reported as deferred.
- Graph pass (`--graph name==version`): rebuild that package's block from PyPI metadata
  (version, files, [package.dependencies], [package.extras]), keeping its groups/markers,
  add any requirement the lock lacks, and refuse when a dependent's constraint or one of
  the new requirements' constraints is not met by the lock. Bump the pyproject pin first
  when the package is a direct dependency.
- `--add name==version:group` adds a new direct dependency (and its missing requirements).

The content-hash is recomputed with Poetry's own API. Afterwards run
`poetry check --lock`, a hash-checked `poetry install --with dev,ml --no-root`, and the tests.
"""

import argparse
import json
import re
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version
from poetry.core.constraints.version import Version as PoetryVersion
from poetry.core.constraints.version import parse_constraint
from poetry.core.packages.dependency import Dependency

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "poetry.lock"
PYTHON = PoetryVersion.parse("3.12.10")
CACHE = {}


# ----------------------------------------------------------------------------- PyPI


def pypi(url):
    if url not in CACHE:
        with urllib.request.urlopen(url, timeout=60) as handle:
            CACHE[url] = json.load(handle)
    return CACHE[url]


def project_json(name):
    return pypi(f"https://pypi.org/pypi/{name}/json")


def version_json(name, version):
    return pypi(f"https://pypi.org/pypi/{name}/{version}/json")


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requires_key(info):
    reqs = sorted(re.sub(r"\s+", " ", r.strip()) for r in (info.get("requires_dist") or []))
    return (tuple(reqs), (info.get("requires_python") or "").strip())


def python_ok(info):
    spec = (info.get("requires_python") or "").strip()
    if not spec:
        return True
    try:
        return parse_constraint(spec).allows(PYTHON)
    except Exception:
        return True


def candidates(name):
    """Release versions newest first: no pre-releases, at least one non-yanked file."""
    out = []
    for version, files in project_json(name)["releases"].items():
        try:
            parsed = Version(version)
        except InvalidVersion:
            continue
        if parsed.is_prerelease or parsed.is_devrelease or not any(not f.get("yanked") for f in files):
            continue
        out.append((parsed, version))
    return [v for _p, v in sorted(out, reverse=True)]


def files_of(name, version):
    files = version_json(name, version)["urls"]
    return sorted(
        ({"file": f["filename"], "hash": f"sha256:{f['digests']['sha256']}"} for f in files if not f.get("yanked")), key=lambda f: f["file"]
    )


# ----------------------------------------------------------------------------- constraints


def direct_constraints():
    """`{name: constraint}` for every direct dependency in pyproject (first entry of a multi-constraint list)."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tables = [data["tool"]["poetry"]["dependencies"]] + [g["dependencies"] for g in data["tool"]["poetry"].get("group", {}).values()]
    out = {}
    for table in tables:
        for name, spec in table.items():
            if name == "python":
                continue
            first = spec[0] if isinstance(spec, list) else spec
            out[norm(name)] = first["version"] if isinstance(first, dict) else first
    return out


def dependents(lock):
    """`{name: [(dependent, constraint)]}` from every locked package's [package.dependencies]."""
    out = {}
    for package in lock["package"]:
        for dep, spec in (package.get("dependencies") or {}).items():
            for entry in spec if isinstance(spec, list) else [spec]:
                out.setdefault(norm(dep), []).append((package["name"], entry["version"] if isinstance(entry, dict) else entry))
    return out


def allowed(version, constraints):
    pv = PoetryVersion.parse(version)
    for _who, constraint in constraints:
        try:
            if not parse_constraint(constraint).allows(pv):
                return False
        except Exception:
            continue
    return True


# ----------------------------------------------------------------------------- lock text


def package_span(text, name, version):
    """(start, end) of the [[package]] block for name/version."""
    match = re.compile(rf'^\[\[package\]\]\nname = "{re.escape(name)}"\nversion = "{re.escape(version)}"\n', re.M).search(text)
    assert match, (name, version)
    end = text.find("\n[[package]]", match.end())
    return match.start(), (end + 1 if end != -1 else text.find("\n[metadata]") + 1)


def replace_version_and_files(block, version, files):
    block = re.sub(r'^version = ".*?"$', f'version = "{version}"', block, count=1, flags=re.M)
    rendered = "files = [\n" + "".join(f'    {{file = "{f["file"]}", hash = "{f["hash"]}"}},\n' for f in files) + "]\n"
    return re.sub(r"^files = \[\n(?:.*\n)*?\]\n", rendered, block, count=1, flags=re.M)


def toml_dep_value(dep):
    constraint = dep.pretty_constraint if dep.pretty_constraint != "*" else "*"
    parts = [f'version = "{constraint}"']
    if dep.extras:
        parts.insert(0, "extras = [" + ", ".join(f'"{e}"' for e in sorted(dep.extras)) + "]")
    if not dep.marker.is_any():
        parts.append(f'markers = "{str(dep.marker).replace(chr(34), chr(92) + chr(34))}"')
    return f'"{constraint}"' if len(parts) == 1 else "{" + ", ".join(parts) + "}"


def dependency_lines(requires):
    deps, extras = {}, {}
    for req in requires:
        dep = Dependency.create_from_pep_508(req)
        if dep.in_extras:
            for extra in dep.in_extras:
                extras.setdefault(extra, []).append(f"{dep.name} ({dep.pretty_constraint})" if dep.pretty_constraint != "*" else dep.name)
            continue
        deps[dep.name] = toml_dep_value(dep)
    return deps, extras


def render_block(name, version, groups, markers=None, optional="false"):
    """A full [[package]] block from PyPI metadata; returns it and the non-extra dependencies."""
    info = version_json(name, version)["info"]
    deps, extras = dependency_lines(info.get("requires_dist") or [])
    lines = [
        "[[package]]",
        f'name = "{name}"',
        f'version = "{version}"',
        f'description = "{(info.get("summary") or "").replace(chr(34), chr(92) + chr(34))}"',
        f"optional = {optional}",
        f'python-versions = "{info.get("requires_python") or "*"}"',
        f"groups = {groups}",
    ]
    if markers:
        lines.append(f"markers = {markers}")
    lines += ["files = [", *[f'    {{file = "{f["file"]}", hash = "{f["hash"]}"}},' for f in files_of(name, version)], "]"]
    if deps:
        lines += ["", "[package.dependencies]"] + [f"{k} = {v}" for k, v in sorted(deps.items())]
    if extras:
        lines += ["", "[package.extras]"] + [f"{k} = [" + ", ".join(f'"{e}"' for e in sorted(v)) + "]" for k, v in sorted(extras.items())]
    return "\n".join(lines) + "\n", deps


def block_meta(block):
    groups = re.search(r"^groups = (\[.*\])$", block, re.M).group(1)
    markers = re.search(r'^markers = (".*")$', block, re.M)
    optional = re.search(r"^optional = (true|false)$", block, re.M).group(1)
    return groups, markers.group(1) if markers else None, optional


def insert_block(text, block):
    """Inserts a block before the first [[package]] whose name sorts after it."""
    name = re.search(r'^name = "(.*?)"$', block, re.M).group(1)
    for match in re.finditer(r'^\[\[package\]\]\nname = "(.*?)"\n', text, re.M):
        if match.group(1) > name:
            return text[: match.start()] + block + "\n" + text[match.start() :]
    metadata = text.find("[metadata]")
    return text[:metadata] + block + "\n" + text[metadata:]


def recompute_hash(text):
    from poetry.factory import Factory

    digest = Factory().create_poetry(str(ROOT)).locker._get_content_hash()
    return re.sub(r'^content-hash = ".*?"$', f'content-hash = "{digest}"', text, count=1, flags=re.M)


# ----------------------------------------------------------------------------- passes


class Refresh:
    def __init__(self):
        self.text = LOCK.read_text(encoding="utf-8")
        self.lock = tomllib.loads(self.text)
        self.direct = direct_constraints()
        self.dependents = dependents(self.lock)
        self.locked = {norm(p["name"]): p["version"] for p in self.lock["package"]}
        self.report = {"bumped": {}, "deferred": {}, "kept": [], "source_pinned": [], "added": {}, "problems": []}

    def constraints_on(self, name):
        constraints = list(self.dependents.get(norm(name), []))
        if norm(name) in self.direct:
            constraints.append(("pyproject", self.direct[norm(name)]))
        return constraints

    def choose(self, name, current):
        """Newest metadata-identical, constraint-respecting version; a dict when the newest allowed differs in metadata."""
        current_key = requires_key(version_json(name, current)["info"])
        for candidate in candidates(name):
            if Version(candidate) <= Version(current):
                return None
            if not allowed(candidate, self.constraints_on(name)):
                continue
            info = version_json(name, candidate)["info"]
            if not python_ok(info):
                continue
            return candidate if requires_key(info) == current_key else {"deferred": candidate}
        return None

    def safe_pass(self):
        for package in self.lock["package"]:
            name, current = package["name"], package["version"]
            if package.get("source"):
                self.report["source_pinned"].append(f"{name} {current}")
                continue
            try:
                choice = self.choose(name, current)
            except Exception as exc:
                self.report["deferred"][name] = {"from": current, "reason": f"lookup failed: {exc}"}
                continue
            self._apply_choice(name, current, choice)

    def _apply_choice(self, name, current, choice):
        if choice is None:
            self.report["kept"].append(f"{name} {current}")
        elif isinstance(choice, dict):
            self.report["deferred"][name] = {"from": current, "to": choice["deferred"], "reason": "dependency metadata changed"}
        else:
            start, end = package_span(self.text, name, current)
            self.text = (
                self.text[:start] + replace_version_and_files(self.text[start:end], choice, files_of(name, choice)) + self.text[end:]
            )
            self.locked[norm(name)] = choice
            self.report["bumped"][name] = {"from": current, "to": choice}
        print(f"{name}: {current} -> {choice}", flush=True)

    def graph_bump(self, name, version):
        current = self.locked[norm(name)]
        if not allowed(version, self.constraints_on(name)):
            self.report["problems"].append(
                f"{name} {version} violates {[c for c in self.constraints_on(name) if not allowed(version, [c])]}"
            )
            return
        start, end = package_span(self.text, name, current)
        groups, markers, optional = block_meta(self.text[start:end])
        block, deps = render_block(name, version, groups, markers, optional)
        self.text = self.text[:start] + block + "\n" + self.text[end:].lstrip("\n")
        self.locked[norm(name)] = version
        self.report["bumped"][name] = {"from": current, "to": version, "graph": True}
        self._settle_requirements(name, deps, "ml")
        print(f"{name}: {current} -> {version} (graph)", flush=True)

    def _settle_requirements(self, name, deps, group):
        for dep, value in deps.items():
            constraint = re.search(r'version = "(.*?)"', value).group(1) if value.startswith("{") else value.strip('"')
            locked = self.locked.get(norm(dep))
            if locked is None:
                self.add(dep, constraint, group)
            elif not allowed(locked, [(name, constraint)]):
                self.report["problems"].append(f"{name} needs {dep} {constraint}, lock has {locked}")

    def add(self, name, spec, group):
        """Adds a package absent from the lock at the newest release satisfying `spec`, then its missing requirements."""
        if norm(name) in self.locked:
            return
        version = next((c for c in candidates(name) if allowed(c, [("spec", spec)]) and python_ok(version_json(name, c)["info"])), None)
        if version is None:
            self.report["problems"].append(f"no release of {name} satisfies {spec}")
            return
        block, deps = render_block(name, version, f'["{group}"]')
        self.text = insert_block(self.text, block)
        self.locked[norm(name)] = version
        self.report["added"][name] = version
        self._settle_requirements(name, deps, group)

    def finish(self, apply, report_path):
        self.text = recompute_hash(self.text)
        if apply and not self.report["problems"]:
            LOCK.write_text(self.text, encoding="utf-8")
        report_path = _report_path_allowed(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.report, indent=1), encoding="utf-8")
        r = self.report
        print(
            f"bumped {len(r['bumped'])}, deferred {len(r['deferred'])}, kept {len(r['kept'])}, added {list(r['added'])}, "
            f"source-pinned {len(r['source_pinned'])}, problems {r['problems']}"
        )
        return 1 if r["problems"] else 0


def _report_path_allowed(report_path):
    """The report path resolved, and refused unless it lies inside the repository or the temp directory.

    The path comes from the command line; a report belongs under `experiments/` or a scratch
    directory, never anywhere a stray argument could point.
    """
    resolved = Path(report_path).resolve()
    allowed = (ROOT.resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(resolved.is_relative_to(base) for base in allowed):
        raise SystemExit(f"--report must lie inside {allowed[0]} or {allowed[1]}: {report_path}")
    return resolved


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--graph", nargs="*", default=[], help="name==version bumps whose dependency metadata changed")
    parser.add_argument("--add", nargs="*", default=[], help="name==version:group new direct dependencies")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "lock_refresh_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    refresh = Refresh()
    refresh.safe_pass()
    for item in args.graph:
        name, version = item.split("==")
        refresh.graph_bump(name, version)
    for item in args.add:
        name_version, _colon, group = item.partition(":")
        name, version = name_version.split("==")
        refresh.add(name, f"=={version}", group or "ml")
    return refresh.finish(args.apply, args.report)


if __name__ == "__main__":
    sys.exit(main())
