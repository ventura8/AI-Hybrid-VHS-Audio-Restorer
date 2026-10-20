"""What the autotune loop does when a render goes wrong: a candidate that failed, a disk running out.

A candidate whose render leaves no output for some tape used to stop the whole loop
(2026-10-20: `apl_burst_ceiling` `gaps` made a stage name 261 characters long, the app could
not write it on three Tata tapes, and APL's round 3 ended after two minutes while the `both`
launcher went on to cathar). Now `RenderFailed` carries each missing tape's reason, read from
the candidate's run log, and the loop sets the candidate aside like an inert one, says why in
`log.md`, and goes on. The incumbent failing still stops the run: nothing can be judged
without it. So do `MAX_FAILURES_IN_A_ROW` failures in a row, which point at the machine, not
at a setting.

A full disk also failed every render after it (the same day, cathar's round 3 at the third
tape of one candidate: "No space left on device"). `check_free_space` stops the loop before a
render while a drive the tapes or the loop's folder live on has less than `MIN_FREE_GB` free;
the run resumes from `state.json`.
"""

import shutil
from pathlib import Path

# The free space a render needs on the tapes' drive and the loop's: the four restored videos the
# app writes beside the Tata tapes before the loop extracts their audio (about 20 GB), a tape's
# 32-bit work files (about 3 GB), the candidate's WAVs (0.5 GB) and the stage cache's growth.
MIN_FREE_GB = 40.0
GB = 1024**3
# Failed renders in a row that stop the loop: a broken machine (a full disk, a missing binary)
# fails every candidate, and each failure costs a render.
MAX_FAILURES_IN_A_ROW = 3
# The longest reason kept per tape (an error line can carry a 300-character path).
REASON_CHARS = 400
TASK_MARK = "Processing Task: "
ERROR_MARK = "[Error]"


class RenderFailed(SystemExit):
    """A candidate's render left no usable output for some tapes; `reasons` is `{slug: why}`."""

    def __init__(self, cid, reasons):
        self.cid = cid
        self.reasons = dict(reasons)
        listed = "; ".join(f"{slug}: {why}" for slug, why in self.reasons.items())
        super().__init__(f"{cid}: no output for {listed}")


def free_gb(path):
    """Free space in GB on the drive holding `path` (its nearest existing parent)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / GB


def check_free_space(cid, paths, min_gb=None):
    """Stops the loop (SystemExit) before `cid` renders while a drive under `paths` has less than `min_gb` free
    (`MIN_FREE_GB` by default)."""
    limit = MIN_FREE_GB if min_gb is None else min_gb
    for path in paths:
        free = free_gb(path)
        if free < limit:
            raise SystemExit(
                f"{cid}: {free:.1f} GB free under {path}, under the {limit:g} GB a render needs; "
                "stopped before the render, free some space and rerun: the loop resumes from state.json"
            )


def _task_lines(text, tape_name):
    """The run-log lines the app wrote while it restored `tape_name`, from its "Processing Task" line to the next one."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if TASK_MARK in line] + [len(lines)]
    for here, after in zip(starts, starts[1:]):
        if _is_task_of(lines[here], tape_name):
            return lines[here:after]
    return []


def _is_task_of(line, tape_name):
    """Whether a "Processing Task" line is about `tape_name`."""
    return line.split(TASK_MARK, 1)[1].strip() == tape_name


def failure_reason(run_log, tape):
    """The first error the app logged while it restored `tape`, from the candidate's run log; a plain note without one."""
    try:
        text = Path(run_log).read_text(encoding="utf-8", errors="replace").replace("\x1b[K", "")
    except OSError:
        return "no output (no run log)"
    errors = [line.strip() for line in _task_lines(text, Path(tape).name) if ERROR_MARK in line]
    return errors[0][:REASON_CHARS] if errors else "no output (the run log names no error)"


class FailureStreak:
    """Counts failed renders in a row and stops the loop at `MAX_FAILURES_IN_A_ROW`."""

    def __init__(self, limit=MAX_FAILURES_IN_A_ROW):
        self.limit = limit
        self.failures = []

    def passed(self):
        self.failures = []

    def failed(self, error):
        self.failures.append(str(error))
        if len(self.failures) >= self.limit:
            raise SystemExit(f"{self.limit} renders in a row failed, which points at the machine; stopping: " + " | ".join(self.failures))
