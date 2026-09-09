# Remote Hardware Validation

Every GPU path in this repository is mocked by the unit and integration suites,
so a broken CUDA route passes them all. Proving a route needs the silicon, and
the machine you develop on is rarely the machine you ship against. This document
describes validating the pipeline on a second machine over SSH.

See [`validation.md`](validation.md) for the canonical local quality gate, which
this supplements rather than replaces.

## One command

```bash
scripts/remote_validate.sh <user>@<host>                  # preflight + hardware audit
scripts/remote_validate.sh <user>@<host> --stage tests    # + the opt-in hardware suite
scripts/remote_validate.sh <user>@<host> --stage execute  # + real fixtures through every mode
scripts/remote_validate.sh <user>@<host> --stage tapes    # + the real VHS corpus
```

The script creates its own SSH key, checks access, syncs the tree, provisions
the environment, audits the hardware and runs the stage you asked for. It stops
with the exact command for the one step that needs your password: authorising
the key.

## What you will be asked for

The **IP or hostname** and the **username**. The username is asked for rather
than guessed: assuming the local one produces a `Permission denied` that looks
like broken key auth when it is only the wrong account.

The validation script never prompts for a password and cannot use one —
`BatchMode=yes` turns a password prompt into an immediate error rather than a
hung session. Authorising the key is the one step that does need your remote
account password, which is why it is run by you, in your own terminal, rather
than by the script:

```bash
ssh-copy-id -i ~/.ssh/vhs_remote_validation.pub <user>@<host>
```

Windows PowerShell ships no `ssh-copy-id`. Either run the line above in Git
Bash, which has one, or use the equivalent from PowerShell:

```powershell
type $env:USERPROFILE\.ssh\vhs_remote_validation.pub | ssh <user>@<host> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

The key is dedicated to this purpose and revocable on its own: delete that one
line from the remote `~/.ssh/authorized_keys` and this script loses access while
nothing else does.

## What the remote host needs

| Requirement | Why |
| :--- | :--- |
| Python **3.12** exactly | `pyproject.toml` pins `>=3.12,<3.13` |
| `ffmpeg` | every restoration mode shells out to it |
| NVIDIA driver + `nvidia-smi` | required by `execute` and `tapes` |
| ~20 GB free in `$HOME` | CUDA wheels, models, fixtures |

On Debian or Ubuntu:

```bash
sudo apt install python3.12 python3.12-venv ffmpeg
```

Rust is **not** a prerequisite. `cathar` is the default restoration mode and is a
Rust binary, so `install_dependencies.sh` provisions one: a system `cargo` is
used when present, otherwise a toolchain is bootstrapped into `.venv/rustup` and
`.venv/cargo` and `cathar-cli` is compiled there. Nothing is installed
system-wide.

Preflight checks each of these and names the missing one, rather than letting
the failure surface half an hour into a dependency install.

## The stages

Each stage contains the one before it.

| Stage | What runs | Roughly |
| :--- | :--- | :--- |
| `audit` | sync, install, `audit_hardware.py` | minutes + first install |
| `tests` | `pytest tests/hardware`, marker enabled | minutes |
| `execute` | Piper fixtures through every mode | tens of minutes to hours |
| `tapes` | the real VHS corpus, measured | one to three hours |

`audit` is the one to run first on any new machine, and it is the only stage
that is meaningful on a host without a GPU.

`tests` sets `AI_RESTORE_HARDWARE_TESTS=1`, which is what
`tests/hardware/conftest.py` gates the `physical_hardware` marker on. Without it
the suite collects, skips and reports a clean pass — a green run that proves
nothing.

`execute` mixes each generated fixture into an MKV and drives it through every
restoration mode, recording elapsed time and peak VRAM per mode. It refuses to
run on a host where the audit reports a CPU fallback, because those numbers
would then measure the CPU.

`tapes` adds the real corpus. Synthetic fixtures have known ground truth, which
is what makes a regression measurable; real transfers are the only material that
shows what the restoration does to damage nobody designed for. It reports noise
reduction, SNR gain, and CRT/mains/rumble suppression per mode, per region and
per genre.

The corpus is gitignored — it is gigabytes of downloaded video — so the
tracked-file sync cannot carry it and `tapes` sends it separately, once per host.
The catalog travels with it: it carries each clip's broadcast standard, and
without it PAL 50 Hz hum would be measured at NTSC's 60 Hz.

```bash
# the curated 44-clip set (~466M), the default
scripts/remote_validate.sh <user>@<host> --stage tapes

# the wide 192-clip set (~1.2G), release depth
scripts/remote_validate.sh <user>@<host> --stage tapes --corpus-dir experiments/ia_corpus_1000
```

## The audit is the authority, not `nvidia-smi`

`scripts/audit_hardware.py` reports what PyTorch actually resolved, not what the
host advertises. A driver `nvidia-smi` lists happily can still leave torch on
the CPU — a mismatched CUDA runtime, a wheel installed from the wrong index —
and every later stage then passes while proving nothing about the accelerator.
The line that matters is:

```json
"pytorch_cuda_ready": true
```

The script checks it, prints a verdict, and refuses `--stage execute` when it is
false.

## Reports

Both reports are copied back into `artifacts/remote/<host>/`:

- `hardware-audit.json` — platform, CPU, `nvidia-smi`, the resolved PyTorch
  device, VRAM, the selected profile and batch size, ONNX providers.
- `hardware-validation.json` — written by `--stage execute`: the hardware
  report, the fixtures used, and per-mode success, elapsed seconds and peak
  VRAM.
- `benchmark_ia_corpus_report.json` / `.md` — written by `--stage tapes`: the
  per-mode, per-region and per-genre measurements over the real corpus.

Pass `--no-fetch` to leave them on the remote.

## Choosing what to run

```bash
# One language, one mode -- the fastest useful physical check
scripts/remote_validate.sh user@host --stage execute --profile short --mode cathar

# Several languages through every mode
scripts/remote_validate.sh user@host --stage execute --language en --language de --language ja
```

Fixtures default to `en` alone. Generating all 50 languages is hours of Piper
synthesis, so widen `--language` deliberately. A matrix already present on the
host is reused unless `--fixtures` forces regeneration — Piper output is
deterministic, so regenerating it changes nothing but the wall clock.

## How the sync works

The file list is `git ls-files`, streamed as a tar over SSH:

- **tar, not rsync**, because Git Bash on Windows — the operator shell this
  repository is developed in — ships `ssh` and `scp` but no `rsync` at all.
- **tracked files, read from the working tree**, so uncommitted edits cross too.
  Validating a branch you have not pushed is the normal case.
- **untracked and ignored files do not cross.** A new module that has never been
  `git add`ed is simply absent on the remote, and the run fails with an
  `ImportError` naming a file that plainly exists locally. Preflight warns about
  untracked sources under `modules/`, `tests/` and `scripts/` for this reason.
- Nothing else crosses. `.venv`, `artifacts/`, `models/`, coverage data and the
  multi-megabyte session log are excluded by construction, with no exclusion
  list to maintain.

The remote checkout lives at `~/ai-hybrid-vhs-audio-restorer` (`--work-dir`
renames it) and is left in place after a failure, so the logs and partial
artifacts survive.

## Re-running

The second run against a host is much faster: it reuses the existing `.venv` and
the existing fixture matrix, syncing only the source. `--install` rebuilds the
environment, `--fixtures` regenerates the matrix. Reach for `--install` after a
dependency change in `poetry.lock`; the sync alone does not reinstall anything.
