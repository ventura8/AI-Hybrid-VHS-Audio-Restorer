#!/usr/bin/env bash
# Validate this restorer on a remote machine's real accelerator, over SSH.
#
# The local test suite mocks every GPU path, so a broken CUDA route passes it. Proving a
# route needs the silicon: this script points the repository at a machine that has it,
# audits what is really there, provisions the environment, and runs the opt-in hardware
# validation on it.
#
# Usage:
#   scripts/remote_validate.sh user@host                        # preflight + hardware audit
#   scripts/remote_validate.sh user@host --stage tests          # + opt-in hardware test suite
#   scripts/remote_validate.sh user@host --stage execute        # + real fixtures through the modes
#   scripts/remote_validate.sh user@host --stage tapes          # + the real VHS corpus
#
# Options:
#   --stage NAME      audit (default) | tests | execute | tapes
#                       audit    preflight, sync, environment, scripts/audit_hardware.py
#                       tests    audit + AI_RESTORE_HARDWARE_TESTS=1 pytest tests/hardware
#                       execute  tests + scripts/run_hardware_validation.py --execute, which
#                                drives generated MKV fixtures through the restoration modes
#                                and records elapsed time and peak VRAM. Needs NVIDIA CUDA.
#                       tapes    execute + the real Internet Archive VHS corpus through
#                                scripts/benchmark_ia_corpus.py, which measures noise
#                                reduction, SNR gain, and CRT/mains/rumble suppression per
#                                mode. Synthetic fixtures prove the code path; only real
#                                tapes show what the restoration does to actual material.
#   --corpus-dir DIR  Real-tape corpus for --stage tapes, relative to the repository root.
#                     Default: experiments/ia_corpus (44 clips, ~466M). The larger
#                     experiments/ia_corpus_1000 (192 clips, ~1.2G) is release-depth.
#   --corpus-refresh  Re-send the corpus even when the remote already has it.
#   --profile NAME    Fixture profile for --stage execute: short | mid | longform | core.
#                     Default: core (short + mid).
#   --language CODE   Fixture language, repeatable. Default: en. Generating all 50 is hours
#                     of Piper synthesis, so widen this deliberately.
#   --mode NAME       Restoration mode for --stage execute, repeatable. Default: every mode
#                     in scripts/run_hardware_validation.py DEFAULT_MODES.
#   --fixtures        Regenerate the remote audio matrix even if it is already there.
#   --install         Re-run install_dependencies.sh even if the remote .venv exists.
#   --work-dir NAME   Remote checkout directory name. Default: ai-hybrid-vhs-audio-restorer.
#   --key PATH        SSH identity. Default: ~/.ssh/vhs_remote_validation (auto-created).
#   --no-fetch        Leave the reports on the remote instead of copying them back.
set -euo pipefail

REPO_ROOT="${VHS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${HOME}/.ssh/vhs_remote_validation"
REMOTE=""
STAGE="audit"
PROFILE="core"
LANGUAGES=()
MODES=()
FIXTURES=false
CORPUS_REL="experiments/ia_corpus"
CORPUS_REFRESH=false
INSTALL=false
FETCH=true
WORK_NAME="ai-hybrid-vhs-audio-restorer"

# Validated during parsing, not where each value is consumed: everything below is minutes
# to hours of remote work, and a typo reported after the sync and the dependency install is
# a typo reported half an hour late.
while [ $# -gt 0 ]; do
	case "$1" in
	--stage)
		STAGE="${2:?--stage needs a value}"
		case "$STAGE" in audit | tests | execute | tapes) ;;
		*)
			echo "unknown --stage '$STAGE' (audit|tests|execute|tapes)" >&2
			exit 2
			;;
		esac
		shift 2
		;;
	--corpus-dir)
		CORPUS_REL="${2:?--corpus-dir needs a value}"
		shift 2
		;;
	--corpus-refresh)
		CORPUS_REFRESH=true
		shift
		;;
	--profile)
		PROFILE="${2:?--profile needs a value}"
		case "$PROFILE" in short | mid | longform | core) ;;
		*)
			echo "unknown --profile '$PROFILE' (short|mid|longform|core)" >&2
			exit 2
			;;
		esac
		shift 2
		;;
	--language)
		LANGUAGES+=("${2:?--language needs a value}")
		shift 2
		;;
	--mode)
		MODES+=("${2:?--mode needs a value}")
		shift 2
		;;
	--fixtures)
		FIXTURES=true
		shift
		;;
	--install)
		INSTALL=true
		shift
		;;
	--work-dir)
		WORK_NAME="${2:?--work-dir needs a value}"
		shift 2
		;;
	--key)
		KEY="${2:?--key needs a value}"
		shift 2
		;;
	--no-fetch)
		FETCH=false
		shift
		;;
	-h | --help)
		grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'
		exit 0
		;;
	*@*)
		REMOTE="$1"
		shift
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done

hdr() { printf '\n=== %s ===\n' "$*"; }
note() { printf '  %s\n' "$*"; }
die() {
	printf '\nERROR: %s\n' "$*" >&2
	exit 1
}

# Asked for rather than assumed: guessing the local username produces a "Permission denied"
# that reads like broken key auth when it is only the wrong account.
if [ -z "$REMOTE" ]; then
	read -r -p "Remote user@host: " REMOTE
fi
case "$REMOTE" in *@*) ;; *) die "expected user@host, got '$REMOTE'" ;; esac
REMOTE_HOST="${REMOTE#*@}"
WORK_DIR="~/${WORK_NAME}"

[ ${#LANGUAGES[@]} -gt 0 ] || LANGUAGES=(en)

# BatchMode turns a would-be password prompt into an immediate error rather than a session
# that hangs forever with no output.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes)

ssh_run() { ssh -i "$KEY" "${SSH_OPTS[@]}" "$REMOTE" "$@"; }

# Every remote stage runs through the project's own virtualenv, exactly as
# run_pipeline_locally.sh does locally. PYTHONPATH is not needed: the scripts insert the
# repository root themselves.
venv_run() { ssh_run "cd ${WORK_DIR} && .venv/bin/python -m poetry run $*"; }

# Defined before the first call, not beside the stage that uses it: the audit-only path
# returns early, and a function defined below that point does not exist yet when it runs.
fetch_reports() {
	hdr "Reports"
	local dest="${REPO_ROOT}/artifacts/remote/${REMOTE_HOST}"
	mkdir -p "$dest"
	# Failure tolerated: hardware-validation.json only exists after --stage execute, and one
	# missing file must not turn a completed audit into a failed run.
	scp -i "$KEY" "${SSH_OPTS[@]}" -q \
		"${REMOTE}:${WORK_NAME}/artifacts/hardware-audit.json" \
		"${REMOTE}:${WORK_NAME}/artifacts/hardware-validation.json" \
		"${REMOTE}:${WORK_NAME}/experiments/benchmark_ia_corpus_report.json" \
		"${REMOTE}:${WORK_NAME}/experiments/benchmark_ia_corpus_report.md" \
		"$dest/" 2>/dev/null || true
	for report in "$dest"/*.json "$dest"/*.md; do
		[ -e "$report" ] && note "$report"
	done
	return 0
}

hdr "Preflight: ${REMOTE}"

if [ ! -f "$KEY" ]; then
	note "No identity at ${KEY} -- generating a dedicated one (revocable on its own)."
	mkdir -p "$(dirname "$KEY")"
	ssh-keygen -t ed25519 -N '' -C 'ai-hybrid-vhs-audio-restorer remote hardware validation' -f "$KEY" >/dev/null
fi

if ! SSH_ERR="$(ssh_run true 2>&1)"; then
	# A refused connection and a rejected key need different fixes, and ssh reports both by
	# failing. Naming the wrong one sends the operator to ssh-copy-id against a host with no
	# sshd -- where it fails identically and the real cause never surfaces.
	case "$SSH_ERR" in
	*"Connection refused"* | *"Connection timed out"* | *"No route to host"*)
		cat <<EOF

Cannot reach an SSH server on ${REMOTE_HOST}:

    ${SSH_ERR}

The host answers, but nothing is listening on port 22. Desktop Ubuntu does not install an
SSH server by default. On that machine:

    sudo apt install -y openssh-server
    sudo systemctl enable --now ssh

Then authorise this key from your own terminal and re-run:

    ssh-copy-id -i ${KEY}.pub ${REMOTE}
EOF
		exit 2
		;;
	esac
	# The PowerShell line needs a PowerShell path. Printing the Git Bash one -- which is what
	# $KEY holds when the script runs, e.g. /c/Users/ventu/.ssh/... -- gave a command that
	# PowerShell cannot resolve at all, in the one message an operator is meant to copy
	# verbatim. cygpath is present in Git Bash; elsewhere the bash form is the right one anyway.
	WIN_KEY="$(cygpath -w "$KEY" 2>/dev/null || printf '%s' "$KEY")"
	cat <<EOF

Key authentication is not set up for ${REMOTE}.

    ${SSH_ERR}

Authorise it yourself -- this is the one step that needs your password, and it has to be
typed by you, in your own terminal:

    ssh-copy-id -i ${KEY}.pub ${REMOTE}

Windows PowerShell has no ssh-copy-id, so from there use the equivalent one-liner:

    type ${WIN_KEY}.pub | ssh ${REMOTE} "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

Then re-run this script. If it still fails the username is probably wrong, or the remote
~/.ssh permissions are too open (chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys).
EOF
	exit 2
fi
note "ssh: OK ($(ssh_run 'echo "$(whoami)@$(hostname) $(uname -sr)"'))"

# Python 3.12 exactly: pyproject pins >=3.12,<3.13. The resolution order mirrors
# install_dependencies.sh -- PATH, ~/.local/bin, a python3 that happens to be 3.12, then a
# uv-managed one -- because a preflight stricter than the installer it guards rejects hosts
# the installer handles fine (Ubuntu 26.04 ships 3.14 and packages no 3.12).
REMOTE_PY="$(ssh_run 'if command -v python3.12 >/dev/null 2>&1; then command -v python3.12;
	elif [ -x "$HOME/.local/bin/python3.12" ]; then echo "$HOME/.local/bin/python3.12";
	elif python3 -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))" 2>/dev/null; then command -v python3;
	elif command -v uv >/dev/null 2>&1; then uv python find 3.12 2>/dev/null || true;
	elif [ -x "$HOME/.local/bin/uv" ]; then "$HOME/.local/bin/uv" python find 3.12 2>/dev/null || true;
	fi' | tr -d '\r' | tail -1)"
if [ -z "$REMOTE_PY" ]; then
	cat <<EOF

Python 3.12 was not found on ${REMOTE_HOST}, and pyproject.toml pins >=3.12,<3.13.

Where the distribution packages one:

    sudo apt install python3.12 python3.12-venv

Ubuntu 26.04 and later package no 3.12 at all -- they ship 3.14. Use uv there, which
install_dependencies.sh already looks for, and which keeps the interpreter under the user's
own home instead of touching the system one:

    curl -LsSf https://astral.sh/uv/install.sh | sh
    ~/.local/bin/uv python install 3.12
EOF
	die "no Python 3.12 on ${REMOTE_HOST}"
fi
note "python: ${REMOTE_PY} ($(ssh_run "${REMOTE_PY} --version" | tr -d '\r'))"

ssh_run 'command -v ffmpeg >/dev/null' ||
	die "ffmpeg not found on ${REMOTE_HOST}; every restoration mode shells out to it (sudo apt install ffmpeg)"
note "ffmpeg: $(ssh_run 'ffmpeg -version | head -1' | tr -d '\r')"

# A missing nvidia-smi is not fatal for the audit -- a CPU host is still worth auditing --
# but --stage execute requires CUDA, so it is refused now rather than after the sync and the
# dependency install.
if ssh_run 'command -v nvidia-smi >/dev/null'; then
	note "gpu: $(ssh_run 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' | tr -d '\r')"
else
	note "gpu: no nvidia-smi on this host"
	case "$STAGE" in execute | tapes) die "--stage ${STAGE} requires NVIDIA CUDA, and ${REMOTE_HOST} has no nvidia-smi." ;; esac
fi
note "disk: $(ssh_run 'df -BG --output=avail ~ | tail -1 | tr -d " "') free in \$HOME"

hdr "Sync source -> ${REMOTE_HOST}"
# tar over ssh rather than rsync: Git Bash on Windows has ssh and scp but no rsync. The
# file list is `git ls-files` read from the working tree, so uncommitted edits cross while
# .venv, artifacts/, models/ and caches are excluded with no exclusion list to maintain.
ssh_run "mkdir -p ${WORK_DIR}"
git -C "$REPO_ROOT" ls-files -z |
	tar -C "$REPO_ROOT" --null -T - -czf - |
	ssh_run "tar -C ${WORK_DIR} -xzf -" ||
	die "sync to ${REMOTE_HOST} failed"
note "synced $(git -C "$REPO_ROOT" ls-files | wc -l | tr -d ' ') tracked files at commit $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# The sync carries tracked files only, so a new module that has never been `git add`ed is
# silently absent on the remote and the run dies on an ImportError naming a file that plainly
# exists locally. Cheaper to say so here than to debug it there.
UNTRACKED_SOURCE="$(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- 'modules/*.py' 'tests/*.py' 'scripts/*.py' | tr '\n' ' ')"
if [ -n "${UNTRACKED_SOURCE// /}" ]; then
	note "WARNING: untracked source files were NOT synced (git add them first):"
	for path in $UNTRACKED_SOURCE; do note "  $path"; done
fi

# Git on Windows checks scripts out with CRLF and the tar copies bytes verbatim, so the
# shebang arrives with a trailing CR and Linux reports a confusing "not found" for bash
# itself. Scripts only: Python reads CRLF fine, and rewriting everything would corrupt
# any binary a future commit tracks.
ssh_run "find ${WORK_DIR} -name '*.sh' -type f -exec sed -i 's/\r\$//' {} +" ||
	die "could not normalise line endings on ${REMOTE_HOST}"

hdr "Environment"
if [ "$INSTALL" = true ] || ! ssh_run "test -x ${WORK_DIR}/.venv/bin/python"; then
	note "running install_dependencies.sh (the first run pulls the CUDA wheels; this takes a while)"
	# One worker, not the default pool. Poetry's parallel installer deadlocks when the index
	# drops connections mid-download: every socket sits in CLOSE-WAIT, every worker thread
	# blocks in futex_do_wait with no timeout, and the install hangs forever at 0% CPU.
	# Measured on this host after 56 minutes. Serial is slower and finishes.
	ssh_run "cd ${WORK_DIR} && chmod +x install_dependencies.sh && POETRY_INSTALLER_MAX_WORKERS=1 ./install_dependencies.sh" ||
		die "remote dependency install failed"
else
	note "reusing the existing ${WORK_DIR}/.venv (pass --install to rebuild it)"
fi
# Single quotes around the -c program, double quotes around the whole argument: the string
# crosses this shell, ssh and the remote shell, and an escaped quote does not survive all
# three.
note "torch: $(venv_run "python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'" | tr -d '\r')"

# install_dependencies.sh installs `--without dev`, which is right for a machine that only
# restores audio -- but pytest is a dev dependency. Added only for the stages that run it,
# so an audit-only run still provisions exactly what the product ships with.
if [ "$STAGE" != "audit" ] && ! venv_run "python -c 'import pytest'" >/dev/null 2>&1; then
	note "installing the dev dependency group (pytest is not part of the runtime install)"
	ssh_run "cd ${WORK_DIR} && .venv/bin/python -m poetry install --no-root --only dev --no-interaction" ||
		die "could not install the dev dependency group on ${REMOTE_HOST}"
fi

hdr "Hardware audit"
# The audit is the authority on what the machine can do, because it reports what PyTorch
# itself resolved rather than what the host advertises. A driver nvidia-smi lists happily
# can still leave torch on the CPU, and every later stage would then pass while proving
# nothing about the accelerator.
ssh_run "mkdir -p ${WORK_DIR}/artifacts"
venv_run "python scripts/audit_hardware.py | tee artifacts/hardware-audit.json" || die "hardware audit failed"

if ssh_run "grep -q '\"pytorch_cuda_ready\": true' ${WORK_DIR}/artifacts/hardware-audit.json"; then
	note "verdict: PyTorch is executing on CUDA"
else
	note "verdict: PyTorch fell back to the CPU on this host"
	case "$STAGE" in execute | tapes) die "--stage ${STAGE} needs CUDA, and the audit reports a CPU fallback. A run would prove nothing." ;; esac
fi

if [ "$STAGE" = "audit" ]; then
	[ "$FETCH" = true ] && fetch_reports
	cat <<EOF

Audit complete. Re-run with a deeper stage to validate the restoration paths:

    scripts/remote_validate.sh ${REMOTE} --stage tests
    scripts/remote_validate.sh ${REMOTE} --stage execute
EOF
	exit 0
fi

hdr "Hardware test suite"
# AI_RESTORE_HARDWARE_TESTS=1 is what tests/hardware/conftest.py gates the physical_hardware
# marker on. Without it the suite collects, skips and reports a clean pass.
venv_run "env AI_RESTORE_HARDWARE_TESTS=1 python -m pytest tests/hardware -v --no-header -o addopts=" ||
	SUITE_FAILED=true

if [ "$STAGE" = "execute" ] || [ "$STAGE" = "tapes" ]; then
	FIXTURE_ARGS=""
	for lang in "${LANGUAGES[@]}"; do FIXTURE_ARGS="${FIXTURE_ARGS} --language ${lang}"; done

	hdr "Audio matrix fixtures (${PROFILE}: ${LANGUAGES[*]})"
	# Piper synthesis is minutes per language and fully deterministic, so a matrix already on
	# the host is reused unless --fixtures says otherwise. Generated on the remote rather than
	# copied: the fixtures are large, and the alternative needs the Piper toolchain and the
	# voice downloads on this machine anyway.
	if [ "$FIXTURES" = true ] || ! ssh_run "test -d ${WORK_DIR}/artifacts/audio-matrix"; then
		venv_run "python scripts/generate_audio_matrix.py ${PROFILE}${FIXTURE_ARGS}" ||
			die "fixture generation failed on ${REMOTE_HOST}"
	else
		note "reusing ${WORK_DIR}/artifacts/audio-matrix (pass --fixtures to regenerate)"
	fi

	MODE_ARGS=""
	for mode in ${MODES[@]+"${MODES[@]}"}; do MODE_ARGS="${MODE_ARGS} --mode ${mode}"; done

	hdr "Physical run: ${PROFILE} fixtures through ${MODES[*]:-every mode}"
	note "each mode is muxed to MKV and processed end to end; expect this to run long"
	venv_run "python scripts/run_hardware_validation.py ${PROFILE}${FIXTURE_ARGS}${MODE_ARGS} --execute" ||
		SUITE_FAILED=true

	# run_hardware_validation.py records per-mode outcomes as {"success": false} and still
	# exits 0, so a run where every mode failed would otherwise report "result: passed". The
	# report is the result here, not the exit code.
	FAILURES="$(ssh_run "grep -c '\"success\": false' ${WORK_DIR}/artifacts/hardware-validation.json || true" | tr -dc '0-9')"
	if [ -n "$FAILURES" ] && [ "$FAILURES" -gt 0 ]; then
		note "${FAILURES} mode(s) FAILED on ${REMOTE_HOST} -- see the report for which"
		SUITE_FAILED=true
	fi
fi

if [ "$STAGE" = "tapes" ]; then
	CORPUS_SRC="${REPO_ROOT}/${CORPUS_REL}"
	[ -d "$CORPUS_SRC" ] ||
		die "no corpus at ${CORPUS_SRC}. Curate one first with scripts/curate_ia_corpus.py, or pass --corpus-dir."

	hdr "Real tape corpus -> ${REMOTE_HOST}"
	# The corpus is gitignored -- deliberately, it is gigabytes of downloaded video -- so the
	# `git ls-files` sync above cannot carry it and the remote would run the benchmark against
	# an empty directory. Sent separately, and only once: it is static material, so a corpus
	# already on the host is reused unless --corpus-refresh says otherwise.
	if [ "$CORPUS_REFRESH" = true ] || ! ssh_run "test -d ${WORK_DIR}/${CORPUS_REL}"; then
		note "sending $(du -sh "$CORPUS_SRC" | cut -f1) of real tape material (this is the slow part)"
		ssh_run "mkdir -p ${WORK_DIR}/$(dirname "$CORPUS_REL")"
		tar -C "$REPO_ROOT" -czf - "$CORPUS_REL" | ssh_run "tar -C ${WORK_DIR} -xzf -" ||
			die "corpus transfer to ${REMOTE_HOST} failed"
		# The catalog carries each clip's region, broadcast standard and the mains/CRT
		# frequencies the metrics are measured against. Without it the benchmark cannot tell a
		# PAL 50 Hz tape from an NTSC 60 Hz one, and every hum measurement is taken at the
		# wrong frequency.
		for catalog in "experiments/ia_corpus_catalog.json" "${CORPUS_REL}/catalog_1000.json"; do
			[ -f "${REPO_ROOT}/${catalog}" ] || continue
			tar -C "$REPO_ROOT" -czf - "$catalog" | ssh_run "tar -C ${WORK_DIR} -xzf -" || true
		done
	else
		note "reusing ${WORK_DIR}/${CORPUS_REL} (pass --corpus-refresh to re-send)"
	fi
	note "clips on remote: $(ssh_run "find ${WORK_DIR}/${CORPUS_REL} -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.avi' \) | wc -l" | tr -d '\r')"

	BENCH_MODES=""
	[ ${#MODES[@]} -gt 0 ] && BENCH_MODES="--modes ${MODES[*]}"

	hdr "Real tape benchmark (${CORPUS_REL})"
	note "measures noise reduction, SNR gain, and CRT/mains/rumble suppression per mode"
	venv_run "python scripts/benchmark_ia_corpus.py --corpus-dir ${CORPUS_REL} ${BENCH_MODES}" ||
		SUITE_FAILED=true
fi

[ "$FETCH" = true ] && fetch_reports

hdr "Done"
note "host:  ${REMOTE_HOST}"
note "stage: ${STAGE}"
if [ "${SUITE_FAILED:-false}" = true ]; then
	note "result: FAILED (see the output above; the remote tree is left in place at ${WORK_DIR})"
	exit 1
fi
note "result: passed"
