#!/usr/bin/env bash
# === AI Hybrid VHS Audio Restorer Installer (Linux & macOS) ===
# Sets up Python 3.12 virtual environment, runtime & ML dependencies, and launchers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up AI Hybrid VHS Audio Restorer Environment (Linux / macOS) ==="

if [ "$(uname -s)" = "Darwin" ]; then
    MAC_ARCH="$(uname -m)"
    if [ "$MAC_ARCH" = "arm64" ]; then
        if [ "$(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
            echo "WARNING: Running under Rosetta. Use a native arm64 terminal for MPS acceleration." >&2
        else
            echo "Detected native Apple Silicon ($MAC_ARCH); MPS acceleration will be available when supported by PyTorch."
        fi
    else
        echo "Detected macOS $MAC_ARCH. This package also supports Apple Silicon through the native arm64 release asset."
    fi
fi

# Step 1: Detect Python 3.12
echo -e "\nStep 1: Checking Python 3.12..."
PYTHON_BIN=""

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif [ -f "$HOME/.local/bin/python3.12" ]; then
    PYTHON_BIN="$HOME/.local/bin/python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ "$PY_VER" = "3.12" ]; then
        PYTHON_BIN="python3"
    fi
fi

if [ -z "$PYTHON_BIN" ] && command -v uv >/dev/null 2>&1; then
    UV_PY="$(uv python find 3.12 2>/dev/null || true)"
    if [ -n "$UV_PY" ] && [ -x "$UV_PY" ]; then
        PYTHON_BIN="$UV_PY"
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.12 is required (>=3.12,<3.13) but was not found." >&2
    echo "Please install Python 3.12 (e.g. 'sudo apt install python3.12 python3.12-venv' or 'brew install python@3.12')." >&2
    exit 1
fi

echo "Found Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# Step 2: Create Virtual Environment
echo -e "\nStep 2: Setting up Python Virtual Environment..."
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [ -x "$VENV_PY" ]; then
    VENV_PY_VER="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ "$VENV_PY_VER" != "3.12" ]; then
        echo "ERROR: Existing .venv uses Python $VENV_PY_VER. Please remove .venv and rerun this script." >&2
        exit 1
    fi
fi

if [ ! -f "$VENV_PY" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Step 3: Verify Media Binaries (FFmpeg & FFprobe)
echo -e "\nStep 3: Checking FFmpeg and FFprobe..."
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    echo "Found system media tools: $(which ffmpeg) and $(which ffprobe)"
elif [ -f "$VENV_DIR/bin/ffmpeg" ] && [ -f "$VENV_DIR/bin/ffprobe" ]; then
    echo "Found venv media tools: $VENV_DIR/bin/ffmpeg and $VENV_DIR/bin/ffprobe"
else
    echo "ERROR: FFmpeg or FFprobe not found. Please install FFmpeg using your package manager:" >&2
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "  brew install ffmpeg" >&2
    else
        echo "  sudo apt-get install -y ffmpeg" >&2
    fi
    echo "Re-run ./install_dependencies.sh once FFmpeg and FFprobe are on PATH." >&2
    exit 1
fi

FFMPEG_DETECTED="$(command -v ffmpeg 2>/dev/null || echo "$VENV_DIR/bin/ffmpeg")"
if [ -x "$FFMPEG_DETECTED" ]; then
    if ! "$FFMPEG_DETECTED" -filters 2>/dev/null | grep -q "arnndn"; then
        echo "WARNING: Detected FFmpeg does not support the 'arnndn' filter. 'arnndn_speech' mode will be unavailable." >&2
    fi
fi

# Step 3b: Verify Cathar Audio Restoration Toolkit
echo -e "\nStep 3b: Checking Cathar audio restoration toolkit..."
CATHAR_EXPECTED_VER="0.7.0"

check_cathar_ver() {
    local bin="$1"
    if [ -x "$bin" ]; then
        local ver
        ver="$("$bin" --version 2>/dev/null || true)"
        if echo "$ver" | grep -q "^cathar 0\.7\.3$"; then
            return 0
        fi
    fi
    return 1
}

if check_cathar_ver "$VENV_DIR/bin/cathar"; then
    echo "Found verified venv Cathar CLI ($CATHAR_EXPECTED_VER): $VENV_DIR/bin/cathar"
else
    rm -f "$VENV_DIR/bin/cathar"
    if check_cathar_ver "$(which cathar 2>/dev/null)"; then
        echo "Found system Cathar CLI ($CATHAR_EXPECTED_VER): $(which cathar)"
        cp "$(which cathar)" "$VENV_DIR/bin/cathar"
    elif check_cathar_ver "$HOME/.cargo/bin/cathar"; then
        echo "Found cargo Cathar CLI ($CATHAR_EXPECTED_VER): $HOME/.cargo/bin/cathar"
        cp "$HOME/.cargo/bin/cathar" "$VENV_DIR/bin/cathar"
    elif command -v cargo >/dev/null 2>&1; then
        echo "Installing cathar-cli==$CATHAR_EXPECTED_VER via cargo into venv..."
        if ! cargo install cathar-cli --version "$CATHAR_EXPECTED_VER" --root "$VENV_DIR"; then
            echo "WARNING: Cathar compilation/installation failed. Continuing without Cathar."
        fi
    else
        echo "WARNING: Cathar CLI not found. Run 'cargo install cathar-cli --version $CATHAR_EXPECTED_VER' to enable cathar mode."
    fi
fi

# Step 4: Install Dependencies via Poetry
echo -e "\nStep 4: Installing Dependencies via Poetry..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install poetry==2.4.1

"$VENV_PY" -m poetry config --local virtualenvs.in-project true
"$VENV_PY" -m poetry config --local virtualenvs.create false

echo "Provisioning isolated Piper fixture runtime..."
export POETRY_REQUESTS_TIMEOUT=600
export PIP_DEFAULT_TIMEOUT=600
for piper_attempt in 1 2 3; do
  if POETRY_VIRTUALENVS_CREATE=true POETRY_VIRTUALENVS_IN_PROJECT=true \
    "$VENV_PY" -m poetry --directory "$SCRIPT_DIR/tools/piper-tts" install \
    --only main --no-root --no-interaction; then
    break
  fi
  if [ "$piper_attempt" = 3 ]; then
    echo "WARNING: Piper runtime installation failed after 3 attempts."
  else
    echo "WARNING: Piper install attempt $piper_attempt failed; retrying in 10 seconds..."
    sleep 10
  fi
done

if [ ! -f "poetry.lock" ]; then
    echo "Generating poetry.lock..."
    "$VENV_PY" -m poetry lock --no-interaction
fi


# The ml group pulls audio-separator -> librosa -> numba -> llvmlite. numba
# stopped publishing macOS x86_64 (Intel) wheels at 0.61, and llvmlite has no
# Intel-mac wheel either, so on an Intel Mac `--with ml` tries to build LLVM
# from source and fails. Intel Macs therefore get the FFmpeg-native DSP modes
# only (vhs_native, auto_ffmpeg_native, arnndn_speech); Apple Silicon and
# Linux/Windows keep the full AI stack.
ML_AVAILABLE=1
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ]; then
    ML_AVAILABLE=0
fi

if [ "$ML_AVAILABLE" = "1" ]; then
    echo "Installing runtime and ML dependencies..."
    "$VENV_PY" -m poetry install --no-root --with ml --without dev --no-interaction
else
    echo "NOTE: Intel Mac detected. numba/llvmlite ship no x86_64 macOS wheels,"
    echo "      so the AI (ml) dependency group is skipped. FFmpeg-native modes"
    echo "      (vhs_native, auto_ffmpeg_native, arnndn_speech) remain available;"
    echo "      AI separation and neural enhancement modes do not."
    "$VENV_PY" -m poetry install --no-root --without dev --no-interaction
fi

# Resemble-Enhance is a VCS install, so pip re-clones and rebuilds it on every
# run unless it is skipped explicitly. Prefer the installed copy: rebuild only
# when the recorded commit does not match the pinned ref (a version string alone
# does not change between revisions), or when FORCE_RESEMBLE_REBUILD=1.
RESEMBLE_REPO="https://github.com/daswer123/resemble-enhance-windows.git"
RESEMBLE_REF="270d8da4ea7c0efc960c52d605b75c0458b708d0"

installed_resemble_ref() {
    "$VENV_PY" - <<'PY' 2>/dev/null
import json
import sys
from importlib import metadata

try:
    raw = metadata.distribution("resemble-enhance").read_text("direct_url.json")
except metadata.PackageNotFoundError:
    sys.exit(1)

if not raw:
    sys.exit(1)
print(json.loads(raw).get("vcs_info", {}).get("commit_id", ""))
PY
}

if [ "$ML_AVAILABLE" = "1" ]; then
    echo "Checking Resemble-Enhance runtime package..."
    CURRENT_RESEMBLE_REF="$(installed_resemble_ref || true)"
    if [ "${FORCE_RESEMBLE_REBUILD:-0}" != "1" ] && [ "$CURRENT_RESEMBLE_REF" = "$RESEMBLE_REF" ]; then
        echo "Resemble-Enhance already installed at pinned commit ${RESEMBLE_REF:0:12}; skipping source build."
    else
        echo "Installing Resemble-Enhance runtime package (building from source)..."
        "$VENV_PY" -m pip install -v "git+${RESEMBLE_REPO}@${RESEMBLE_REF}" --no-deps --force-reinstall
    fi

    echo "Applying runtime patches..."
    "$VENV_PY" scripts/apply_patches.py
else
    echo "Skipping Resemble-Enhance install and runtime patches (AI stack not present on Intel Mac)."
fi

# Step 5: Create Folders & Launcher
echo -e "\nStep 5: Creating Project Directories & Launcher..."
mkdir -p input output temp_work

cat << 'EOF' > "$SCRIPT_DIR/start.sh"
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    for arg in "$@"; do
        case "$arg" in
            -h|--help)
                echo "AI Hybrid VHS Audio Restorer"
                echo
                echo "Usage: ./start.sh [PATH ...]   video files or folders; omit for interactive mode"
                echo "       ./start.sh --help"
                echo
                echo "Run ./install_dependencies.sh first to enable restoration."
                exit 0
                ;;
        esac
    done
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/.venv" >&2
    echo "Please run ./install_dependencies.sh first." >&2
    exit 1
fi
exec "$VENV_PY" "$SCRIPT_DIR/restore_audio_hybrid.py" "$@"
EOF
chmod +x "$SCRIPT_DIR/start.sh"

echo -e "\n=== Installation Complete! ==="
echo "1. Place your input video files into the 'input' folder."
echo "2. Run './start.sh' to start restoring audio."
