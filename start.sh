#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ] && [ ! -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
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
    echo "Virtual environment not found at $SCRIPT_DIR/.venv."
    echo "Automatically running ./install_dependencies.sh..."
    if [ -f "$SCRIPT_DIR/install_dependencies.sh" ]; then
        bash "$SCRIPT_DIR/install_dependencies.sh"
    fi
fi

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    echo "ERROR: Virtual environment setup failed or python executable not found at $SCRIPT_DIR/.venv" >&2
    exit 1
fi
exec "$VENV_PY" "$SCRIPT_DIR/restore_audio_hybrid.py" "$@"
