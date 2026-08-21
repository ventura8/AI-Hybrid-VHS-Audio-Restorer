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
