#!/usr/bin/env bash
# Cross-Platform Docker Test Suite Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-ubuntu}"

echo "=== Cross-Platform Docker Test Runner ==="
echo "Target: $TARGET"

case "$TARGET" in
    ubuntu)
        echo "Building and running clean Ubuntu container test..."
        docker build -f docker/Dockerfile.ubuntu -t vhs-restorer-ubuntu-test .
        docker run --rm vhs-restorer-ubuntu-test
        echo "Ubuntu container test completed successfully!"
        ;;
    macos|windows)
        echo "Target '$TARGET' has no automated test flow." >&2
        echo "docker/docker-compose.test.yml declares the ${TARGET}_test service, but it boots an" >&2
        echo "interactive KVM virtual machine and does not exit with a pass/fail status." >&2
        echo "Start it manually if needed: docker compose -f docker/docker-compose.test.yml up ${TARGET}_test" >&2
        exit 1
        ;;
    *)
        echo "Unknown target: $TARGET. Usage: $0 [ubuntu]"
        exit 1
        ;;
esac
