#!/usr/bin/env bash
# ==============================================================================
# Script: package_macos_pkg.sh
# Description: Generates native macOS .pkg installer for
#              AI Hybrid VHS Audio Restorer.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-}"
OUTPUT_DIR="${2:-release-assets}"
ARCH="${3:-arm64}"

if [ -z "$TAG" ]; then
    echo "Usage: $0 <tag> [output_dir] [arch]" >&2
    exit 1
fi

VERSION="${TAG#v}"
PKG_ID="com.ventura8.ai-hybrid-vhs-audio-restorer"
INSTALL_PREFIX="/Applications/AI-Hybrid-VHS-Audio-Restorer"

mkdir -p "$OUTPUT_DIR"
BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT

echo "=== Building macOS .app bundle & .pkg installer for $PKG_ID v$VERSION ($ARCH) ==="

APP_BUNDLE="$BUILD_TMP/payload/Applications/AI-Hybrid-VHS-Audio-Restorer.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Copy full application payload into Resources
git archive HEAD | tar -x -C "$RESOURCES_DIR"
chmod -R 755 "$RESOURCES_DIR"

# Create standard macOS Info.plist
cat << EOF > "$CONTENTS_DIR/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>AI Hybrid VHS Audio Restorer</string>
    <key>CFBundleExecutable</key>
    <string>ai-hybrid-vhs-audio-restorer</string>
    <key>CFBundleIdentifier</key>
    <string>$PKG_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>AI Hybrid VHS Audio Restorer</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
</dict>
</plist>
EOF

# Create executable entry point in Contents/MacOS
cat << 'EOF' > "$MACOS_DIR/ai-hybrid-vhs-audio-restorer"
#!/usr/bin/env bash
BUNDLE_RESOURCES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../Resources" && pwd)"
exec "$BUNDLE_RESOURCES/start.sh" "$@"
EOF
chmod 755 "$MACOS_DIR/ai-hybrid-vhs-audio-restorer"

SCRIPTS_DIR="$BUILD_TMP/scripts"
mkdir -p "$SCRIPTS_DIR"

cat << 'EOF' > "$SCRIPTS_DIR/postinstall"
#!/usr/bin/env bash
set -e
APP_PATH="/Applications/AI-Hybrid-VHS-Audio-Restorer.app"
chmod -R 755 "$APP_PATH"
mkdir -p /usr/local/bin
cat << 'LAUNCHER' > /usr/local/bin/ai-hybrid-vhs-audio-restorer
#!/usr/bin/env bash
exec /Applications/AI-Hybrid-VHS-Audio-Restorer.app/Contents/MacOS/ai-hybrid-vhs-audio-restorer "$@"
LAUNCHER
chmod 755 /usr/local/bin/ai-hybrid-vhs-audio-restorer
exit 0
EOF
chmod 755 "$SCRIPTS_DIR/postinstall"

PKG_OUT="$REPO_ROOT/$OUTPUT_DIR/AI-Hybrid-VHS-Audio-Restorer-${TAG}-macos-${ARCH}.pkg"

if command -v pkgbuild >/dev/null 2>&1; then
    pkgbuild \
        --root "$BUILD_TMP/payload" \
        --identifier "$PKG_ID" \
        --version "$VERSION" \
        --install-location "/" \
        --scripts "$SCRIPTS_DIR" \
        "$PKG_OUT"
    echo "Generated macOS installer package: $PKG_OUT"
else
    echo "pkgbuild command not found (not running on macOS). Skipping .pkg build."
fi
