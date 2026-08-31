#!/usr/bin/env bash
# ==============================================================================
# Script: package_linux_deb_rpm.sh
# Description: Generates native .deb and .rpm Linux packages for
#              AI Hybrid VHS Audio Restorer.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-}"
OUTPUT_DIR="${2:-release-assets}"

if [ -z "$TAG" ]; then
    echo "Usage: $0 <tag> [output_dir]" >&2
    exit 1
fi

VERSION="${TAG#v}"
PKG_NAME="ai-hybrid-vhs-audio-restorer"
INSTALL_PREFIX="/opt/$PKG_NAME"

mkdir -p "$OUTPUT_DIR"
BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT

echo "=== Building Linux .deb and .rpm packages for $PKG_NAME v$VERSION ==="

APP_DIR="$BUILD_TMP/app"
mkdir -p "$APP_DIR"
git archive HEAD | tar -x -C "$APP_DIR"

# 1. Build .deb package structure
DEB_ROOT="$BUILD_TMP/deb"
mkdir -p "$DEB_ROOT$INSTALL_PREFIX"
mkdir -p "$DEB_ROOT/DEBIAN"
mkdir -p "$DEB_ROOT/usr/local/bin"

cp -a "$APP_DIR/." "$DEB_ROOT$INSTALL_PREFIX/"
chmod -R 755 "$DEB_ROOT$INSTALL_PREFIX"

# Wrapper binary in /usr/local/bin
cat << 'EOF' > "$DEB_ROOT/usr/local/bin/ai-hybrid-vhs-audio-restorer"
#!/usr/bin/env bash
exec /opt/ai-hybrid-vhs-audio-restorer/start.sh "$@"
EOF
chmod 755 "$DEB_ROOT/usr/local/bin/ai-hybrid-vhs-audio-restorer"

# Control file
cat << EOF > "$DEB_ROOT/DEBIAN/control"
Package: $PKG_NAME
Version: $VERSION
Section: sound
Priority: optional
Architecture: all
Maintainer: AI Hybrid VHS Audio Restorer Team
Depends: python3 (>= 3.12), ffmpeg
Description: High-performance AI Hybrid VHS Audio Restoration Pipeline
 AI-powered restoration engine specializing in VHS tape audio captures.
EOF

# Post-install script
cat << 'EOF' > "$DEB_ROOT/DEBIAN/postinst"
#!/usr/bin/env bash
set -e
echo "AI Hybrid VHS Audio Restorer installed to /opt/ai-hybrid-vhs-audio-restorer."
echo "Run 'ai-hybrid-vhs-audio-restorer' or run /opt/ai-hybrid-vhs-audio-restorer/install_dependencies.sh to initialize environment."
exit 0
EOF
chmod 755 "$DEB_ROOT/DEBIAN/postinst"

DEB_OUT="$REPO_ROOT/$OUTPUT_DIR/AI-Hybrid-VHS-Audio-Restorer-${TAG}-linux.deb"
dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUT"
echo "Generated Debian package: $DEB_OUT"

# 2. Build .rpm package if alien / rpmbuild / fpm is available
if command -v alien >/dev/null 2>&1; then
    echo "Generating RPM from Debian package via alien..."
    (
        cd "$BUILD_TMP"
        alien --to-rpm --scripts "$DEB_OUT"
        RPM_GENERATED="$(find . -maxdepth 1 -name '*.rpm' -print -quit)"
        if [ -n "$RPM_GENERATED" ]; then
            mv "$RPM_GENERATED" "$REPO_ROOT/$OUTPUT_DIR/AI-Hybrid-VHS-Audio-Restorer-${TAG}-linux.rpm"
            echo "Generated RPM package: $REPO_ROOT/$OUTPUT_DIR/AI-Hybrid-VHS-Audio-Restorer-${TAG}-linux.rpm"
        fi
    )
elif command -v rpmbuild >/dev/null 2>&1; then
    echo "Generating RPM using rpmbuild..."
    RPM_TOP="$BUILD_TMP/rpmbuild"
    mkdir -p "$RPM_TOP"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
    SPEC_FILE="$RPM_TOP/SPECS/$PKG_NAME.spec"
    cat << EOF > "$SPEC_FILE"
Name:           $PKG_NAME
Version:        $VERSION
Release:        1%{?dist}
Summary:        High-performance AI Hybrid VHS Audio Restoration Pipeline
License:        MIT
BuildArch:      noarch
Requires:       python3 >= 3.12, ffmpeg

%description
High-performance AI Hybrid VHS Audio Restoration Pipeline.

%install
mkdir -p %{buildroot}$INSTALL_PREFIX
mkdir -p %{buildroot}/usr/local/bin
cp -a $APP_DIR/. %{buildroot}$INSTALL_PREFIX/
cat << 'WRAPPER' > %{buildroot}/usr/local/bin/ai-hybrid-vhs-audio-restorer
#!/usr/bin/env bash
exec /opt/ai-hybrid-vhs-audio-restorer/start.sh "\$@"
WRAPPER
chmod 755 %{buildroot}/usr/local/bin/ai-hybrid-vhs-audio-restorer

%files
$INSTALL_PREFIX
/usr/local/bin/ai-hybrid-vhs-audio-restorer

%post
echo "AI Hybrid VHS Audio Restorer installed to /opt/ai-hybrid-vhs-audio-restorer."
EOF
    rpmbuild --define "_topdir $RPM_TOP" -bb "$SPEC_FILE"
    RPM_GENERATED="$(find "$RPM_TOP/RPMS" -name '*.rpm' -print -quit)"
    if [ -n "$RPM_GENERATED" ]; then
        mv "$RPM_GENERATED" "$REPO_ROOT/$OUTPUT_DIR/AI-Hybrid-VHS-Audio-Restorer-${TAG}-linux.rpm"
        echo "Generated RPM package: $REPO_ROOT/$OUTPUT_DIR/AI-Hybrid-VHS-Audio-Restorer-${TAG}-linux.rpm"
    fi
else
    echo "Neither alien nor rpmbuild available; skipping .rpm generation."
fi
