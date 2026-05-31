#!/usr/bin/env bash
# Rebuild linux-zen-p2p from the latest upstream Arch linux-zen package +
# our 3-line cosmetic customization. Use this when you want to bump to a
# newer kernel version.
#
# IMPORTANT: the `-p2p` suffix is HISTORICAL — what actually gives us NVLink
# P2P (NV4 topology) is the physical NVLink bridge + the kernel boot args
# (pcie_acs_override + iommu=pt) in /etc/kernel/cmdline, NOT this package's
# `CONFIG_HSA_AMD_P2P=y` flag (which is an AMD-HSA driver setting we don't
# actually use — vestigial from earlier debugging). The reason we still
# build this package is to keep the `linux-zen` family of stability fixes
# under sustained TP=2/256K load, with a pinned pkgname that pacman won't
# auto-upgrade. Stock `linux-zen` would serve the same role; the rename
# just gives us a separate package identity to manage.
#
# What this does:
#   1. `git pull --rebase` the upstream Arch linux-zen PKGBUILD repo
#   2. Reapply our 3-line diff:
#        - PKGBUILD pkgbase: linux-zen -> linux-zen-p2p
#        - PKGBUILD pkgdesc: 'Linux ZEN' -> 'Linux ZEN with HSA_AMD_P2P enabled'
#        - config: append CONFIG_HSA_AMD_P2P=y (cosmetic; we don't run amdkfd)
#   3. `updpkgsums` to regenerate hashes for the new kernel tarball
#   4. `makepkg -c` to build (no sudo needed for build)
#   5. Print the `pacman -U ...` install line — does NOT auto-install unless
#      `--install` is passed; install needs a reboot to take effect.
#   6. Post-install verification commands
#
# What this does NOT touch:
#   - /etc/kernel/cmdline (boot args). The P2P-enabling args
#     `pcie_acs_override=downstream,multifunction iommu=pt amd_iommu=on
#      pcie_ports=native pcie_ecrc=on` live there and survive kernel upgrades.
#   - /etc/pacman.conf IgnorePkg pinning. Already in place; this script's
#     install step uses `pacman -U <file>` which bypasses IgnorePkg.
#   - NVIDIA driver. nvidia-open-dkms rebuilds against the new headers
#     automatically when pacman installs the headers package.
#
# Build dir is hard-coded to ~/kernel-build/linux-zen/ (the existing working
# copy from the 2025-12 build). To start fresh on a different host:
#   git clone https://gitlab.archlinux.org/archlinux/packaging/packages/linux-zen.git \
#       ~/kernel-build/linux-zen
#
# Usage:
#   ./rebuild_linux_zen_p2p.sh             # build only, do not install
#   ./rebuild_linux_zen_p2p.sh --install   # build AND `pacman -U` (still prompts via sudo)
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$HOME/kernel-build/linux-zen}"
INSTALL=0
[[ "${1:-}" = "--install" ]] && INSTALL=1

if [[ ! -d "$BUILD_DIR/.git" ]]; then
    echo "ERROR: $BUILD_DIR is not a git checkout. Clone first:" >&2
    echo "    git clone https://gitlab.archlinux.org/archlinux/packaging/packages/linux-zen.git $BUILD_DIR" >&2
    exit 1
fi

cd "$BUILD_DIR"
echo "== $(pwd) =="

# Save current local diffs so we don't lose them if anything goes wrong.
if [[ -n "$(git status --porcelain)" ]]; then
    backup="/tmp/linux-zen-p2p-local-diff-$(date +%Y%m%d-%H%M%S).patch"
    git diff > "$backup"
    echo "Saved current local-diff to $backup ($(wc -l <"$backup") lines)"
fi

# Stash, sync, re-apply
git stash push -u -m "rebuild_linux_zen_p2p.sh autosave" || true
echo "[1/5] git fetch + rebase ..."
git fetch origin
git pull --rebase origin main
git stash pop 2>/dev/null || true   # may have nothing if no local edits

echo "[2/5] Re-apply 3-line P2P customization ..."

# (a) PKGBUILD pkgbase rename — leave alone if already linux-zen-p2p
if grep -q '^pkgbase=linux-zen$' PKGBUILD; then
    sed -i 's/^pkgbase=linux-zen$/pkgbase=linux-zen-p2p/' PKGBUILD
    echo "  PKGBUILD: pkgbase -> linux-zen-p2p"
fi

# (b) PKGBUILD pkgdesc
if grep -q "^pkgdesc='Linux ZEN'$" PKGBUILD; then
    sed -i "s/^pkgdesc='Linux ZEN'$/pkgdesc='Linux ZEN with HSA_AMD_P2P enabled'/" PKGBUILD
    echo "  PKGBUILD: pkgdesc updated"
fi

# (c) config: enable HSA_AMD_P2P (idempotent — only append if not already set)
if ! grep -q '^CONFIG_HSA_AMD_P2P=y$' config; then
    # Replace `# CONFIG_HSA_AMD_P2P is not set` if present, else append.
    if grep -q '^# CONFIG_HSA_AMD_P2P is not set' config; then
        sed -i 's/^# CONFIG_HSA_AMD_P2P is not set$/CONFIG_HSA_AMD_P2P=y/' config
        echo "  config: HSA_AMD_P2P toggled from 'not set' to y"
    else
        echo 'CONFIG_HSA_AMD_P2P=y' >> config
        echo "  config: appended CONFIG_HSA_AMD_P2P=y"
    fi
fi

# Sanity-check the customization landed
grep -q '^pkgbase=linux-zen-p2p$' PKGBUILD       || { echo "FAIL: pkgbase not linux-zen-p2p" >&2; exit 1; }
grep -q '^CONFIG_HSA_AMD_P2P=y$'    config       || { echo "FAIL: CONFIG_HSA_AMD_P2P=y missing"  >&2; exit 1; }
grep -q '^CONFIG_PCI_P2PDMA=y$'     config       || { echo "WARN: CONFIG_PCI_P2PDMA=y not set — upstream may have changed default" >&2; }

echo "[3/5] updpkgsums (regenerate sha256/b2 sums for new kernel tarball) ..."
updpkgsums

echo "[4/5] makepkg -c (build, takes ~30-60 min on 12C/24T) ..."
makepkg -c --noconfirm

# Show outputs
echo "[5/5] Built packages:"
ls -lh linux-zen-p2p-*.pkg.tar.zst linux-zen-p2p-headers-*.pkg.tar.zst 2>/dev/null || {
    echo "ERROR: expected .pkg.tar.zst not found" >&2
    exit 1
}

INSTALL_CMD="sudo pacman -U linux-zen-p2p-*.pkg.tar.zst linux-zen-p2p-headers-*.pkg.tar.zst"
echo ""
echo "============================================================"
echo "Build complete. To install + reboot into the new kernel:"
echo "    cd $BUILD_DIR && $INSTALL_CMD"
echo ""
echo "Post-install verification (run after reboot):"
echo "    uname -r                          # should end in '-zen-p2p'"
echo "    nvidia-smi topo -m | head -3      # GPU0 <-> GPU1 must show NV4"
echo "    pacman -Q linux-zen-p2p           # version matches build"
echo "============================================================"

if [[ "$INSTALL" -eq 1 ]]; then
    echo ""
    echo "--install passed: running $INSTALL_CMD"
    cd "$BUILD_DIR"
    eval "$INSTALL_CMD"
    echo ""
    echo "Installed. Reboot when ready; verify with the post-install commands above."
fi
