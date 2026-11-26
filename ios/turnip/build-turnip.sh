#!/bin/bash
# SPDX-License-Identifier: MIT
# Turnip (Mesa's Vulkan driver for Adreno) for Crimson on iOS arm64e: stock Mesa 26.2.5 plus patches/mesa-26.2.5-turnip-ios.patch (iOS build
# and Mach-O fixes, the msm kernel backend over Bergamota's libdrm stand-in gpu/freedreno-sweet, ir3 fixes, the program-state pointer).
# Moved from Chlorophyll's turnip/build-turnip.sh on 2026-11-26 without the AIR frontend and Chlorophyll's runtime: libcrimson only uses the
# Vulkan ICD (vk_icdGetInstanceProcAddr).
#   GPU_METAL_WORK=<work dir> BERGAMOTA=<ios-beryllium checkout> ios/turnip/build-turnip.sh [OUTDIR]     (default target/ios/out)
# Needs: TURNIP_MESA (pristine 26.2.5 checkout, default $WORK/mesa-src-26.2.5, cloned when missing), GPU_EAGL_LIBDRM (libdrm 2.4.124 source,
# default $WORK/libdrm-src), GPU_EAGL_VENV (mako, pyyaml, meson; default $WORK/venv), brew bison and glslang (glslangValidator).
# Tree: $WORK/mesa-crimson26 (TURNIP_TREE). Output: OUTDIR/libturnip.dylib (install name @rpath/libturnip.dylib, PAC-stripped and signed).
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
BG="${BERGAMOTA:?set BERGAMOTA to the ios-beryllium checkout}"
WORK="${GPU_METAL_WORK:?set GPU_METAL_WORK}"
OUT="${1:-$R/target/ios/out}"
MESA="${TURNIP_MESA:-$WORK/mesa-src-26.2.5}"
MESA_TAG=mesa-26.2.5
LIBDRM="${GPU_EAGL_LIBDRM:-$WORK/libdrm-src}"
VENV="${GPU_EAGL_VENV:-$WORK/venv}"
command -v glslangValidator >/dev/null || { echo "glslangValidator missing: brew install glslang" >&2; exit 1; }
export PATH="$(brew --prefix bison)/bin:$VENV/bin:$PATH"
D="${TURNIP_TREE:-$WORK/mesa-crimson26}"
[ -d "$MESA" ] || git clone -q --depth 1 --branch "$MESA_TAG" https://gitlab.freedesktop.org/mesa/mesa.git "$MESA"
mkdir -p "$D"
cd "$D"
if [ ! -d mesa ]; then
  git clone -q --shared "$MESA" mesa
  git -C mesa apply "$R/ios/turnip/patches/mesa-26.2.5-turnip-ios.patch"
fi
"$BG/gpu/freedreno-sweet/build.sh" "$D/drm-ios" >/dev/null
mkdir -p pc-ios drm-ios-link
cp drm-ios/libdrm.2.arm64e.dylib drm-ios-link/libdrm.2.dylib; ln -sf libdrm.2.dylib drm-ios-link/libdrm.dylib
cat > pc-ios/libdrm.pc <<EOP
Name: libdrm
Description: Bergamota libdrm stand-in (gpu/freedreno-sweet)
Version: 2.4.124
Cflags: -I$LIBDRM/include/drm -I$LIBDRM
Libs: -L$D/drm-ios-link -ldrm
EOP
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
# pkg_config_libdir (not only _path): Homebrew's host SPIRV-Tools would otherwise be found and linked into the arm64e dylib
cat > ios.cross <<EOT
[binaries]
c = ['xcrun','--sdk','iphoneos','clang','-arch','arm64e']
cpp = ['xcrun','--sdk','iphoneos','clang++','-arch','arm64e']
objc = ['xcrun','--sdk','iphoneos','clang','-arch','arm64e']
ar = ['xcrun','--sdk','iphoneos','ar']
strip = ['xcrun','--sdk','iphoneos','strip']
ranlib = ['xcrun','--sdk','iphoneos','ranlib']
bison = '$(brew --prefix bison)/bin/bison'
pkg-config = '$(command -v pkg-config)'

[built-in options]
pkg_config_path = ['$D/pc-ios']
c_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']
cpp_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']
c_link_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']
cpp_link_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']

[host_machine]
system = 'darwin'
subsystem = 'ios'
cpu_family = 'aarch64'
cpu = 'arm64e'
endian = 'little'

[properties]
pkg_config_libdir = ['$D/pc-ios']
needs_exe_wrapper = true
EOT
[ -d build-ios ] || meson setup build-ios mesa --cross-file ios.cross --buildtype=release -Dgallium-drivers= -Dvulkan-drivers=freedreno \
  -Dfreedreno-kmds=msm -Dvulkan-layers= -Dplatforms= -Dllvm=disabled -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Dglx=disabled \
  -Degl=disabled -Dgbm=disabled -Dtools= -Dvalgrind=disabled -Dlibunwind=disabled -Dzstd=disabled -Dxmlconfig=disabled -Dexpat=disabled \
  -Dglvnd=disabled -Dlmsensors=disabled -Dbuild-tests=false -Dshader-cache=enabled -Dvideo-codecs= -Dperfetto=false -Dspirv-tools=disabled \
  -Ddisplay-info=disabled -Dprefix=/ > setup-ios.log
ninja -j"${JOBS:-8}" -C build-ios src/freedreno/vulkan/libvulkan_freedreno.dylib
mkdir -p "$OUT"
cp -f build-ios/src/freedreno/vulkan/libvulkan_freedreno.dylib "$OUT/libturnip.dylib"
xcrun install_name_tool -id @rpath/libturnip.dylib "$OUT/libturnip.dylib"
xcrun strip -x "$OUT/libturnip.dylib"
python3 "$BG/tools/renegade/pacstrip.py" --sign "$OUT/libturnip.dylib" >/dev/null
ls -l "$OUT/libturnip.dylib"
