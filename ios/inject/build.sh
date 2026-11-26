#!/bin/bash
# SPDX-License-Identifier: MIT
# crimson-inject.dylib (ios/inject/crimson_inject.m) with the Turnip build it links, for /usr/local/lib/crimson on the phone:
#   CHLOROPHYLL=<chlorophyll-turnip checkout> GPU_METAL_WORK=<work dir with mesa-turnip26/build-ios> BERGAMOTA=<ios-beryllium checkout> \
#     ios/inject/build.sh [OUTDIR]          (default target/ios/out)
# Outputs: crimson-inject.dylib, libturnip.dylib (Chlorophyll's Turnip build, libchlorophyll.dylib, under its Crimson name: install name
# @rpath/libturnip.dylib), libdrm.2.dylib (Bergamota's libdrm stand-in). Until Chlorophyll's own Metal device is retired the inject still
# links its bgmtl.o (src/bgmtl.m) as the fallback device.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
CHL="${CHLOROPHYLL:?set CHLOROPHYLL to the chlorophyll-turnip checkout}"
WORK="${GPU_METAL_WORK:?set GPU_METAL_WORK}"
BG="${BERGAMOTA:?set BERGAMOTA to the ios-beryllium checkout}"
OUT="${1:-$R/target/ios/out}"
D="${TURNIP_TREE:-$WORK/mesa-turnip26}"
mkdir -p "$OUT"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CC="xcrun --sdk iphoneos clang -arch arm64e -miphoneos-version-min=18.0 -isysroot $SDK"
OBJC="-O1 -g -fno-objc-arc -Wall -Wno-deprecated-declarations -Wno-unguarded-availability-new"
cp -f "$D/build-ios/src/freedreno/vulkan/libchlorophyll.dylib" "$OUT/libturnip.dylib"
xcrun install_name_tool -id @rpath/libturnip.dylib "$OUT/libturnip.dylib"
xcrun strip -x "$OUT/libturnip.dylib"
"$BG/gpu/freedreno-sweet/build.sh" "$OUT/drm" >/dev/null
cp -f "$OUT/drm/libdrm.2.arm64e.dylib" "$OUT/libdrm.2.dylib"
$CC $OBJC -I"$CHL/air" -c "$CHL/src/bgmtl.m" -o "$OUT/bgmtl.o"
$CC $OBJC -c "$R/ios/inject/crimson_inject.m" -o "$OUT/crimson_inject.o"
$CC -dynamiclib -o "$OUT/crimson-inject.dylib" "$OUT/bgmtl.o" "$OUT/crimson_inject.o" "$OUT/libturnip.dylib" "$OUT/libdrm.2.dylib" \
    -framework Foundation -framework Metal -framework IOSurface -install_name /usr/local/lib/crimson/crimson-inject.dylib -Wl,-rpath,@loader_path
for f in crimson-inject.dylib libturnip.dylib libdrm.2.dylib; do
  python3 "$BG/tools/renegade/pacstrip.py" --sign "$OUT/$f" >/dev/null
done
rm -rf "$OUT/drm" "$OUT/bgmtl.o" "$OUT/crimson_inject.o"
ls -l "$OUT/crimson-inject.dylib" "$OUT/libturnip.dylib" "$OUT/libdrm.2.dylib"
