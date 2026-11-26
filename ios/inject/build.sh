#!/bin/bash
# SPDX-License-Identifier: MIT
# crimson-inject.dylib (ios/inject/crimson_inject.m) and the libdrm stand-in it links, for /usr/local/lib/crimson on the phone:
#   BERGAMOTA=<ios-beryllium checkout> ios/inject/build.sh [OUTDIR]          (default target/ios/out)
# Outputs: crimson-inject.dylib, libdrm.2.dylib (Bergamota's gpu/freedreno-sweet). Turnip (libturnip.dylib) is built by
# ios/turnip/build-turnip.sh and loaded by libcrimson, not by the inject.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
BG="${BERGAMOTA:?set BERGAMOTA to the ios-beryllium checkout}"
OUT="${1:-$R/target/ios/out}"
mkdir -p "$OUT"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CC="xcrun --sdk iphoneos clang -arch arm64e -miphoneos-version-min=18.0 -isysroot $SDK"
OBJC="-O1 -g -fno-objc-arc -Wall -Wno-deprecated-declarations -Wno-unguarded-availability-new"
"$BG/gpu/freedreno-sweet/build.sh" "$OUT/drm" >/dev/null
cp -f "$OUT/drm/libdrm.2.arm64e.dylib" "$OUT/libdrm.2.dylib"
$CC $OBJC -I"$BG/xnu/gpu-abi" -c "$R/ios/inject/crimson_inject.m" -o "$OUT/crimson_inject.o"
$CC -dynamiclib -o "$OUT/crimson-inject.dylib" "$OUT/crimson_inject.o" "$OUT/libdrm.2.dylib" \
    -framework Foundation -framework Metal -framework IOSurface -install_name /usr/local/lib/crimson/crimson-inject.dylib -Wl,-rpath,@loader_path
for f in crimson-inject.dylib libdrm.2.dylib; do
  python3 "$BG/tools/renegade/pacstrip.py" --sign "$OUT/$f" >/dev/null
done
rm -rf "$OUT/drm" "$OUT/crimson_inject.o"
ls -l "$OUT/crimson-inject.dylib" "$OUT/libdrm.2.dylib"
