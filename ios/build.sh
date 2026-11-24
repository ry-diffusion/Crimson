#!/bin/bash
# SPDX-License-Identifier: MIT
# libcrimson.dylib: Crimson (this fork of metal2vulkan) as a Metal device on Chlorophyll's Turnip, loaded by Chlorophyll's inject
# (chlorophyll-turnip inject/bgmtl_inject.m: backboardd with /private/var/tmp/bgmtl-bb.m2v, apps with bgmtl.m2v.<bundle id>).
#   BERGAMOTA=<ios-beryllium checkout> ios/build.sh [OUTDIR] [probe]
# Sources, all in this repository: crates/ (the runtime), ios/metal2/ (ios-guacamole's NativeAdrenoMetal frontend and bridge,
# MIT, with Chlorophyll's changes), ios/crimson.mm (the device entry point and log). The bridge is built for arm64e with
# -Zbuild-std (RUSTC_BOOTSTRAP: rust 1.98 has the target, not its prebuilt std); everything PAC-stripped (Kryo 385), with
# Bergamota's tools/renegade/pacstrip.py. "probe" also builds ios/test/crimson_probe.m (OUTDIR/crimson_probe).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
BG="${BERGAMOTA:?set BERGAMOTA to the ios-beryllium checkout (tools/renegade/pacstrip.py)}"
OUT="${1:-$R/target/ios/out}"
B="$R/target/ios/bridge"
M="$R/ios/metal2"
export PATH=/opt/homebrew/opt/rustup/bin:$PATH
mkdir -p "$B" "$OUT"
LLVM_AS=$(ls /opt/homebrew/opt/llvm*/bin/llvm-as | head -1)
"$LLVM_AS" -o "$B/runtime_examples.air" "$R/crates/metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll"
"$LLVM_AS" -o "$B/function_constant.air" "$R/crates/metal2vulkan-validation/fixtures/public/kernel_function_constant_gate_fold.ll"
cat > "$B/Cargo.toml" <<EOT
[package]
name = "native-metal-bridge"
version = "0.1.0"
edition = "2024"
rust-version = "1.98"
[workspace]
[lib]
crate-type = ["staticlib"]
path = "$M/bridge/NativeMetalBridge.rs"
[dependencies]
metal2vulkan = { path = "$R/crates/metal2vulkan", features = ["ash"] }
metal2vulkan-runtime = { path = "$R/crates/metal2vulkan-runtime" }
libc = "0.2"
sha2 = { version = "0.10", features = ["asm"] }
[profile.release]
panic = "unwind"
debug = 1
lto = "fat"
codegen-units = 1
EOT
triple=arm64e-apple-ios
RUSTC_BOOTSTRAP=1 NMB_FIXTURE_AIR=$B/runtime_examples.air NMB_FC_AIR=$B/function_constant.air IPHONEOS_DEPLOYMENT_TARGET=18.0 \
  cargo +1.98.0 build --release -Zbuild-std=std,panic_unwind --manifest-path "$B/Cargo.toml" --target $triple \
  --target-dir "$B/target" > "$B/build-arm64e.log" 2>&1 || { grep -E "^error" -A8 "$B/build-arm64e.log" | head -60; exit 1; }
BRIDGE="$B/target/$triple/release/libnative_metal_bridge.a"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CC="xcrun --sdk iphoneos clang -arch arm64e -miphoneos-version-min=18.0 -isysroot $SDK"
$CC -x objective-c++ -std=c++17 -fobjc-arc -O2 -g -Wall -Wno-deprecated-declarations -Wno-unused-function \
  -Wno-unguarded-availability-new -I"$M" -dynamiclib -o "$OUT/libcrimson.dylib" \
  "$R/ios/crimson.mm" "$M/NativeBridgeMetalFrontend.mm" -x none "$BRIDGE" -Wl,-dead_strip -lc++ \
  -framework Foundation -framework Metal -framework QuartzCore -framework IOSurface -framework IOKit -framework CoreFoundation \
  -install_name /usr/local/lib/bgmtl/libcrimson.dylib -Wl,-rpath,@loader_path
xcrun dsymutil "$OUT/libcrimson.dylib" -o "$OUT/libcrimson.dylib.dSYM" 2>/dev/null || true
xcrun strip -x "$OUT/libcrimson.dylib"
python3 "$BG/tools/renegade/pacstrip.py" --sign "$OUT/libcrimson.dylib" >/dev/null
ls -l "$OUT/libcrimson.dylib"
if [ "${2:-}" = probe ]; then
  $CC -fobjc-arc -O1 -Wno-deprecated-declarations -o "$OUT/crimson_probe" "$R/ios/test/crimson_probe.m" \
    -framework Foundation -framework Metal -framework IOSurface
  codesign -f -s - --entitlements "$R/ios/test/entitlements.plist" "$OUT/crimson_probe" 2>/dev/null
  python3 "$BG/tools/renegade/pacstrip.py" --sign "$OUT/crimson_probe" >/dev/null
  ls -l "$OUT/crimson_probe"
fi
