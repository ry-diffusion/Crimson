# Crimson on iOS (libcrimson)

`libcrimson.dylib` is this fork of metal2vulkan as a Metal device for iOS 26 on the Poco F1 (Adreno 630), drawing through
Turnip (`libturnip.dylib`, Chlorophyll's Mesa build) and the Bergamota kernel. The inject (`inject/`, `crimson-inject.dylib`) loads it
from `/usr/local/lib/crimson/libcrimson.dylib` for backboardd (`/private/var/tmp/crimson-bb.on`) and per app
(`/private/var/tmp/crimson.app.<bundle id>`).

- `metal2/`: ios-guacamole's NativeAdrenoMetal frontend and bridge (MIT, see `metal2/bridge/LICENSE`) with Chlorophyll's changes.
- `crimson.mm`: the device entry point (`m2v_create_device`), the log (`/private/var/tmp/crimson.log`) and `crimson_switch()`.
- `inject/`: `crimson_inject.m` (moved from Chlorophyll's `inject/bgmtl_inject.m`) and `build.sh`, which builds it with `libturnip.dylib`
  and `libdrm.2.dylib` (`CHLOROPHYLL=<chlorophyll-turnip> GPU_METAL_WORK=<work dir> BERGAMOTA=<ios-beryllium> ios/inject/build.sh`).
- `build.sh`: `BERGAMOTA=<ios-beryllium checkout> ios/build.sh [OUTDIR] [probe]`, output in `target/ios/out/`.
- `test/crimson_probe.m`: the device outside any system process (benches, pipeline switching).
- `tools/`: phone-side test drivers (fps runs, screenshots, poster GPU timing, dump conversion).

Phone layout (since 2026-11-26): libraries in `/usr/local/lib/crimson/`; `/usr/local/lib/bgmtl/` holds links under the old names,
because re-signed system binaries (backboardd, SpringBoard, poster and widget extensions) load `/usr/local/lib/bgmtl/bgmtl-inject.dylib`
by path. Switches are files in `/private/var/tmp`: `crimson-*` (Crimson), `crimson-bb.*` and `crimson.*` (the inject). Every reader
still accepts the old `m2v-*`, `bgmtl-bb.*` and `bgmtl.*` names while only those exist. Environment variables (`CHL_M2V_*`, `M2V_*`)
keep their names.
