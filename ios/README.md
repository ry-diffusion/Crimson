# Crimson on iOS (libcrimson)

`libcrimson.dylib` is this fork of metal2vulkan as a Metal device for iOS 26 on the Poco F1 (Adreno 630), drawing through
Chlorophyll's Turnip (`libchlorophyll.dylib`) and the Bergamota kernel. Chlorophyll's inject loads it from
`/usr/local/lib/bgmtl/libcrimson.dylib` for backboardd (`/private/var/tmp/bgmtl-bb.m2v`) and per app
(`/private/var/tmp/bgmtl.m2v.<bundle id>`).

- `metal2/`: ios-guacamole's NativeAdrenoMetal frontend and bridge (MIT, see `metal2/bridge/LICENSE`) with Chlorophyll's changes.
- `crimson.mm`: the device entry point (`m2v_create_device`) and the log (`/private/var/tmp/m2v.log`).
- `build.sh`: `BERGAMOTA=<ios-beryllium checkout> ios/build.sh [OUTDIR] [probe]`, output in `target/ios/out/`.
- `test/crimson_probe.m`: the device outside any system process (benches, pipeline switching).
- `tools/`: phone-side test drivers (fps runs, screenshots, poster GPU timing, dump conversion).

The runtime switches are files on the phone (`/private/var/tmp/m2v-*`); their names are unchanged from the libm2v days.
