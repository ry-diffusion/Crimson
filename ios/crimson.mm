/*
 * libcrimson.dylib (Crimson, a fork of metal2vulkan): a Metal device from metal2vulkan (steelbrain/metal2vulkan, GPL-2.0-or-later OR LGPL-3.0-or-later) through
 * ios-guacamole's NativeAdrenoMetal frontend and bridge (MIT, UniXNU via ios-flame), on Chlorophyll's own Turnip
 * (libchlorophyll.dylib's ICD entry points, our kernel through the libdrm stand-in). A trial beside Chlorophyll's own
 * runtime: the inject (ios/inject/crimson_inject.m) asks m2v_create_device() for a process whose
 * /private/var/tmp/bgmtl.m2v.<bundle id> exists, and registers the answer with MTLAddDevice the same way.
 * Log: /private/var/tmp/crimson.log (or the old m2v.log; when it exists), lines "M2V[pid name]: ...".
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <IOSurface/IOSurfaceRef.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "NativeBridgeMetalFrontend.h"

/* Phone paths (2026-11-26 rename): /usr/local/lib/crimson/ and /private/var/tmp/crimson-*; the old bgmtl/m2v names are still
 * read while they exist (and /usr/local/lib/bgmtl/ keeps links for the system binaries that load the inject by that path). */
static const char *crimson_pick(const char *now, const char *old)
{
	return access(now, F_OK) == 0 || access(old, F_OK) != 0 ? now : old;
}
#define M2V_TURNIP crimson_pick("/usr/local/lib/crimson/libturnip.dylib", "/usr/local/lib/bgmtl/libchlorophyll.dylib")
#define M2V_LOG crimson_pick("/private/var/tmp/crimson.log", "/private/var/tmp/m2v.log")

/* A switch file: /private/var/tmp/crimson-NAME, or the old /private/var/tmp/m2v-NAME while only that one exists. */
extern "C" const char *crimson_switch(const char *name)
{
	static __thread char buf[8][160];
	static __thread unsigned k;
	char *now = buf[k++ % 8], old[160];
	snprintf(now, sizeof(buf[0]), "/private/var/tmp/crimson-%s", name);
	snprintf(old, sizeof(old), "/private/var/tmp/m2v-%s", name);
	if (access(now, F_OK) != 0 && access(old, F_OK) == 0)
		strcpy(now, old);
	return now;
}

extern "C" void sm2_logs(const char *s)
{
	static int fd = -2;
	if (fd == -2)
		fd = open(M2V_LOG, O_WRONLY | O_APPEND);
	if (fd < 0)
		return;
	char buf[700];
	int n = snprintf(buf, sizeof(buf), "M2V[%d %s]: %s\n", getpid(), getprogname(), s);
	write(fd, buf, n < (int)sizeof(buf) ? (size_t)n : sizeof(buf) - 1);
	/* the log is shared by every process for the whole boot: at 50 MB, with old backups, it filled the data partition and
	 * mobile processes could no longer create files (SpringBoard died at unlock, 2026-11-25). Past 32 MB it starts over. */
	static unsigned lines;
	struct stat st;
	if (++lines % 4096 == 0 && fstat(fd, &st) == 0 && st.st_size > (32 << 20))
		ftruncate(fd, 0);
}

static void m2v_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void m2v_log(const char *fmt, ...)
{
	char buf[600];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	sm2_logs(buf);
}

/* The A630 does not snoop the CPU caches (as guacamole's A640): libSystem's sys_dcache_flush is a no-op on Apple GPUs.
 * User space may clean by VA (XNU sets SCTLR_EL1.UCI). This definition wins at link time for the frontend and bridge. */
extern "C" void sys_dcache_flush(void *start, size_t len)
{
	uintptr_t a = (uintptr_t)start & ~(uintptr_t)63, end = (uintptr_t)start + len;
	for (; a < end; a += 64)
		__asm__ volatile("dc civac, %0" ::"r"(a) : "memory");
	__asm__ volatile("dsb sy" ::: "memory");
}

/* ash's Entry::load_from(path) wants vkGetInstanceProcAddr; our Turnip exports only the ICD entry points. The bridge is
 * given this dylib's own path as its loader, and this forwards to Turnip's vk_icdGetInstanceProcAddr (Mesa answers the
 * global commands for a NULL instance). */
typedef void (*m2v_vkfn)(void);
extern "C" __attribute__((visibility("default"))) m2v_vkfn vkGetInstanceProcAddr(void *instance, const char *name)
{
	static m2v_vkfn (*icd)(void *, const char *);
	if (!icd) {
		/* the Turnip the inject already loaded (by @rpath, maybe through the /usr/local/lib/bgmtl link): never a second copy */
		icd = (m2v_vkfn(*)(void *, const char *))dlsym(RTLD_DEFAULT, "vk_icdGetInstanceProcAddr");
		void *h = icd ? NULL : dlopen(M2V_TURNIP, RTLD_NOW | RTLD_GLOBAL);
		if (!icd)
			icd = h ? (m2v_vkfn(*)(void *, const char *))dlsym(h, "vk_icdGetInstanceProcAddr") : NULL;
		if (!icd)
			m2v_log("no vk_icdGetInstanceProcAddr in %s: %s", M2V_TURNIP, dlerror());
	}
	return icd ? icd(instance, name) : NULL;
}

static void missing(const char *cls, const char *sel)
{
	static NSMutableSet *seen;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		seen = [NSMutableSet new];
	});
	NSString *k = [NSString stringWithFormat:@"%s %s", cls, sel];
	@synchronized(seen) {
		if ([seen containsObject:k])
			return;
		[seen addObject:k];
	}
	m2v_log("MISSING %s -%s", cls, sel);
}

/* Private probes CoreImage and QuartzCore send (guacamole BergamotaMetal2.mm): unanswered, CoreImage's dispatch_once never
 * finished. */
@interface NBDeviceMethods : NSObject
@end
@implementation NBDeviceMethods (M2V)
- (NSUInteger)sharedMemorySize
{
	return 1ull << 30;
}
- (NSUInteger)dedicatedMemorySize
{
	return 0;
}
- (uint64_t)recommendedMaxWorkingSetSize
{
	return 768ull << 20;
}
- (NSUInteger)currentAllocatedSize
{
	return 0;
}
- (id)newPipelineLibraryWithFilePath:(NSString *)path error:(NSError **)error
{
	if (error)
		*error = [NSError errorWithDomain:@"M2V" code:1 userInfo:@{NSLocalizedDescriptionKey : @"no pipeline libraries"}];
	return nil;
}
@end

/*
 * Crimson (/private/var/tmp/m2v-sbkeepicons, SpringBoard): every icon cell of a home-screen page counts as prefetched, so
 * SpringBoard keeps all their images instead of dropping the off-screen ones and asking iconservicesagent for them again as
 * the page scrolls. Each ask re-composites iOS 26's layered icon on the CPU (RenderBox's CoreGraphics backend), ~1 core
 * during page swipes, and SpringBoard waits for the replies (2026-11-25 profiles).
 */
static void (*sb_set_cells)(id, SEL, long, id, id);
static Class sb_home_list;
static void sb_set_cells_all(id me, SEL cmd, long visibility, id visible, id prefetched)
{
	static NSIndexSet *all = [[NSIndexSet alloc] initWithIndexesInRange:NSMakeRange(0, 64)];
	(void)prefetched;
	/* visibility: 0 on screen, 1 partly, 2 off. A home list (SBIconListView itself: the pages and the Today view; not the dock,
	 * library or Control Center) that goes off has its icon views removed (-removeAllIconViews), and the Today view's 9 widget
	 * stacks load whenever page 1 settles: a drag between pages 1 and 2 rebuilt page 2 and Today on every return. Off and
	 * partly visible are both reported as partly visible with every cell, so they stay built. */
	if (visibility != 0 && object_getClass(me) == sb_home_list) {
		visibility = 1;
		visible = all;
	}
	sb_set_cells(me, cmd, visibility, visible, all);
}
static void sb_keep_icons(int tries)
{
	Class c = objc_getClass("SBIconListView");
	SEL sel = sel_registerName("setCellVisibility:visibleGridCellIndexes:prefetchedGridCellIndexes:");
	Method m = c ? class_getInstanceMethod(c, sel) : NULL;
	if (!m) {
		if (tries > 0)
			dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ sb_keep_icons(tries - 1); });
		else
			m2v_log("sbkeepicons: SBIconListView -setCellVisibility:... not found");
		return;
	}
	m2v_log("sbkeepicons: hooking -[SBIconListView setCellVisibility:visibleGridCellIndexes:prefetchedGridCellIndexes:] (%s)",
		method_getTypeEncoding(m));
	sb_home_list = c;
	sb_set_cells = (void (*)(id, SEL, long, id, id))method_setImplementation(m, (IMP)sb_set_cells_all);
	/* -setContentVisibility: (0 on screen, 3 off and prepared, 2 off): a home list going 2 -> 3 sets up every widget stack's
	 * multiplexed view controller and 3 -> 2 tears them down; SpringBoard flaps page 2 and the Today view between the two
	 * each time page 1 settles. Home lists stay prepared: 2 is reported as 3. */
	Method cv = class_getInstanceMethod(c, sel_registerName("setContentVisibility:"));
	if (cv && !strcmp(method_getTypeEncoding(cv), "v24@0:8Q16")) {
		static void (*set_content)(id, SEL, unsigned long);
		set_content = (void (*)(id, SEL, unsigned long))method_getImplementation(cv);
		method_setImplementation(cv, imp_implementationWithBlock(^(id me, unsigned long v) {
			if (v == 2 && object_getClass(me) == sb_home_list)
				v = 3;
			set_content(me, sel_registerName("setContentVisibility:"), v);
		}));
	} else
		m2v_log("sbkeepicons: no -[SBIconListView setContentVisibility:] with the expected signature");
	/* When home page 1 settles, SpringBoard flaps the Today view (left of page 1) to prepared and back within ~70 ms, and
	 * -[SBHTodayViewController viewWillDisappear:] runs appearance transitions on every widget stack, whose
	 * -viewIsAppearing: sets each multiplexed widget view controller up again: ~0.25-0.45 s on SpringBoard's main thread,
	 * once per return to page 1. The Today view's children skip those transitions; its widgets still show when it is opened.
	 * With all three changes a 20 s drag between pages 1 and 2 runs at 57-60 fps (40-43 without, a 0.4 s stall per sweep). */
	Method today = class_getInstanceMethod(objc_getClass("SBTodayViewController"),
					       sel_registerName("_beginAppearanceTransitionForChildViewControllersToVisible:animated:"));
	if (today && !strcmp(method_getTypeEncoding(today), "v24@0:8B16B20"))
		method_setImplementation(today, imp_implementationWithBlock(^(id me, BOOL visible, BOOL animated) {
			(void)me; (void)visible; (void)animated;
		}));
	else
		m2v_log("sbkeepicons: no -[SBTodayViewController _beginAppearanceTransitionForChildViewControllersToVisible:animated:]");
	/* /private/var/tmp/m2v-blurcap: no blur finer than half resolution. The App Library's glass took 20-25 ms of GPU a frame
	 * (~40 fps); some of its backdrops captured and filtered the screen at full resolution (830x1794). */
	Method sc = class_getInstanceMethod(objc_getClass("CABackdropLayer"), sel_registerName("setScale:"));
	if (access(crimson_switch("blurcap"), F_OK) == 0 && sc && !strcmp(method_getTypeEncoding(sc), "v24@0:8d16")) {
		static void (*set_scale)(id, SEL, double);
		set_scale = (void (*)(id, SEL, double))method_getImplementation(sc);
		method_setImplementation(sc, imp_implementationWithBlock(^(id me, double v) {
			set_scale(me, sel_registerName("setScale:"), v > 0.5 ? 0.5 : v);
		}));
		m2v_log("blurcap: backdrop scale capped at 0.5");
	}
}

extern "C" __attribute__((visibility("default"))) id m2v_create_device(void)
{
	static id device;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		Dl_info self = {0};
		dladdr((void *)m2v_create_device, &self);
		char cache[256];
		const char *cdir = crimson_switch("cache");
		snprintf(cache, sizeof(cache), "%s/%s", cdir, getprogname());
		mkdir(cdir, 0777);
		chmod(cdir, 0777);
		mkdir(cache, 0777);
		if (access(cache, W_OK) != 0) {   /* sandboxed: its own caches directory */
			char dir[200];
			if (confstr(_CS_DARWIN_USER_CACHE_DIR, dir, sizeof(dir)))
				snprintf(cache, sizeof(cache), "%sm2v", dir), mkdir(cache, 0700);
		}
		/* plain rows everywhere: no Apple framebuffer compression, no 10-bit deep framebuffer (guacamole) */
		setenv("CA_DISABLE_FRAMEBUFFER_COMPRESSION", "1", 1);
		setenv("METAL2VULKAN_LOG_PIPELINES", "1", 0);   /* the runtime counts its VkPipeline makings (fps meter, telemetry) */
		CFPreferencesSetAppValue(CFSTR("CAEnableFramebufferCompression"), kCFBooleanFalse, kCFPreferencesCurrentApplication);
		CFPreferencesSetAppValue(CFSTR("CAEnableIOSurfaceCompression"), kCFBooleanFalse, kCFPreferencesCurrentApplication);
		CFPreferencesSetAppValue(CFSTR("CAEnableDeepFramebuffer"), kCFBooleanFalse, kCFPreferencesCurrentApplication);
		/* debugging: KEY=VALUE lines into this process's environment (TU_DEBUG, MESA_LOG_FILE, METAL2VULKAN_*);
		 * PROCESS:KEY=VALUE only into the process named PROCESS (e.g. backboardd:MESA_GPU_TRACES=print) */
		if (FILE *ef = fopen(crimson_switch("env"), "r")) {
			char line[256];
			while (fgets(line, sizeof(line), ef)) {
				line[strcspn(line, "\r\n")] = 0;
				char *eq = strchr(line, '='), *key = line, *colon = strchr(line, ':');
				if (colon && eq && colon < eq) {
					*colon = 0;
					if (strcmp(line, getprogname()))
						continue;
					key = colon + 1;
				}
				if (line[0] != '#' && eq) {
					*eq = 0;
					setenv(key, eq + 1, 1);
					m2v_log("env %s=%s", key, eq + 1);
				}
			}
			fclose(ef);
		}
		/* debugging: CHL_M2V_STDERR=PATH (m2v-env) appends this process's stdout and stderr there (ir3's IR3_SHADER_DEBUG=fs disassembly) */
		if (const char *ep = getenv("CHL_M2V_STDERR")) {
			int fd = open(ep, O_WRONLY | O_APPEND | O_CREAT, 0666);
			if (fd >= 0) {
				dup2(fd, 1);
				dup2(fd, 2);
				close(fd);
			}
		}
		CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
		@try {
			device = native_bridge_metal_open(self.dli_fname, cache, 0, missing);
		} @catch (NSException *e) {
			m2v_log("device: exception %s", e.reason.UTF8String);
			device = nil;
		}
		m2v_log("device %s in %.0f ms (loader %s, cache %s)", device ? [[device name] UTF8String] : "refused",
			(CFAbsoluteTimeGetCurrent() - t0) * 1000, self.dli_fname, cache);
		/* debugging: the frontend's counters every 10 s into m2v.log while /private/var/tmp/m2v-telemetry exists */
		if (device && access(crimson_switch("telemetry"), F_OK) == 0)
			dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
				for (;;) {
					sleep(10);
					static char text[8192];
					native_bridge_metal_telemetry(text, sizeof(text));
					for (char *line = text, *next; *line; line = next) {
						next = strchr(line, '\n');
						if (next) *next++ = 0; else next = line + strlen(line);
						if (line[0] == 'J' || line[0] == 'G' || line[0] == 'T' || line[0] == 'S' || line[0] == 'V') m2v_log("tel %s", line);
					}
				}
			});
		if (strcmp(getprogname(), "SpringBoard") == 0 && access(crimson_switch("sbkeepicons"), F_OK) == 0)
			sb_keep_icons(15);
		if (device && strcmp(getprogname(), "backboardd") != 0)
			class_replaceMethod(object_getClass(device), sel_registerName("supportsExtendedXR10Formats"),
					    imp_implementationWithBlock(^BOOL(id me) {
						    (void)me;
						    return YES;
					    }),
					    "B16@0:8");
	});
	return device;
}
