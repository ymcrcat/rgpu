// Checks the fake driver itself: that it keeps contexts apart the way a real
// driver does.
//
// Every other test reaches the fake through the server, where one thread
// serves a whole session, so none of them can see what the fake does with two
// threads or two devices. This one links the fake directly and drives it from
// several OS threads, because that is where the real driver keeps its state:
// the current context and the context stack belong to the calling thread, and
// what a context holds belongs to that context, whichever context is current
// when it is used.
//
// Issue #2 was invisible because the fake did not model this. These are the
// behaviours later tests lean on to fail loudly, so they are pinned here
// first.
//
// Run with RGPU_FAKE_DEVICES=2 (CMake sets it).

#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include <cuda.h>

#include "tests/fake_stats.h"

namespace {

int g_failures = 0;

#define EXPECT_RC(call, want)                                               \
  do {                                                                      \
    CUresult r_ = (call);                                                   \
    if (r_ != (want)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d, expected %d\n", __FILE__, \
                   __LINE__, #call, r_, (want));                            \
      g_failures++;                                                         \
    }                                                                       \
  } while (0)

#define CHECK(call) EXPECT_RC(call, CUDA_SUCCESS)

#define EXPECT(cond, msg)                                                    \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

// Runs `fn` on a fresh OS thread and waits for it. A fresh thread has no
// current context and an empty stack, which is the point.
void on_other_thread(const std::function<void()>& fn) {
  std::thread(fn).join();
}

long stat(rgpu_fake::Kind k) { return rgpu_fake::value(k); }

// A fatbin header the fake accepts, as in launch_smoke.
std::vector<unsigned char> fake_fatbin() {
  std::vector<unsigned char> img(16 + 64, 0xAB);
  const unsigned int magic = 0xBA55ED50u;
  std::memcpy(img.data(), &magic, sizeof(magic));
  return img;
}

CUcontext current() {
  CUcontext c = reinterpret_cast<CUcontext>(0x1);
  if (cuCtxGetCurrent(&c) != CUDA_SUCCESS) return reinterpret_cast<CUcontext>(0x1);
  return c;
}

void devices() {
  int n = 0;
  CHECK(cuDeviceGetCount(&n));
  EXPECT(n == 2, "RGPU_FAKE_DEVICES=2 should give two devices");
  CUdevice d = -1;
  CHECK(cuDeviceGet(&d, 1));
  EXPECT(d == 1, "ordinal 1 should be device 1");
  EXPECT_RC(cuDeviceGet(&d, 2), CUDA_ERROR_INVALID_DEVICE);
  size_t mem = 0;
  CHECK(cuDeviceTotalMem(&mem, 1));
  CUcontext c = nullptr;
  EXPECT_RC(cuDevicePrimaryCtxRetain(&c, 2), CUDA_ERROR_INVALID_DEVICE);
}

// Each device has a primary context of its own, retained and released on its
// own count.
void primary_per_device() {
  CUcontext p0 = nullptr, p0_again = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p0_again, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  EXPECT(p0 && p1 && p0 != p1, "two devices must not share a primary context");
  EXPECT(p0 == p0_again, "a device's primary context is one context");

  unsigned int flags = 0;
  int active = -1;
  CHECK(cuDevicePrimaryCtxGetState(1, &flags, &active));
  EXPECT(active == 1, "device 1's primary context should be active");
  CHECK(cuDevicePrimaryCtxRelease(1));
  CHECK(cuDevicePrimaryCtxGetState(1, &flags, &active));
  EXPECT(active == 0, "releasing device 1's only retain leaves it inactive");
  CHECK(cuDevicePrimaryCtxGetState(0, &flags, &active));
  EXPECT(active == 1, "releasing device 1 must not touch device 0");

  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(0));
  EXPECT_RC(cuDevicePrimaryCtxRelease(0), CUDA_ERROR_INVALID_CONTEXT);
}

// The current context is the calling thread's, and one thread changing its
// own leaves every other thread's alone.
void currency_is_per_thread() {
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  CHECK(cuCtxSetCurrent(p0));

  CUcontext seen = reinterpret_cast<CUcontext>(0x1);
  on_other_thread([&] {
    seen = current();
    CHECK(cuCtxSetCurrent(p1));
  });
  EXPECT(seen == nullptr, "a new thread must start with no current context");
  EXPECT(current() == p0,
         "another thread making its context current changed this thread's");

  // And with nothing current, a call that needs a context says so.
  on_other_thread([&] {
    CUdeviceptr d = 0;
    EXPECT_RC(cuMemAlloc(&d, 64), CUDA_ERROR_INVALID_CONTEXT);
    EXPECT_RC(cuCtxSynchronize(), CUDA_ERROR_INVALID_CONTEXT);
  });

  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// A pointer carries its context with it. The header's Unified Addressing
// overview says "Since pointers are unique, it is not necessary to specify
// information about the pointers specified to the various copy functions",
// and that "the CUDA driver should infer the location of the pointer from its
// value", so a copy, a memset and a free of another context's pointer act on
// that context's memory and succeed. What a test can rely on is placement:
// which context and device own the allocation, read back through the pointer
// attributes.
void pointers_carry_their_context() {
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  CHECK(cuCtxSetCurrent(p0));
  CUdeviceptr a = 0;
  CHECK(cuMemAlloc(&a, 256));
  const unsigned char bytes[16] = {1, 2, 3, 4, 5, 6, 7, 8};
  const long allocs = stat(rgpu_fake::kAlloc);
  const long stale = stat(rgpu_fake::kStale);

  const long sets = stat(rgpu_fake::kCtxSetCurrent);
  const long cross = stat(rgpu_fake::kCrossContextUse);
  CHECK(cuMemcpyHtoD(a, bytes, 1));
  EXPECT(stat(rgpu_fake::kCrossContextUse) == cross,
         "a copy under the pointer's own context is not a cross-context use");
  CHECK(cuCtxSetCurrent(p1));
  EXPECT(stat(rgpu_fake::kCtxSetCurrent) == sets + 1,
         "every cuCtxSetCurrent should be counted");
  unsigned char out[16] = {0};
  CHECK(cuMemcpyHtoD(a, bytes, sizeof(bytes)));
  EXPECT(stat(rgpu_fake::kCrossContextUse) == cross + 1,
         "a copy under another context should be counted as one");
  CHECK(cuMemcpyDtoH(out, a, sizeof(out)));
  EXPECT(std::memcmp(out, bytes, sizeof(out)) == 0,
         "a copy under another context should reach the pointer's own memory");
  CHECK(cuMemsetD8(a + 8, 0x7f, 8));
  CHECK(cuMemcpyDtoH(out, a, sizeof(out)));
  EXPECT(out[0] == 1 && out[8] == 0x7f, "memset under another context");

  // Placement, from any context and from a thread with none.
  CUcontext owner = nullptr;
  int ordinal = -1;
  unsigned int type = 0;
  CHECK(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, a));
  CHECK(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, a));
  CHECK(cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, a));
  EXPECT(owner == p0, "the pointer should report the context it was made in");
  EXPECT(ordinal == 0, "the pointer should report device 0");
  EXPECT(type == CU_MEMORYTYPE_DEVICE, "the pointer should be device memory");
  owner = nullptr;
  CHECK(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, a + 100));
  EXPECT(owner == p0, "a pointer into the middle of an allocation is placed too");

  CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_CONTEXT,
                                 CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                 CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
  owner = nullptr;
  ordinal = -1;
  type = 0;
  void* data[] = {&owner, &ordinal, &type};
  CHECK(cuPointerGetAttributes(3, attrs, data, a));
  EXPECT(owner == p0 && ordinal == 0 && type == CU_MEMORYTYPE_DEVICE,
         "cuPointerGetAttributes should agree with cuPointerGetAttribute");

  // Not a CUDA pointer: one call refuses, the other answers with nothing.
  const CUdeviceptr bogus = 0x1234;
  EXPECT_RC(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, bogus),
            CUDA_ERROR_INVALID_VALUE);
  owner = p1;
  ordinal = 5;
  type = 9;
  CHECK(cuPointerGetAttributes(3, attrs, data, bogus));
  EXPECT(owner == nullptr && ordinal == 0 && type == 0,
         "cuPointerGetAttributes should give default values for a non-CUDA "
         "pointer");

  on_other_thread([&] {
    CUcontext seen = nullptr;
    CHECK(cuPointerGetAttribute(&seen, CU_POINTER_ATTRIBUTE_CONTEXT, a));
    EXPECT(seen == p0, "placement should not need a current context");
    unsigned char none[4] = {0};
    EXPECT_RC(cuMemcpyDtoH(none, a, sizeof(none)), CUDA_ERROR_INVALID_CONTEXT);
    CUdevice dev = -1;
    EXPECT_RC(cuCtxGetDevice(&dev), CUDA_ERROR_INVALID_CONTEXT);
  });

  // The current context's device.
  CUdevice dev = -1;
  CHECK(cuCtxGetDevice(&dev));
  EXPECT(dev == 1, "device 1's primary context should be on device 1");
  CUcontext mine = nullptr;
  CHECK(cuCtxCreate(&mine, 0, 1));
  dev = -1;
  CHECK(cuCtxGetDevice(&dev));
  EXPECT(dev == 1, "a context created on device 1 should be on device 1");
  CUdeviceptr b = 0;
  CHECK(cuMemAlloc(&b, 64));
  CHECK(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, b));
  CHECK(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, b));
  EXPECT(owner == mine && ordinal == 1,
         "an allocation should be placed in the context current when made");
  CHECK(cuCtxDestroy(mine));
  CHECK(cuCtxSetCurrent(p1));

  // Freed under another context, and really freed: not stale, and gone.
  CHECK(cuMemFree(a));
  EXPECT(stat(rgpu_fake::kAlloc) == allocs - 1,
         "a free under another context should free the pointer's memory");
  EXPECT(stat(rgpu_fake::kStale) == stale, "a live pointer is not stale");
  EXPECT_RC(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, a),
            CUDA_ERROR_INVALID_VALUE);

  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// Launches the checked kernel with the arguments the fake expects.
CUresult launch(CUfunction f, CUstream s) {
  unsigned char packed[28] = {0};
  const unsigned long long ptrs[3] = {0x1111111111111111ull,
                                      0x2222222222222222ull,
                                      0x3333333333333333ull};
  const int n = 42;
  std::memcpy(packed, ptrs, sizeof(ptrs));
  std::memcpy(packed + 24, &n, sizeof(n));
  size_t size = sizeof(packed);
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, nullptr, extra);
}

// Streams, events, modules and graphs carry their context the same way: the
// header says nothing that confines them to the current context, so they are
// used, and destroyed, wherever they live. Two calls are stricter:
//   - cuModuleUnload "Unloads a module hmod from the current context", and
//     reports it with the CUDA_ERROR_INVALID_VALUE its return list names;
//   - a launch with a stream from another context than the function's
//     ("The CUDA context associated with this stream must match that
//     associated with function f", on cuLaunchKernelEx), and - inferred, not
//     documented for a CUfunction - a launch with the null stream while the
//     function's context is not current.
void handles_carry_their_context() {
  const long base[] = {stat(rgpu_fake::kModule), stat(rgpu_fake::kGraph),
                       stat(rgpu_fake::kGraphExec), stat(rgpu_fake::kStream),
                       stat(rgpu_fake::kEvent)};
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  CHECK(cuCtxSetCurrent(p0));
  CUstream s = nullptr;
  CUevent e = nullptr;
  CUmodule m = nullptr;
  CUfunction f = nullptr;
  CUgraph g = nullptr;
  CHECK(cuStreamCreate(&s, 0));
  CHECK(cuEventCreate(&e, 0));
  const std::vector<unsigned char> image = fake_fatbin();
  CHECK(cuModuleLoadData(&m, image.data()));
  CHECK(cuModuleGetFunction(&f, m, "rgpu_check_args"));
  CHECK(launch(f, s));
  CHECK(launch(f, nullptr));
  CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(s, &g));

  const long stale = stat(rgpu_fake::kStale);
  CHECK(cuCtxSetCurrent(p1));
  CUstream s1 = nullptr;
  CHECK(cuStreamCreate(&s1, 0));
  CUfunction f1 = nullptr;
  size_t off = 0, size = 0;
  CHECK(cuStreamSynchronize(s));
  CHECK(cuModuleGetFunction(&f1, m, "rgpu_check_args"));
  EXPECT(f1 == f, "the same function from the same module");
  CHECK(cuFuncGetParamInfo(f, 0, &off, &size));
  CHECK(launch(f, s));
  EXPECT_RC(launch(f, nullptr), CUDA_ERROR_INVALID_HANDLE);
  EXPECT_RC(launch(f, s1), CUDA_ERROR_INVALID_HANDLE);
  EXPECT_RC(cuModuleUnload(m), CUDA_ERROR_INVALID_VALUE);
  EXPECT(stat(rgpu_fake::kModule) == base[0] + 1,
         "an unload refused under another context must not unload");
  CUgraph clone = nullptr;
  CUgraphExec exec = nullptr;
  CHECK(cuGraphClone(&clone, g));
  CHECK(cuGraphInstantiateWithFlags(&exec, clone, 0));
  CHECK(cuGraphLaunch(exec, s));
  CHECK(cuEventDestroy(e));
  CHECK(cuStreamDestroy(s1));
  EXPECT(stat(rgpu_fake::kStale) == stale,
         "a handle used under another context is live, not stale");

  // The clone and its executable were made from device 0's graph, so they
  // live in device 0's primary context however they were reached: resetting
  // device 0 takes them, the module, the graph and the stream with it.
  CHECK(cuDevicePrimaryCtxReset(0));
  // A module that no longer exists is the same code, and stale.
  CHECK(cuCtxSetCurrent(p0));
  EXPECT_RC(cuModuleUnload(m), CUDA_ERROR_INVALID_VALUE);
  EXPECT(stat(rgpu_fake::kStale) == stale + 1,
         "unloading a module the reset destroyed should count as stale");
  EXPECT(stat(rgpu_fake::kModule) == base[0] &&
             stat(rgpu_fake::kGraph) == base[1] &&
             stat(rgpu_fake::kGraphExec) == base[2] &&
             stat(rgpu_fake::kStream) == base[3] &&
             stat(rgpu_fake::kEvent) == base[4],
         "everything made from device 0's objects should die with its "
         "primary context");

  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// The context stack is per thread too, and creating a context pushes it.
void stack_is_per_thread() {
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  CHECK(cuCtxSetCurrent(p0));
  CHECK(cuCtxPushCurrent(p1));
  EXPECT(current() == p1, "a pushed context should be current");

  on_other_thread([&] {
    CUcontext popped = nullptr;
    EXPECT_RC(cuCtxPopCurrent(&popped), CUDA_ERROR_INVALID_CONTEXT);
    CHECK(cuCtxPushCurrent(p0));
    CHECK(cuCtxPopCurrent(&popped));
    EXPECT(popped == p0, "popped the wrong context on the other thread");
  });

  CUcontext popped = nullptr;
  CHECK(cuCtxPopCurrent(&popped));
  EXPECT(popped == p1, "pop should return the context that was pushed");
  EXPECT(current() == p0, "pop should restore the context beneath");

  CUcontext mine = nullptr;
  CHECK(cuCtxCreate(&mine, 0, 1));
  EXPECT(current() == mine, "a created context should be current");
  CHECK(cuCtxPopCurrent(&popped));
  EXPECT(popped == mine && current() == p0,
         "popping a created context should restore the one it supplanted");
  CHECK(cuCtxDestroy(mine));

  CHECK(cuCtxSetCurrent(nullptr));
  EXPECT(current() == nullptr, "setting null on a one-deep stack empties it");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// A context destroyed on one thread stays current on another, and that thread
// is told so rather than working on freed state.
void destroyed_under_another_thread() {
  CUcontext c = nullptr;
  on_other_thread([&] {
    CHECK(cuCtxCreate(&c, 0, 0));
    CHECK(cuCtxPopCurrent(nullptr));
  });
  CHECK(cuCtxSetCurrent(c));
  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, 64));
  const long allocs = stat(rgpu_fake::kAlloc);
  on_other_thread([&] { CHECK(cuCtxDestroy(c)); });
  EXPECT(stat(rgpu_fake::kAlloc) == allocs - 1,
         "destroying a context should free what was allocated in it");
  EXPECT(current() == c, "a context destroyed elsewhere stays current here");
  EXPECT_RC(cuMemAlloc(&d, 64), CUDA_ERROR_CONTEXT_IS_DESTROYED);
  EXPECT_RC(cuCtxSetCurrent(c), CUDA_ERROR_INVALID_CONTEXT);
  CHECK(cuCtxSetCurrent(nullptr));
}

// Device pointers are never handed out twice, so a free of an address that
// was freed before is always recognisably stale.
void addresses_are_not_reused() {
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuCtxSetCurrent(p0));
  CUdeviceptr a = 0, b = 0;
  CHECK(cuMemAlloc(&a, 256));
  CHECK(cuMemFree(a));
  CHECK(cuMemAlloc(&b, 256));
  EXPECT(a != b, "a freed device address was handed out again");
  const long stale = stat(rgpu_fake::kStale);
  EXPECT_RC(cuMemFree(a), CUDA_ERROR_INVALID_VALUE);
  EXPECT(stat(rgpu_fake::kStale) == stale + 1,
         "freeing an address twice should count as stale");
  CHECK(cuMemFree(b));
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// Resetting a device throws away everything in its primary context - not only
// memory - and nothing in any other context, on this device or another. It
// leaves the retain count alone.
void reset_empties_the_primary_context() {
  const long base[] = {stat(rgpu_fake::kAlloc),  stat(rgpu_fake::kModule),
                       stat(rgpu_fake::kStream), stat(rgpu_fake::kEvent),
                       stat(rgpu_fake::kGraph),  stat(rgpu_fake::kGraphExec)};
  const long base_retains = stat(rgpu_fake::kPrimaryRetain);
  const long over = stat(rgpu_fake::kOverRelease);
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  // Something on the other device's primary context, which must survive.
  CHECK(cuCtxSetCurrent(p1));
  CUdeviceptr other_device = 0;
  CHECK(cuMemAlloc(&other_device, 64));

  // And something in a context of the client's own on the same device.
  CUcontext own = nullptr;
  CHECK(cuCtxCreate(&own, 0, 0));
  CUdeviceptr own_ptr = 0;
  CHECK(cuMemAlloc(&own_ptr, 64));
  CHECK(cuCtxPopCurrent(nullptr));

  CHECK(cuCtxSetCurrent(p0));
  CUdeviceptr a = 0;
  CUstream s = nullptr;
  CUevent e = nullptr;
  CUmodule m = nullptr;
  CUgraph g = nullptr;
  CUgraphExec x = nullptr;
  CHECK(cuMemAlloc(&a, 64));
  CHECK(cuStreamCreate(&s, 0));
  CHECK(cuEventCreate(&e, 0));
  const std::vector<unsigned char> image = fake_fatbin();
  CHECK(cuModuleLoadData(&m, image.data()));
  CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(s, &g));
  CHECK(cuGraphInstantiateWithFlags(&x, g, 0));

  CHECK(cuDevicePrimaryCtxReset(0));
  EXPECT(stat(rgpu_fake::kAlloc) == base[0] + 2,
         "reset should free the primary context's memory and only that");
  EXPECT(stat(rgpu_fake::kModule) == base[1], "reset should unload modules");
  EXPECT(stat(rgpu_fake::kStream) == base[2], "reset should destroy streams");
  EXPECT(stat(rgpu_fake::kEvent) == base[3], "reset should destroy events");
  EXPECT(stat(rgpu_fake::kGraph) == base[4], "reset should destroy graphs");
  EXPECT(stat(rgpu_fake::kGraphExec) == base[5],
         "reset should destroy graph executables");

  // A reset does not release the context: "Resetting the primary context
  // does not release it, an application that has retained the primary
  // context should explicitly release its usage." The retain is still held,
  // the context is still active and usable, and releasing it later succeeds.
  unsigned int flags = 0;
  int active = -1;
  CHECK(cuDevicePrimaryCtxGetState(0, &flags, &active));
  EXPECT(active == 1, "a reset must not deactivate a retained primary context");
  EXPECT(stat(rgpu_fake::kPrimaryRetain) == base_retains + 2,
         "a reset must not drop the retain count");
  CUdeviceptr after = 0;
  CHECK(cuMemAlloc(&after, 64));
  CHECK(cuMemFree(after));

  // What the reset destroyed is gone, and freeing it again is stale.
  const long stale = stat(rgpu_fake::kStale);
  EXPECT_RC(cuMemFree(a), CUDA_ERROR_INVALID_VALUE);
  EXPECT_RC(cuStreamDestroy(s), CUDA_ERROR_INVALID_HANDLE);
  EXPECT(stat(rgpu_fake::kStale) == stale + 2,
         "releasing what a reset destroyed should count as stale");
  CHECK(cuDevicePrimaryCtxRelease(0));
  EXPECT_RC(cuDevicePrimaryCtxRelease(0), CUDA_ERROR_INVALID_CONTEXT);
  EXPECT(stat(rgpu_fake::kOverRelease) == over + 1,
         "one release per retain, reset or not");

  CHECK(cuCtxSetCurrent(p1));
  CHECK(cuMemFree(other_device));
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuCtxDestroy(own));  // frees own_ptr with it
  CHECK(cuDevicePrimaryCtxRelease(1));
  EXPECT(stat(rgpu_fake::kAlloc) == base[0], "allocations left behind");
}

// Releasing the last retain resets the primary context, as the driver
// documents, so what was in it goes too.
void last_release_resets() {
  const long allocs = stat(rgpu_fake::kAlloc);
  CUcontext p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  CHECK(cuCtxSetCurrent(p1));
  CUdeviceptr a = 0;
  CHECK(cuMemAlloc(&a, 64));
  CHECK(cuDevicePrimaryCtxRelease(1));
  EXPECT(stat(rgpu_fake::kAlloc) == allocs,
         "releasing the last retain should free the primary context's memory");
  // What the server asks to learn that a release was the last one.
  unsigned int flags = 0;
  int active = -1;
  CHECK(cuDevicePrimaryCtxGetState(1, &flags, &active));
  EXPECT(active == 0, "a primary context with no retains must report inactive");
  // Still current here, but no longer initialised.
  EXPECT_RC(cuMemAlloc(&a, 64), CUDA_ERROR_CONTEXT_IS_DESTROYED);
  CHECK(cuCtxSetCurrent(nullptr));
}

}  // namespace

int main() {
  devices();
  primary_per_device();
  currency_is_per_thread();
  pointers_carry_their_context();
  handles_carry_their_context();
  stack_is_per_thread();
  destroyed_under_another_thread();
  addresses_are_not_reused();
  reset_empties_the_primary_context();
  last_release_resets();

  for (int k = 0; k < rgpu_fake::kKindCount; k++) {
    if (k == rgpu_fake::kStale || k == rgpu_fake::kOverRelease ||
        k == rgpu_fake::kCtxSetCurrent || k == rgpu_fake::kCrossContextUse) {
      continue;  // mistakes and observations, not resources
    }
    if (rgpu_fake::value(static_cast<rgpu_fake::Kind>(k)) != 0) {
      std::fprintf(stderr, "FAIL: counter %d is %ld at the end, expected 0\n",
                   k, rgpu_fake::value(static_cast<rgpu_fake::Kind>(k)));
      g_failures++;
    }
  }

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: the fake driver keeps contexts, threads and devices apart\n");
  return 0;
}
