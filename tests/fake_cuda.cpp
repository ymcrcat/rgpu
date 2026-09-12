// A driver that pretends to be a GPU, backed by host memory.
//
// Linked into rgpu-server-fake so the whole stack can be exercised without a
// GPU: real client stubs, real wire format, real server dispatch. Only the
// bottom of the stack is fake.
//
// These definitions are strong and override the weak ones in
// tests/generated/fake_driver.cpp.

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include <cuda.h>

#include "tests/fake_stats.h"

namespace {

std::mutex g_mu;
// Device pointers are just host allocations here. Sizes are tracked so a copy
// running off the end is caught rather than corrupting the heap, and contexts
// so that resetting a device throws away what was allocated in its primary
// context and nothing else.
struct Alloc {
  size_t size;
  CUcontext ctx;
};
std::map<CUdeviceptr, Alloc> g_allocs;

// Everything else the fake hands out. A real driver knows which of its handles
// are still alive; this one has to as well, or a test could not tell a handle
// that was released from one that was merely forgotten. Destroying something
// that was never created, or twice, is an error here rather than a shrug.
std::set<unsigned long long> g_contexts, g_modules, g_streams, g_events,
    g_graphs, g_graph_execs;
// Per device, how many retains the whole process holds. A primary context is
// shared, so this is the count that must not go negative.
std::map<int, int> g_primary_retains;
unsigned long long g_next_handle = 1;

// Hands out a distinct token per creation, tagged so a value that turns up in
// the wrong place is recognisable in a log.
unsigned long long mint(std::set<unsigned long long>* into, unsigned tag) {
  std::lock_guard<std::mutex> lk(g_mu);
  unsigned long long h = (static_cast<unsigned long long>(tag) << 32) |
                         g_next_handle++;
  into->insert(h);
  return h;
}

bool retire(std::set<unsigned long long>* from, unsigned long long h) {
  std::lock_guard<std::mutex> lk(g_mu);
  return from->erase(h) != 0;
}

bool alive(std::set<unsigned long long>* in, unsigned long long h) {
  std::lock_guard<std::mutex> lk(g_mu);
  return in->count(h) != 0;
}

// The context this thread is running in, which is thread state in a real
// driver too.
thread_local CUcontext t_current = nullptr;

// The primary context's token. One per device, so an allocation made in it can
// be told from one made in a context of the client's own.
CUcontext primary_token() { return reinterpret_cast<CUcontext>(0xC0FFEE01); }

bool range_ok(CUdeviceptr p, size_t n) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Find the allocation containing p: the last one starting at or below it.
  auto it = g_allocs.upper_bound(p);
  if (it == g_allocs.begin()) return false;
  --it;
  return p >= it->first && p + n <= it->first + it->second.size;
}

constexpr int kFakeDevice = 0;
constexpr int kFakeDriverVersion = 12080;

}  // namespace

extern "C" {

CUresult cuInit(unsigned int) { return CUDA_SUCCESS; }

CUresult cuDriverGetVersion(int* v) {
  if (!v) return CUDA_ERROR_INVALID_VALUE;
  *v = kFakeDriverVersion;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  *count = 1;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
  *device = kFakeDevice;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  if (!name || len <= 0 || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  std::snprintf(name, static_cast<size_t>(len), "rgpu fake device");
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  if (!pi || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = 8; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = 9; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *pi = 58; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = 1024; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *pi = 32; break;
    default: *pi = 0; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  if (!bytes || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  *bytes = size_t(24) << 30;
  return CUDA_SUCCESS;
}

// Contexts are opaque to everything above, but they are counted here: a
// context nobody destroyed is exactly the kind of leak this fake exists to
// make visible. Creating one makes it current, as the real driver does.
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int, CUdevice dev) {
  if (!pctx || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  *pctx = reinterpret_cast<CUcontext>(mint(&g_contexts, 0xC0FFEE00u));
  t_current = *pctx;
  rgpu_fake::count(rgpu_fake::kContext, 1);
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  if (!retire(&g_contexts, reinterpret_cast<unsigned long long>(ctx))) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  if (t_current == ctx) t_current = nullptr;
  rgpu_fake::count(rgpu_fake::kContext, -1);
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) { return CUDA_SUCCESS; }

// The runtime binds a primary context per device, so the fake needs these too.
// One token per device however many times it is retained, which is the whole
// point of a primary context: the count is what is owned, not the handle.
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  if (!pctx || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_primary_retains[dev]++;
  }
  *pctx = primary_token();
  rgpu_fake::count(rgpu_fake::kPrimaryRetain, 1);
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  if (dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  {
    // A release that nobody paid for would take the context away from whoever
    // else is using it. The real driver treats it as an error, and so does
    // this: a test that over-releases should fail here, loudly, rather than
    // leave a live session running on a context that has been torn down.
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_primary_retains.find(dev);
    if (it == g_primary_retains.end() || it->second <= 0) {
      // Recorded, because this is the whole hazard: a release nobody paid for
      // is one taken off another session, and refusing it here means the
      // counters stay right and nothing else would show that it happened.
      rgpu_fake::count(rgpu_fake::kOverRelease, 1);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    it->second--;
  }
  rgpu_fake::count(rgpu_fake::kPrimaryRetain, -1);
  return CUDA_SUCCESS;
}

// Destroys everything in the device's primary context and drops the reference
// count with it, which is the reading the server is careful to survive: a
// session that had retains before a reset owes nothing afterwards.
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  if (dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  int dropped = 0;
  std::map<CUdeviceptr, Alloc> freed;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_primary_retains.find(dev);
    if (it != g_primary_retains.end()) {
      dropped = it->second;
      it->second = 0;
    }
    for (auto a = g_allocs.begin(); a != g_allocs.end();) {
      if (a->second.ctx == primary_token()) {
        freed.insert(*a);
        a = g_allocs.erase(a);
      } else {
        ++a;
      }
    }
  }
  for (const auto& a : freed) {
    std::free(reinterpret_cast<void*>(a.first));
    rgpu_fake::count(rgpu_fake::kAlloc, -1);
  }
  for (int i = 0; i < dropped; i++) {
    rgpu_fake::count(rgpu_fake::kPrimaryRetain, -1);
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = t_current;
  return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
  if (!free || !total) return CUDA_ERROR_INVALID_VALUE;
  *total = size_t(24) << 30;
  *free = size_t(20) << 30;
  return CUDA_SUCCESS;
}

// Deliberately permissive about which context: the primary context's token is
// not in g_contexts, and neither are the tokens a client may still be holding
// from a context it destroyed. Refusing one of those here would fail calls
// that a real driver accepts.
CUresult cuCtxSetCurrent(CUcontext ctx) {
  t_current = ctx;
  return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
  void* p = std::malloc(bytesize);
  if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
  auto d = reinterpret_cast<CUdeviceptr>(p);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_allocs[d] = Alloc{bytesize, t_current};
  }
  rgpu_fake::count(rgpu_fake::kAlloc, 1);
  *dptr = d;
  return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_allocs.erase(dptr) == 0) return CUDA_ERROR_INVALID_VALUE;
  }
  std::free(reinterpret_cast<void*>(dptr));
  rgpu_fake::count(rgpu_fake::kAlloc, -1);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t n) {
  if (!src) return CUDA_ERROR_INVALID_VALUE;
  if (!range_ok(dst, n)) return CUDA_ERROR_INVALID_VALUE;
  std::memcpy(reinterpret_cast<void*>(dst), src, n);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t n) {
  if (!dst) return CUDA_ERROR_INVALID_VALUE;
  if (!range_ok(src, n)) return CUDA_ERROR_INVALID_VALUE;
  std::memcpy(dst, reinterpret_cast<const void*>(src), n);
  return CUDA_SUCCESS;
}

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t n) {
  if (!range_ok(dst, n)) return CUDA_ERROR_INVALID_VALUE;
  std::memset(reinterpret_cast<void*>(dst), value, n);
  return CUDA_SUCCESS;
}

// The one fire-and-forget call the fake actually completes. Every other
// asynchronous entry point falls through to the generated stub and fails,
// which is fine until a test needs asynchronous work that works: without it
// there is no way to put successful no-reply traffic either side of a failure.
CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t n,
                         CUstream) {
  return cuMemsetD8_v2(dst, value, n);
}

// ---------------------------------------------------------------------------
// Modules and kernels
// ---------------------------------------------------------------------------
//
// No kernel can actually run without a GPU, but everything up to the launch
// can be checked: sizing the fatbin, shipping it, resolving a function,
// reporting its parameter layout, and packing arguments into that layout.
// That path is the most intricate part of the system, so the fake validates
// the arguments it receives instead of ignoring them.
//
// The kernel named by kCheckedKernel expects exactly the values in
// kExpectedArgs. A mismatch is reported as an error, which reaches the client
// as a failed launch.

constexpr const char* kCheckedKernel = "rgpu_check_args";
const unsigned long long kExpectedPtrs[3] = {0x1111111111111111ull,
                                             0x2222222222222222ull,
                                             0x3333333333333333ull};
constexpr int kExpectedInt = 42;

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
  // The client sized this from the image header; a wrong size would have
  // truncated it before it got here.
  unsigned int magic = 0;
  std::memcpy(&magic, image, sizeof(magic));
  if (magic != 0xBA55ED50u) return CUDA_ERROR_INVALID_IMAGE;
  *module = reinterpret_cast<CUmodule>(mint(&g_modules, 0x0D0100u));
  rgpu_fake::count(rgpu_fake::kModule, 1);
  return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule m) {
  if (!retire(&g_modules, reinterpret_cast<unsigned long long>(m))) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  rgpu_fake::count(rgpu_fake::kModule, -1);
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                             const char* name) {
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  if (!alive(&g_modules, reinterpret_cast<unsigned long long>(hmod))) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  if (std::strcmp(name, kCheckedKernel) != 0) return CUDA_ERROR_NOT_FOUND;
  *hfunc = reinterpret_cast<CUfunction>(0xF0C0100ull);
  return CUDA_SUCCESS;
}

// Three pointers then an int, which is the layout of the checked kernel.
CUresult cuFuncGetParamInfo(CUfunction f, size_t index, size_t* offset,
                            size_t* size) {
  if (!f || !offset || !size) return CUDA_ERROR_INVALID_VALUE;
  if (index < 3) {
    *offset = index * 8;
    *size = 8;
    return CUDA_SUCCESS;
  }
  if (index == 3) {
    *offset = 24;
    *size = 4;
    return CUDA_SUCCESS;
  }
  return CUDA_ERROR_INVALID_VALUE;  // past the last parameter
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy,
                        unsigned int gz, unsigned int bx, unsigned int by,
                        unsigned int bz, unsigned int shmem, CUstream stream,
                        void** kernelParams, void** extra) {
  (void)gy; (void)gz; (void)by; (void)bz; (void)shmem; (void)stream;
  if (!f) return CUDA_ERROR_INVALID_HANDLE;
  if (gx == 0 || bx == 0) return CUDA_ERROR_INVALID_VALUE;
  // The client always packs arguments and hands them over through extra.
  if (kernelParams) return CUDA_ERROR_INVALID_VALUE;
  if (!extra) return CUDA_ERROR_INVALID_VALUE;

  const unsigned char* buf = nullptr;
  size_t size = 0;
  for (size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; i++) {
    if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buf = static_cast<const unsigned char*>(extra[++i]);
    } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      size = *static_cast<const size_t*>(extra[++i]);
    } else {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  if (!buf || size < 28) return CUDA_ERROR_INVALID_VALUE;

  for (int i = 0; i < 3; i++) {
    unsigned long long v = 0;
    std::memcpy(&v, buf + i * 8, sizeof(v));
    if (v != kExpectedPtrs[i]) return CUDA_ERROR_INVALID_VALUE;
  }
  int n = 0;
  std::memcpy(&n, buf + 24, sizeof(n));
  if (n != kExpectedInt) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}

CUresult cuStreamCreate(CUstream* stream, unsigned int) {
  if (!stream) return CUDA_ERROR_INVALID_VALUE;
  *stream = reinterpret_cast<CUstream>(mint(&g_streams, 0x57EAu));
  rgpu_fake::count(rgpu_fake::kStream, 1);
  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream stream) {
  if (!retire(&g_streams, reinterpret_cast<unsigned long long>(stream))) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  rgpu_fake::count(rgpu_fake::kStream, -1);
  return CUDA_SUCCESS;
}

CUresult cuEventCreate(CUevent* event, unsigned int) {
  if (!event) return CUDA_ERROR_INVALID_VALUE;
  *event = reinterpret_cast<CUevent>(mint(&g_events, 0xE7E17u));
  rgpu_fake::count(rgpu_fake::kEvent, 1);
  return CUDA_SUCCESS;
}

CUresult cuEventDestroy_v2(CUevent event) {
  if (!retire(&g_events, reinterpret_cast<unsigned long long>(event))) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  rgpu_fake::count(rgpu_fake::kEvent, -1);
  return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream) { return CUDA_SUCCESS; }

// --- stream capture --------------------------------------------------------
//
// Enough to tell whether the calls arrive intact. The dependency array is the
// part worth checking: the driver owns it, so it cannot cross the wire as a
// pointer, and the client has to be handed a copy of the contents instead.

namespace {
bool g_capturing = false;
CUgraphNode g_nodes[2] = {reinterpret_cast<CUgraphNode>(0xDEB1),
                          reinterpret_cast<CUgraphNode>(0xDEB2)};
}  // namespace

CUresult cuStreamBeginCapture_v2(CUstream, CUstreamCaptureMode mode) {
  if (mode != CU_STREAM_CAPTURE_MODE_GLOBAL &&
      mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
      mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  g_capturing = true;
  return CUDA_SUCCESS;
}

CUresult cuStreamEndCapture(CUstream, CUgraph* graph) {
  if (!g_capturing) return CUDA_ERROR_ILLEGAL_STATE;
  g_capturing = false;
  if (!graph) return CUDA_ERROR_INVALID_VALUE;
  *graph = reinterpret_cast<CUgraph>(mint(&g_graphs, 0xC0FFEEu));
  rgpu_fake::count(rgpu_fake::kGraph, 1);
  return CUDA_SUCCESS;
}

CUresult cuStreamIsCapturing(CUstream, CUstreamCaptureStatus* status) {
  if (status) {
    *status = g_capturing ? CU_STREAM_CAPTURE_STATUS_ACTIVE
                          : CU_STREAM_CAPTURE_STATUS_NONE;
  }
  return CUDA_SUCCESS;
}

CUresult cuStreamGetCaptureInfo_v2(CUstream, CUstreamCaptureStatus* status,
                                   cuuint64_t* id, CUgraph* graph,
                                   const CUgraphNode** deps,
                                   size_t* ndeps) {
  if (status) {
    *status = g_capturing ? CU_STREAM_CAPTURE_STATUS_ACTIVE
                          : CU_STREAM_CAPTURE_STATUS_NONE;
  }
  if (id) *id = g_capturing ? 0x1D : 0;
  if (graph) *graph = g_capturing ? reinterpret_cast<CUgraph>(0xC0FFEEull)
                                  : nullptr;
  if (deps) *deps = g_nodes;
  if (ndeps) *ndeps = g_capturing ? 2 : 0;
  return CUDA_SUCCESS;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* exec, CUgraph graph,
                                     unsigned long long) {
  if (!alive(&g_graphs, reinterpret_cast<unsigned long long>(graph))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!exec) return CUDA_ERROR_INVALID_VALUE;
  *exec = reinterpret_cast<CUgraphExec>(mint(&g_graph_execs, 0xE7E0u));
  rgpu_fake::count(rgpu_fake::kGraphExec, 1);
  return CUDA_SUCCESS;
}

CUresult cuGraphLaunch(CUgraphExec exec, CUstream) {
  return alive(&g_graph_execs, reinterpret_cast<unsigned long long>(exec))
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphDestroy(CUgraph graph) {
  if (!retire(&g_graphs, reinterpret_cast<unsigned long long>(graph))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  rgpu_fake::count(rgpu_fake::kGraph, -1);
  return CUDA_SUCCESS;
}

CUresult cuGraphExecDestroy(CUgraphExec exec) {
  if (!retire(&g_graph_execs, reinterpret_cast<unsigned long long>(exec))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  rgpu_fake::count(rgpu_fake::kGraphExec, -1);
  return CUDA_SUCCESS;
}

}  // extern "C"
