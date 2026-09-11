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
#include <string>

#include <cuda.h>

namespace {

std::mutex g_mu;
// Device pointers are just host allocations here. Sizes are tracked so a copy
// running off the end is caught rather than corrupting the heap.
std::map<CUdeviceptr, size_t> g_allocs;

bool range_ok(CUdeviceptr p, size_t n) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Find the allocation containing p: the last one starting at or below it.
  auto it = g_allocs.upper_bound(p);
  if (it == g_allocs.begin()) return false;
  --it;
  return p >= it->first && p + n <= it->first + it->second;
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

// Contexts are opaque to everything above; a non-null token is enough.
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int, CUdevice dev) {
  if (!pctx || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  *pctx = reinterpret_cast<CUcontext>(0xC0FFEE00);
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  return ctx ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuCtxSynchronize(void) { return CUDA_SUCCESS; }

// The runtime binds a primary context per device, so the fake needs these too.
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  if (!pctx || dev != kFakeDevice) return CUDA_ERROR_INVALID_VALUE;
  *pctx = reinterpret_cast<CUcontext>(0xC0FFEE01);
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  return dev == kFakeDevice ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
  if (!free || !total) return CUDA_ERROR_INVALID_VALUE;
  *total = size_t(24) << 30;
  *free = size_t(20) << 30;
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext) { return CUDA_SUCCESS; }

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
  void* p = std::malloc(bytesize);
  if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
  auto d = reinterpret_cast<CUdeviceptr>(p);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_allocs[d] = bytesize;
  }
  *dptr = d;
  return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_allocs.erase(dptr) == 0) return CUDA_ERROR_INVALID_VALUE;
  }
  std::free(reinterpret_cast<void*>(dptr));
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
  *module = reinterpret_cast<CUmodule>(0x0D0100ull);
  return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule) { return CUDA_SUCCESS; }

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                             const char* name) {
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
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
  *stream = reinterpret_cast<CUstream>(0x57EA);
  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream stream) {
  return stream ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
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
  if (graph) *graph = reinterpret_cast<CUgraph>(0xC0FFEEull);
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
  if (graph != reinterpret_cast<CUgraph>(0xC0FFEEull)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (exec) *exec = reinterpret_cast<CUgraphExec>(0xE7E0ull);
  return CUDA_SUCCESS;
}

CUresult cuGraphLaunch(CUgraphExec exec, CUstream) {
  return exec == reinterpret_cast<CUgraphExec>(0xE7E0ull)
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphDestroy(CUgraph graph) {
  return graph == reinterpret_cast<CUgraph>(0xC0FFEEull)
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphExecDestroy(CUgraphExec exec) {
  return exec == reinterpret_cast<CUgraphExec>(0xE7E0ull)
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_VALUE;
}

}  // extern "C"
