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

// No kernels without a GPU. Reporting this rather than pretending to succeed
// keeps the fake honest: a test that needs a real launch must fail here.
CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, unsigned int,
                        CUstream, void**, void**) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuFuncGetParamInfo(CUfunction, size_t, size_t*, size_t*) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
