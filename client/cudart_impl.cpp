// The CUDA runtime API, translated into driver API calls.
//
// This exists because stock libcudart cannot run on top of a remoted driver:
// it calls cuGetExportTable immediately after cuInit and refuses to initialise
// without a table of undocumented internal driver function pointers, which
// cannot be forwarded because they are addresses in the server's process. See
// the design document.
//
// So the runtime stops here instead. Everything below is expressed in driver
// API calls, which our libcuda.so.1 remotes. The server stays purely
// driver-level and unchanged.
//
// Definitions here are strong and override the weak, logging ones in
// client/generated/cudart_stubs.cpp.

#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "client/rpc.h"

namespace {

// ---------------------------------------------------------------------------
// Error state
// ---------------------------------------------------------------------------
//
// The runtime remembers the last error per thread and clears it on read; the
// driver does not, so we keep that state here.

thread_local cudaError_t t_last_error = cudaSuccess;

cudaError_t from_cu(CUresult r) {
  switch (r) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_IMAGE: return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorDeviceUninitialized;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorSymbolNotFound;
    case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
    case CUDA_ERROR_ILLEGAL_ADDRESS: return cudaErrorIllegalAddress;
    case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_LAUNCH_TIMEOUT: return cudaErrorLaunchTimeout;
    case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
    case CUDA_ERROR_FILE_NOT_FOUND: return cudaErrorFileNotFound;
    default: return cudaErrorUnknown;
  }
}

// Records the error the runtime should report from cudaGetLastError.
cudaError_t record(cudaError_t e) {
  if (e != cudaSuccess) t_last_error = e;
  return e;
}

cudaError_t record_cu(CUresult r) { return record(from_cu(r)); }

// ---------------------------------------------------------------------------
// Device and context
// ---------------------------------------------------------------------------
//
// The runtime binds a primary context to each device lazily on first use and
// keeps a current device per thread. We reproduce that here.

std::mutex g_ctx_mu;
std::map<int, CUcontext> g_primary;  // device ordinal -> retained context
bool g_inited = false;

thread_local int t_device = 0;
thread_local bool t_ctx_set = false;

CUresult ensure_init() {
  std::lock_guard<std::mutex> lk(g_ctx_mu);
  if (g_inited) return CUDA_SUCCESS;
  CUresult r = cuInit(0);
  if (r == CUDA_SUCCESS) g_inited = true;
  return r;
}

// Retains and makes current the primary context for this thread's device.
CUresult ensure_context() {
  CUresult r = ensure_init();
  if (r != CUDA_SUCCESS) return r;
  if (t_ctx_set) return CUDA_SUCCESS;

  CUcontext ctx = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    auto it = g_primary.find(t_device);
    if (it != g_primary.end()) {
      ctx = it->second;
    } else {
      CUdevice dev;
      r = cuDeviceGet(&dev, t_device);
      if (r != CUDA_SUCCESS) return r;
      r = cuDevicePrimaryCtxRetain(&ctx, dev);
      if (r != CUDA_SUCCESS) return r;
      g_primary[t_device] = ctx;
    }
  }
  r = cuCtxSetCurrent(ctx);
  if (r == CUDA_SUCCESS) t_ctx_set = true;
  return r;
}

// ---------------------------------------------------------------------------
// Allocation tracking
// ---------------------------------------------------------------------------
//
// cudaMemcpyDefault asks us to infer from the pointer whether each side is
// host or device. With remoting we cannot probe a device pointer, since it is
// an address in the server's process, so we remember what we handed out.

std::mutex g_alloc_mu;
std::map<CUdeviceptr, size_t> g_device_allocs;

void note_alloc(CUdeviceptr p, size_t n) {
  std::lock_guard<std::mutex> lk(g_alloc_mu);
  g_device_allocs[p] = n;
}

void forget_alloc(CUdeviceptr p) {
  std::lock_guard<std::mutex> lk(g_alloc_mu);
  g_device_allocs.erase(p);
}

bool is_device_ptr(const void* p) {
  if (!p) return false;
  auto v = reinterpret_cast<CUdeviceptr>(p);
  std::lock_guard<std::mutex> lk(g_alloc_mu);
  auto it = g_device_allocs.upper_bound(v);
  if (it == g_device_allocs.begin()) return false;
  --it;
  return v >= it->first && v < it->first + it->second;
}

// Resolves cudaMemcpyDefault to a concrete direction.
cudaMemcpyKind resolve_kind(void* dst, const void* src, cudaMemcpyKind kind) {
  if (kind != cudaMemcpyDefault) return kind;
  const bool d = is_device_ptr(dst);
  const bool s = is_device_ptr(src);
  if (d && s) return cudaMemcpyDeviceToDevice;
  if (d) return cudaMemcpyHostToDevice;
  if (s) return cudaMemcpyDeviceToHost;
  return cudaMemcpyHostToHost;
}

CUresult do_memcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind,
                   CUstream stream, bool async) {
  switch (resolve_kind(dst, src, kind)) {
    case cudaMemcpyHostToDevice:
      return async ? cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src,
                                       n, stream)
                   : cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, n);
    case cudaMemcpyDeviceToHost:
      return async ? cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(
                                                const_cast<void*>(src)),
                                       n, stream)
                   : cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(
                                           const_cast<void*>(src)), n);
    case cudaMemcpyDeviceToDevice:
      return async ? cuMemcpyDtoDAsync(
                         reinterpret_cast<CUdeviceptr>(dst),
                         reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)),
                         n, stream)
                   : cuMemcpyDtoD(
                         reinterpret_cast<CUdeviceptr>(dst),
                         reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)),
                         n);
    case cudaMemcpyHostToHost:
      // Both sides are ours; no reason to involve the GPU at all.
      std::memcpy(dst, src, n);
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

// ---------------------------------------------------------------------------
// Module and kernel registration
// ---------------------------------------------------------------------------
//
// nvcc emits calls to these hidden entry points in every translation unit that
// contains device code. They are how a host function pointer comes to stand
// for a kernel, which is what cudaLaunchKernel is given.
//
// Modules load lazily. PyTorch's device code runs to hundreds of megabytes and
// most of it is never used in a given process, so shipping every fatbin at
// import time would be slow for no benefit.

struct Module {
  const void* image = nullptr;  // owned by the calling library, lives as long
  CUmodule loaded = nullptr;
  bool tried = false;
};

struct Kernel {
  Module* module = nullptr;
  std::string device_name;
  CUfunction fn = nullptr;
};

std::mutex g_reg_mu;
std::vector<Module*> g_modules;
std::map<const void*, Kernel> g_kernels;   // host function pointer -> kernel
std::map<const void*, std::string> g_vars;  // host variable -> device symbol

// Loads a module the first time one of its kernels is launched.
CUresult ensure_module(Module* m) {
  if (m->loaded) return CUDA_SUCCESS;
  if (m->tried) return CUDA_ERROR_INVALID_IMAGE;
  m->tried = true;
  // Loading ships the whole image to the server. The math libraries carry very
  // large fatbins, so the size is worth seeing: it is the main cost of the
  // first launch of any kernel from a given library.
  const size_t bytes = rgpu::image_size(m->image);
  if (bytes > (1u << 20)) {
    rgpu::log("shipping a %.1f MiB module image to the server",
              double(bytes) / (1 << 20));
  }
  CUresult r = cuModuleLoadData(&m->loaded, m->image);
  if (r != CUDA_SUCCESS) {
    rgpu::log("cuModuleLoadData failed (%d) while loading a registered fatbin",
              r);
    m->loaded = nullptr;
  }
  return r;
}

CUresult resolve_kernel(const void* host_fn, CUfunction* out) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = g_kernels.find(host_fn);
  if (it == g_kernels.end()) {
    rgpu::log("launch of an unregistered host function %p", host_fn);
    return CUDA_ERROR_NOT_FOUND;
  }
  Kernel& k = it->second;
  if (k.fn) {
    *out = k.fn;
    return CUDA_SUCCESS;
  }
  CUresult r = ensure_module(k.module);
  if (r != CUDA_SUCCESS) return r;
  r = cuModuleGetFunction(&k.fn, k.module->loaded, k.device_name.c_str());
  if (r != CUDA_SUCCESS) {
    rgpu::log("cuModuleGetFunction(%s) failed (%d)", k.device_name.c_str(), r);
    return r;
  }
  *out = k.fn;
  return CUDA_SUCCESS;
}

// The wrapper nvcc passes to __cudaRegisterFatBinary. Its `data` field points
// at the fatbin image proper, whose length the shim works out from its header.
struct FatBinWrapper {
  int magic;
  int version;
  const unsigned long long* data;
  void* filename_or_fatbins;
};
constexpr int kFatBinWrapperMagic = 0x466243b1;

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------
//
// The <<<>>> syntax lowers to a push of the configuration, then a call to the
// kernel's host stub, which pops it back off.

struct CallConfig {
  dim3 grid;
  dim3 block;
  size_t shared;
  cudaStream_t stream;
};
thread_local std::vector<CallConfig> t_configs;

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Version and error reporting
// ---------------------------------------------------------------------------

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
  if (!runtimeVersion) return record(cudaErrorInvalidValue);
  *runtimeVersion = CUDART_VERSION;
  return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int* driverVersion) {
  if (!driverVersion) return record(cudaErrorInvalidValue);
  return record_cu(cuDriverGetVersion(driverVersion));
}

cudaError_t cudaGetLastError(void) {
  cudaError_t e = t_last_error;
  t_last_error = cudaSuccess;
  return e;
}

cudaError_t cudaPeekAtLastError(void) { return t_last_error; }

const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInitializationError: return "initialization error";
    case cudaErrorNoDevice: return "no CUDA-capable device is detected";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorNotSupported: return "operation not supported";
    case cudaErrorNotReady: return "device not ready";
    case cudaErrorSymbolNotFound: return "named symbol not found";
    case cudaErrorInvalidResourceHandle: return "invalid resource handle";
    case cudaErrorIllegalAddress: return "an illegal memory access was encountered";
    case cudaErrorLaunchFailure: return "unspecified launch failure";
    default: return "unknown error";
  }
}

const char* cudaGetErrorName(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorNoDevice: return "cudaErrorNoDevice";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    default: return "cudaErrorUnknown";
  }
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

cudaError_t cudaGetDeviceCount(int* count) {
  if (!count) return record(cudaErrorInvalidValue);
  CUresult r = ensure_init();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuDeviceGetCount(count));
}

cudaError_t cudaGetDevice(int* device) {
  if (!device) return record(cudaErrorInvalidValue);
  *device = t_device;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  CUresult r = ensure_init();
  if (r != CUDA_SUCCESS) return record_cu(r);
  int count = 0;
  r = cuDeviceGetCount(&count);
  if (r != CUDA_SUCCESS) return record_cu(r);
  if (device < 0 || device >= count) return record(cudaErrorInvalidDevice);
  if (device != t_device) {
    t_device = device;
    t_ctx_set = false;  // the new device needs its own context made current
  }
  return record_cu(ensure_context());
}

// cudaDeviceAttr and CU_DEVICE_ATTRIBUTE_* share their numeric values by
// design, so the enum passes straight through.
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  if (!value) return record(cudaErrorInvalidValue);
  CUresult r = ensure_init();
  if (r != CUDA_SUCCESS) return record_cu(r);
  CUdevice dev;
  r = cuDeviceGet(&dev, device);
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuDeviceGetAttribute(
      value, static_cast<CUdevice_attribute>(attr), dev));
}

cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
  if (!prop) return record(cudaErrorInvalidValue);
  CUresult r = ensure_init();
  if (r != CUDA_SUCCESS) return record_cu(r);
  CUdevice dev;
  r = cuDeviceGet(&dev, device);
  if (r != CUDA_SUCCESS) return record_cu(r);

  std::memset(prop, 0, sizeof(*prop));
  cuDeviceGetName(prop->name, sizeof(prop->name), dev);
  size_t total = 0;
  cuDeviceTotalMem(&total, dev);
  prop->totalGlobalMem = total;
  cuDeviceGetUuid(reinterpret_cast<CUuuid*>(&prop->uuid), dev);

  // Fetch one attribute into a struct field, ignoring failures so a single
  // unsupported attribute cannot fail the whole query.
  auto attr = [&](CUdevice_attribute a, int* dst) {
    int v = 0;
    if (cuDeviceGetAttribute(&v, a, dev) == CUDA_SUCCESS) *dst = v;
  };
  attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &prop->major);
  attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &prop->minor);
  attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, &prop->multiProcessorCount);
  attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &prop->maxThreadsPerBlock);
  attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE, &prop->warpSize);
  attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, &prop->clockRate);
  attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, &prop->memoryClockRate);
  attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, &prop->memoryBusWidth);
  attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, &prop->l2CacheSize);
  attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
       &prop->maxThreadsPerMultiProcessor);
  attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, &prop->regsPerBlock);
  attr(CU_DEVICE_ATTRIBUTE_INTEGRATED, &prop->integrated);
  attr(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, &prop->canMapHostMemory);
  attr(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, &prop->computeMode);
  attr(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, &prop->concurrentKernels);
  attr(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, &prop->ECCEnabled);
  attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, &prop->pciBusID);
  attr(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, &prop->pciDeviceID);
  attr(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, &prop->pciDomainID);
  attr(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, &prop->unifiedAddressing);
  attr(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, &prop->managedMemory);
  attr(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, &prop->cooperativeLaunch);
  attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
       &prop->maxBlocksPerMultiProcessor);

  int v = 0;
  if (cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                           dev) == CUDA_SUCCESS) {
    prop->sharedMemPerBlock = static_cast<size_t>(v);
  }
  if (cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev) ==
      CUDA_SUCCESS) {
    prop->sharedMemPerMultiprocessor = static_cast<size_t>(v);
  }
  if (cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev) ==
      CUDA_SUCCESS) {
    prop->totalConstMem = static_cast<size_t>(v);
  }
  attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, &prop->maxThreadsDim[0]);
  attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, &prop->maxThreadsDim[1]);
  attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, &prop->maxThreadsDim[2]);
  attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, &prop->maxGridSize[0]);
  attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, &prop->maxGridSize[1]);
  attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, &prop->maxGridSize[2]);
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuCtxSynchronize());
}

cudaError_t cudaDeviceReset(void) {
  // Releasing the primary contexts is the observable part; the server drops
  // the rest when the connection closes.
  std::lock_guard<std::mutex> lk(g_ctx_mu);
  for (auto& kv : g_primary) {
    CUdevice dev;
    if (cuDeviceGet(&dev, kv.first) == CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(dev);
    }
  }
  g_primary.clear();
  t_ctx_set = false;
  return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuMemGetInfo(free, total));
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  if (!devPtr) return record(cudaErrorInvalidValue);
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  CUdeviceptr d = 0;
  // The driver rejects a zero-byte allocation; the runtime returns a pointer
  // that can be freed, so ask for one byte.
  r = cuMemAlloc(&d, size ? size : 1);
  if (r != CUDA_SUCCESS) return record_cu(r);
  note_alloc(d, size ? size : 1);
  *devPtr = reinterpret_cast<void*>(d);
  return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
  if (!devPtr) return cudaSuccess;  // freeing null is legal
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  auto d = reinterpret_cast<CUdeviceptr>(devPtr);
  r = cuMemFree(d);
  if (r == CUDA_SUCCESS) forget_alloc(d);
  return record_cu(r);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuMemAllocHost(ptr, size ? size : 1));
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuMemHostAlloc(pHost, size ? size : 1, flags));
}

cudaError_t cudaFreeHost(void* ptr) {
  if (!ptr) return cudaSuccess;
  return record_cu(cuMemFreeHost(ptr));
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                       cudaMemcpyKind kind) {
  if (count == 0) return cudaSuccess;
  if (!dst || !src) return record(cudaErrorInvalidValue);
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(do_memcpy(dst, src, count, kind, nullptr, false));
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
  if (count == 0) return cudaSuccess;
  if (!dst || !src) return record(cudaErrorInvalidValue);
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(do_memcpy(dst, src, count, kind,
                             reinterpret_cast<CUstream>(stream), true));
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  if (count == 0) return cudaSuccess;
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuMemsetD8(reinterpret_cast<CUdeviceptr>(devPtr),
                              static_cast<unsigned char>(value), count));
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count,
                            cudaStream_t stream) {
  if (count == 0) return cudaSuccess;
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(devPtr),
                                   static_cast<unsigned char>(value), count,
                                   reinterpret_cast<CUstream>(stream)));
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuStreamCreate(reinterpret_cast<CUstream*>(pStream),
                                  CU_STREAM_DEFAULT));
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream,
                                      unsigned int flags) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(
      cuStreamCreate(reinterpret_cast<CUstream*>(pStream), flags));
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream,
                                         unsigned int flags, int priority) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuStreamCreateWithPriority(
      reinterpret_cast<CUstream*>(pStream), flags, priority));
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return record_cu(cuStreamDestroy(reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuStreamSynchronize(reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return record_cu(cuStreamQuery(reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
  return record_cu(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),
                                     reinterpret_cast<CUevent>(event), flags));
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

cudaError_t cudaEventCreate(cudaEvent_t* event) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuEventCreate(reinterpret_cast<CUevent*>(event),
                                 CU_EVENT_DEFAULT));
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  return record_cu(cuEventCreate(reinterpret_cast<CUevent*>(event), flags));
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return record_cu(cuEventDestroy(reinterpret_cast<CUevent>(event)));
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return record_cu(cuEventRecord(reinterpret_cast<CUevent>(event),
                                 reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return record_cu(cuEventSynchronize(reinterpret_cast<CUevent>(event)));
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return record_cu(cuEventQuery(reinterpret_cast<CUevent>(event)));
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start,
                                 cudaEvent_t end) {
  return record_cu(cuEventElapsedTime(ms, reinterpret_cast<CUevent>(start),
                                      reinterpret_cast<CUevent>(end)));
}

// ---------------------------------------------------------------------------
// Kernel registration and launch
// ---------------------------------------------------------------------------

void** __cudaRegisterFatBinary(void* fatCubin) {
  auto* w = static_cast<FatBinWrapper*>(fatCubin);
  const void* image = nullptr;
  if (w && w->magic == kFatBinWrapperMagic) {
    image = w->data;
  } else {
    // Some builds hand over the image directly rather than a wrapper.
    image = fatCubin;
  }
  auto* m = new Module();
  m->image = image;
  std::lock_guard<std::mutex> lk(g_reg_mu);
  g_modules.push_back(m);
  // The handle is opaque to the caller; it only ever comes back to us.
  return reinterpret_cast<void**>(m);
}

void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
  (void)fatCubinHandle;  // nothing to finalise; modules load on first launch
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
  auto* m = reinterpret_cast<Module*>(fatCubinHandle);
  if (!m) return;
  std::lock_guard<std::mutex> lk(g_reg_mu);
  if (m->loaded) cuModuleUnload(m->loaded);
  for (auto it = g_modules.begin(); it != g_modules.end(); ++it) {
    if (*it == m) {
      g_modules.erase(it);
      break;
    }
  }
  // Kernels referring to this module are gone with it.
  for (auto it = g_kernels.begin(); it != g_kernels.end();) {
    it = (it->second.module == m) ? g_kernels.erase(it) : std::next(it);
  }
  delete m;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
                            char* deviceFun, const char* deviceName,
                            int thread_limit, uint3* tid, uint3* bid,
                            dim3* bDim, dim3* gDim, int* wSize) {
  (void)deviceFun; (void)thread_limit; (void)tid; (void)bid;
  (void)bDim; (void)gDim; (void)wSize;
  auto* m = reinterpret_cast<Module*>(fatCubinHandle);
  if (!m || !hostFun || !deviceName) return;
  Kernel k;
  k.module = m;
  k.device_name = deviceName;
  std::lock_guard<std::mutex> lk(g_reg_mu);
  g_kernels[static_cast<const void*>(hostFun)] = k;
}

void __cudaRegisterVar(void** fatCubinHandle, char* hostVar,
                       char* deviceAddress, const char* deviceName, int ext,
                       size_t size, int constant, int global) {
  (void)fatCubinHandle; (void)deviceAddress; (void)ext; (void)size;
  (void)constant; (void)global;
  if (!hostVar || !deviceName) return;
  std::lock_guard<std::mutex> lk(g_reg_mu);
  g_vars[static_cast<const void*>(hostVar)] = deviceName;
}

// The rest of the family nvcc can emit. A library referencing one we do not
// define would fail to load at all, so they exist even where there is nothing
// useful to do: textures and surfaces are not reachable remotely, and managed
// variables need managed memory, which cannot span a network.
void __cudaRegisterManagedVar(void** fatCubinHandle, void** hostVarPtrAddress,
                              char* deviceAddress, const char* deviceName,
                              int ext, size_t size, int constant, int global) {
  (void)fatCubinHandle; (void)hostVarPtrAddress; (void)deviceAddress;
  (void)ext; (void)size; (void)constant; (void)global;
  rgpu::unimplemented_rt("__cudaRegisterManagedVar");
  (void)deviceName;
}

void __cudaRegisterHostVar(void** fatCubinHandle, const char* deviceName,
                           char* hostVar, size_t size) {
  (void)fatCubinHandle; (void)deviceName; (void)hostVar; (void)size;
}

void __cudaRegisterTexture(void** fatCubinHandle, const void* hostVar,
                           const void** deviceAddress, const char* deviceName,
                           int dim, int norm, int ext) {
  (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
  (void)dim; (void)norm; (void)ext;
  rgpu::unimplemented_rt("__cudaRegisterTexture");
}

void __cudaRegisterSurface(void** fatCubinHandle, const void* hostVar,
                           const void** deviceAddress, const char* deviceName,
                           int dim, int ext) {
  (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
  (void)dim; (void)ext;
  rgpu::unimplemented_rt("__cudaRegisterSurface");
}

char __cudaInitModule(void** fatCubinHandle) {
  (void)fatCubinHandle;
  return 1;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem, void* stream) {
  t_configs.push_back({gridDim, blockDim, sharedMem,
                       static_cast<cudaStream_t>(stream)});
  return 0;  // zero means the configuration was accepted
}

cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                       size_t* sharedMem, void* stream) {
  if (t_configs.empty()) return cudaErrorInvalidConfiguration;
  const CallConfig c = t_configs.back();
  t_configs.pop_back();
  if (gridDim) *gridDim = c.grid;
  if (blockDim) *blockDim = c.block;
  if (sharedMem) *sharedMem = c.shared;
  if (stream) *static_cast<cudaStream_t*>(stream) = c.stream;
  return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem,
                             cudaStream_t stream) {
  CUresult r = ensure_context();
  if (r != CUDA_SUCCESS) return record_cu(r);
  CUfunction fn = nullptr;
  r = resolve_kernel(func, &fn);
  if (r != CUDA_SUCCESS) return record_cu(r);
  // Argument sizes are recovered by the driver shim, which asks the server for
  // the kernel's parameter layout. Passing args straight through reuses that.
  return record_cu(cuLaunchKernel(fn, gridDim.x, gridDim.y, gridDim.z,
                                  blockDim.x, blockDim.y, blockDim.z,
                                  static_cast<unsigned>(sharedMem),
                                  reinterpret_cast<CUstream>(stream), args,
                                  nullptr));
}

}  // extern "C"
