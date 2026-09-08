// Hand-written parts of the libcuda.so.1 replacement.
//
// Everything here needs knowledge the generator does not have: how cudart
// discovers entry points, how kernel arguments are sized, and which objects
// live on the client rather than the server.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>

#include "client/rpc.h"
#include "common/generated/api_ids.h"
#include "common/internal_ids.h"

namespace {

// ---------------------------------------------------------------------------
// Kernel argument marshalling
// ---------------------------------------------------------------------------
//
// cuLaunchKernel's kernelParams is an array of N pointers with no sizes; the
// driver recovers the sizes from module metadata the shim cannot see. The
// server can, so it answers cuFuncGetParamInfo on our behalf and we cache the
// layout per function.

struct ParamSlot {
  uint64_t offset;
  uint64_t size;
};

std::mutex g_layout_mu;
std::map<CUfunction, std::vector<ParamSlot>> g_layouts;

bool fetch_layout(CUfunction f, std::vector<ParamSlot>* out) {
  {
    std::lock_guard<std::mutex> lk(g_layout_mu);
    auto it = g_layouts.find(f);
    if (it != g_layouts.end()) {
      *out = it->second;
      return true;
    }
  }

  rgpu::Buffer req;
  req.put<uint64_t>(reinterpret_cast<uint64_t>(f));
  rgpu::Buffer rsp;
  if (rgpu::call(rgpu::API_rgpu_param_layout, req, &rsp) != CUDA_SUCCESS) {
    return false;
  }
  uint32_t n = 0;
  if (!rsp.get(&n)) return false;
  std::vector<ParamSlot> slots(n);
  for (uint32_t i = 0; i < n; i++) {
    if (!rsp.get(&slots[i].offset) || !rsp.get(&slots[i].size)) return false;
  }

  std::lock_guard<std::mutex> lk(g_layout_mu);
  g_layouts[f] = slots;
  *out = slots;
  return true;
}

// Packs kernelParams into the device-side layout. The server then hands the
// blob to the driver through the extra/CU_LAUNCH_PARAM_BUFFER_POINTER path,
// which takes an explicit size.
bool pack_args(CUfunction f, void** kernelParams, std::vector<uint8_t>* blob) {
  std::vector<ParamSlot> slots;
  if (!fetch_layout(f, &slots)) return false;

  size_t total = 0;
  for (const auto& s : slots) total = std::max<size_t>(total, s.offset + s.size);
  blob->assign(total, 0);
  for (size_t i = 0; i < slots.size(); i++) {
    if (!kernelParams[i]) return false;
    std::memcpy(blob->data() + slots[i].offset, kernelParams[i], slots[i].size);
  }
  return true;
}

// Reads an already-packed argument buffer out of the extra[] array.
bool extract_extra(void** extra, const uint8_t** data, size_t* size) {
  const void* buf = nullptr;
  const size_t* sz = nullptr;
  for (size_t i = 0; extra && extra[i] != CU_LAUNCH_PARAM_END; i++) {
    if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buf = extra[++i];
    } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      sz = static_cast<const size_t*>(extra[++i]);
    } else {
      return false;  // unknown tag
    }
  }
  if (!buf || !sz) return false;
  *data = static_cast<const uint8_t*>(buf);
  *size = *sz;
  return true;
}

CUresult launch_common(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                       unsigned bx, unsigned by, unsigned bz,
                       unsigned shmem, CUstream stream,
                       void** kernelParams, void** extra) {
  std::vector<uint8_t> packed;
  const uint8_t* args = nullptr;
  size_t args_len = 0;

  if (extra) {
    if (!extract_extra(extra, &args, &args_len)) return CUDA_ERROR_INVALID_VALUE;
  } else if (kernelParams) {
    if (!pack_args(f, kernelParams, &packed)) return CUDA_ERROR_INVALID_VALUE;
    args = packed.data();
    args_len = packed.size();
  }

  rgpu::Buffer req;
  req.put<uint64_t>(reinterpret_cast<uint64_t>(f));
  req.put<uint32_t>(gx);
  req.put<uint32_t>(gy);
  req.put<uint32_t>(gz);
  req.put<uint32_t>(bx);
  req.put<uint32_t>(by);
  req.put<uint32_t>(bz);
  req.put<uint32_t>(shmem);
  req.put<uint64_t>(reinterpret_cast<uint64_t>(stream));
  req.put_sized(args, args_len);

  rgpu::Buffer rsp;
  return rgpu::call(rgpu::API_rgpu_launch, req, &rsp);
}

// ---------------------------------------------------------------------------
// Host allocations
// ---------------------------------------------------------------------------
//
// Pinned host memory is only meaningful to the local driver, and there is no
// local driver. We hand back ordinary aligned host memory: correct, just
// without the DMA benefit, which a network transfer has already erased.

std::mutex g_host_mu;
std::set<void*> g_host_allocs;

CUresult host_alloc(void** pp, size_t size) {
  if (!pp) return CUDA_ERROR_INVALID_VALUE;
  void* p = nullptr;
  if (::posix_memalign(&p, 4096, size ? size : 1) != 0) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  {
    std::lock_guard<std::mutex> lk(g_host_mu);
    g_host_allocs.insert(p);
  }
  *pp = p;
  return CUDA_SUCCESS;
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry point table
// ---------------------------------------------------------------------------
//
// cudart resolves driver functions through cuGetProcAddress rather than dlsym
// (CUDA 11.3 and later), so exporting the symbols is not enough on its own:
// this table has to answer for every name it asks about.

namespace {

struct Entry {
  const char* name;
  void* fn;
};

const Entry kEntries[] = {
#define RGPU_ENTRY(name, fn) {name, reinterpret_cast<void*>(&fn)},
#include "client/generated/proc_table.inc"
#undef RGPU_ENTRY
};

void* find_entry(const char* symbol) {
  if (!symbol) return nullptr;
  for (const auto& e : kEntries) {
    if (std::strcmp(e.name, symbol) == 0) return e.fn;
  }
  return nullptr;
}

// Names we were asked for and could not supply. Logged once each; this is the
// discovery mechanism for what still needs implementing.
void note_missing(const char* symbol, int version) {
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lk(mu);
  if (seen.insert(symbol).second) {
    rgpu::log("MISSING entry point %s (requested version %d)", symbol, version);
  }
}

}  // namespace

extern "C" {

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult* symbolStatus) {
  if (!pfn) return CUDA_ERROR_INVALID_VALUE;
  // The per-thread-default-stream flag selects the _ptsz entry points, which
  // treat the default stream as per-thread rather than legacy. We only carry
  // the regular variants, so serve those and say so once: the difference only
  // shows up in code that relies on the default stream instead of an explicit
  // one, which PyTorch does not.
  if (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {
    static std::once_flag once;
    std::call_once(once, [] {
      rgpu::log("note: per-thread default stream requested; serving the "
                "legacy default stream variants instead");
    });
  }
  void* fn = find_entry(symbol);
  *pfn = fn;
  if (symbolStatus) {
    *symbolStatus = fn ? CU_GET_PROC_ADDRESS_SUCCESS
                       : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  }
  if (!fn) {
    note_missing(symbol, cudaVersion);
    return CUDA_ERROR_NOT_FOUND;
  }
  return CUDA_SUCCESS;
}

// The pre-CUDA-12 form, without the status out-parameter. cuda.h defines
// cuGetProcAddress as an alias for the _v2 name, so the macro has to go before
// we can define the older entry point under its own name.
#undef cuGetProcAddress
CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                          cuuint64_t flags) {
  return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}

// Error strings are static data in the real driver; we answer locally so the
// caller gets a valid pointer without a round trip.
CUresult cuGetErrorString(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  switch (error) {
    case CUDA_SUCCESS: *pStr = "no error"; break;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "invalid argument"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "out of memory"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "initialization error"; break;
    case CUDA_ERROR_NO_DEVICE: *pStr = "no CUDA-capable device is detected"; break;
    case CUDA_ERROR_INVALID_DEVICE: *pStr = "invalid device ordinal"; break;
    case CUDA_ERROR_NOT_FOUND: *pStr = "named symbol not found"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "operation not supported"; break;
    default: *pStr = "unknown error"; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  switch (error) {
    case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
    case CUDA_ERROR_NO_DEVICE: *pStr = "CUDA_ERROR_NO_DEVICE"; break;
    case CUDA_ERROR_NOT_FOUND: *pStr = "CUDA_ERROR_NOT_FOUND"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *pStr = "CUDA_ERROR_UNKNOWN"; break;
  }
  return CUDA_SUCCESS;
}

// Reported as the server's driver version, capped at what we were built
// against, since we cannot forward entry points we do not know about.
CUresult cuDriverGetVersion(int* driverVersion) {
  if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
  static int cached = 0;
  if (cached == 0) {
    rgpu::Buffer req, rsp;
    if (rgpu::call(rgpu::API_rgpu_hello, req, &rsp) == CUDA_SUCCESS) {
      int remote = 0;
      if (rsp.get(&remote) && remote > 0) {
        cached = remote < CUDA_VERSION ? remote : CUDA_VERSION;
      }
    }
    if (cached == 0) cached = CUDA_VERSION;
  }
  *driverVersion = cached;
  return CUDA_SUCCESS;
}

// cuGetExportTable hands back a table of undocumented internal driver
// function pointers, keyed by UUID. cudart calls it right after cuInit and
// uses it extensively.
//
// It cannot be forwarded: the pointers would be addresses in the server
// process, and the client would call straight into them. The Cricket paper
// reaches the same conclusion, that a driver-level layer cannot carry the
// stock runtime on top of it for exactly this reason.
//
// We log which table was asked for and refuse. RGPU_EXPORT_TABLE_ERROR lets
// the refusal code be changed while probing what cudart tolerates.
CUresult cuGetExportTable(const void** ppExportTable,
                          const CUuuid* pExportTableId) {
  if (!ppExportTable || !pExportTableId) return CUDA_ERROR_INVALID_VALUE;
  char uuid[64];
  const unsigned char* b =
      reinterpret_cast<const unsigned char*>(pExportTableId->bytes);
  std::snprintf(uuid, sizeof(uuid),
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                "%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  rgpu::log("cuGetExportTable requested for %s", uuid);
  *ppExportTable = nullptr;

  const char* mode = std::getenv("RGPU_EXPORT_TABLE_ERROR");
  if (mode) return static_cast<CUresult>(std::atoi(mode));
  return CUDA_ERROR_NOT_FOUND;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void** kernelParams, void** extra) {
  return launch_common(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void** kernelParams) {
  // Cooperative launch has stricter co-residency guarantees, which the remote
  // driver still provides; only the marshalling differs.
  return launch_common(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream, kernelParams,
                       nullptr);
}

CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) {
  return host_alloc(pp, bytesize);
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
  // CU_MEMHOSTALLOC_DEVICEMAP would promise a device pointer for this host
  // memory, which cannot exist across a network.
  if (Flags & CU_MEMHOSTALLOC_DEVICEMAP) {
    return rgpu::unimplemented("cuMemHostAlloc(DEVICEMAP)",
                               "zero-copy host mapping cannot span a network");
  }
  return host_alloc(pp, bytesize);
}

CUresult cuMemFreeHost(void* p) {
  if (!p) return CUDA_SUCCESS;
  {
    std::lock_guard<std::mutex> lk(g_host_mu);
    if (g_host_allocs.erase(p) == 0) return CUDA_ERROR_INVALID_VALUE;
  }
  ::free(p);
  return CUDA_SUCCESS;
}

// Registration only affects DMA behaviour for a local driver. There is none,
// so accepting it changes nothing observable.
CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) {
  (void)p; (void)bytesize;
  if (Flags & CU_MEMHOSTREGISTER_DEVICEMAP) {
    return rgpu::unimplemented("cuMemHostRegister(DEVICEMAP)",
                               "zero-copy host mapping cannot span a network");
  }
  return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void* p) {
  (void)p;
  return CUDA_SUCCESS;
}

}  // extern "C"
