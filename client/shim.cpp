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

// Every global below that a CUDA call can reach is allocated and never
// destroyed. Threads keep calling in while the process exits - a thread's
// late thread-local destructors free device memory, say - and a mutex or
// container that static destruction has already taken down is undefined
// behaviour (on libc++ a destroyed mutex throws, from places that cannot).

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

auto& g_layout_mu = *new std::mutex();
auto& g_layouts = *new std::map<CUfunction, std::vector<ParamSlot>>();

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

  // A launch returns nothing and its effect is only visible at the next
  // synchronization, so it goes without a reply. This is the round trip that
  // matters: it is the call a real workload makes most.
  return rgpu::call_async(rgpu::API_rgpu_launch, req);
}

// ---------------------------------------------------------------------------
// Host allocations
// ---------------------------------------------------------------------------
//
// Pinned host memory is only meaningful to the local driver, and there is no
// local driver. We hand back ordinary aligned host memory: correct, just
// without the DMA benefit, which a network transfer has already erased.

auto& g_host_mu = *new std::mutex();
auto& g_host_allocs = *new std::set<void*>();

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
  static auto& mu = *new std::mutex();
  static auto& seen = *new std::set<std::string>();
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

// --- device allocations ----------------------------------------------------
//
// Triton's kernel launcher asks the driver for the device pointer behind every
// argument, on every launch: 144 round trips per compiled ResNet-18 inference,
// which was 63% of them. For memory we allocated ourselves the answer is the
// pointer it was given, and we are the ones who allocated it.
//
// Only that question is answered here. Anything else about a pointer, and any
// pointer we do not recognise, goes to the server as before.

namespace {

auto& g_alloc_mu = *new std::mutex();
auto& g_allocs = *new std::map<CUdeviceptr, size_t>();  // base -> size

bool is_ours(CUdeviceptr p) {
  std::lock_guard<std::mutex> lk(g_alloc_mu);
  if (g_allocs.empty()) return false;
  auto it = g_allocs.upper_bound(p);
  if (it == g_allocs.begin()) return false;
  --it;
  return p >= it->first && p < it->first + it->second;
}

}  // namespace

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  // Same frame the generated stub sent: a presence byte, then the size.
  rgpu::Buffer req, rsp;
  req.put<uint8_t>(1);
  req.put<size_t>(bytesize);
  CUresult r = rgpu::call(rgpu::API_cuMemAlloc_v2, req, &rsp);
  if (r != CUDA_SUCCESS) return r;
  CUdeviceptr p = 0;
  if (!rsp.get(&p)) return CUDA_ERROR_UNKNOWN;
  *dptr = p;
  if (bytesize) {
    std::lock_guard<std::mutex> lk(g_alloc_mu);
    g_allocs[*dptr] = bytesize;
  }
  return r;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  {
    std::lock_guard<std::mutex> lk(g_alloc_mu);
    g_allocs.erase(dptr);
  }
  rgpu::Buffer req, rsp;
  req.put<CUdeviceptr>(dptr);
  return rgpu::call(rgpu::API_cuMemFree_v2, req, &rsp);
}

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute,
                               CUdeviceptr ptr) {
  if (!data) return CUDA_ERROR_INVALID_VALUE;
  if (attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER && is_ours(ptr)) {
    // A plain device allocation is its own device pointer.
    *static_cast<CUdeviceptr*>(data) = ptr;
    return CUDA_SUCCESS;
  }
  // Same frame the generated stub sent: the output buffer's presence and
  // size, then the attribute and the pointer.
  const uint64_t size = rgpu::pointer_attr_size(attribute);
  if (size == 0) {
    // P2P tokens and mempool handles among others: refused, rather than
    // guessing a size and moving the wrong number of bytes.
    rgpu::log("unknown pointer attribute %u; refusing rather than guessing a size",
              static_cast<unsigned>(attribute));
    return CUDA_ERROR_INVALID_VALUE;
  }
  rgpu::Buffer req, rsp;
  req.put<uint8_t>(1);
  req.put<uint64_t>(size);
  req.put<CUpointer_attribute>(attribute);
  req.put<CUdeviceptr>(ptr);
  CUresult r = rgpu::call(rgpu::API_cuPointerGetAttribute, req, &rsp);
  if (r != CUDA_SUCCESS) return r;
  const uint8_t* bytes = nullptr;
  size_t n = 0;
  if (!rsp.get_sized(&bytes, &n)) return CUDA_ERROR_UNKNOWN;
  std::memcpy(data, bytes, n < size ? n : size);
  return r;
}

// --- stream capture --------------------------------------------------------

// One of this call's outputs is a pointer to an array the driver owns, which
// cannot cross a wire as a pointer. The server sends the contents and they are
// kept here for as long as the driver would have kept its own: until the next
// call that changes the capture. The buffer is per thread because the caller
// may only use it before its own next call.
CUresult cuStreamGetCaptureInfo_v2(CUstream hStream,
                                   CUstreamCaptureStatus* captureStatus_out,
                                   cuuint64_t* id_out, CUgraph* graph_out,
                                   const CUgraphNode** dependencies_out,
                                   size_t* numDependencies_out) {
  static thread_local std::vector<CUgraphNode> deps;

  rgpu::Buffer req, rsp;
  req.put<uint64_t>(reinterpret_cast<uint64_t>(hStream));
  req.put<uint8_t>(dependencies_out ? 1 : 0);
  CUresult r = rgpu::call(rgpu::API_rgpu_capture_info, req, &rsp);
  if (r != CUDA_SUCCESS) return r;

  int32_t status = 0;
  uint64_t id = 0, graph = 0, ndeps = 0;
  if (!rsp.get(&status) || !rsp.get(&id) || !rsp.get(&graph) ||
      !rsp.get(&ndeps)) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (dependencies_out) {
    const uint8_t* bytes = nullptr;
    size_t n = 0;
    if (!rsp.get_sized(&bytes, &n)) return CUDA_ERROR_UNKNOWN;
    deps.assign(reinterpret_cast<const CUgraphNode*>(bytes),
                reinterpret_cast<const CUgraphNode*>(bytes + n));
    *dependencies_out = deps.data();
  }
  if (captureStatus_out) {
    *captureStatus_out = static_cast<CUstreamCaptureStatus>(status);
  }
  if (id_out) *id_out = id;
  if (graph_out) *graph_out = reinterpret_cast<CUgraph>(graph);
  if (numDependencies_out) *numDependencies_out = ndeps;
  return CUDA_SUCCESS;
}

// The count here is the caller's capacity going in and the number written
// coming out, so the generated form was wrong in a quiet way: it would have
// reported no nodes rather than failing.
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) {
  if (nodes && !numNodes) return CUDA_ERROR_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  req.put<uint64_t>(reinterpret_cast<uint64_t>(hGraph));
  req.put<uint8_t>(nodes ? 1 : 0);
  req.put<uint64_t>(nodes ? static_cast<uint64_t>(*numNodes) : 0);
  CUresult r = rgpu::call(rgpu::API_rgpu_graph_nodes, req, &rsp);
  if (r != CUDA_SUCCESS) return r;
  uint64_t n = 0;
  if (!rsp.get(&n)) return CUDA_ERROR_UNKNOWN;
  if (nodes) {
    const uint8_t* bytes = nullptr;
    size_t len = 0;
    if (!rsp.get_sized(&bytes, &len)) return CUDA_ERROR_UNKNOWN;
    const size_t room = *numNodes * sizeof(CUgraphNode);
    std::memcpy(nodes, bytes, len < room ? len : room);
  }
  if (numNodes) *numNodes = static_cast<size_t>(n);
  return CUDA_SUCCESS;
}

// The mode is an argument as well as a result: the caller says which mode it
// wants and is told which was in force. PyTorch uses this to make an
// allocation legal in the middle of a stream capture, so a version that always
// asks for mode zero does not fail here - it invalidates the capture, and the
// error surfaces later at cuStreamEndCapture with nothing to point at.
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) {
  if (!mode) return CUDA_ERROR_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  req.put<int32_t>(static_cast<int32_t>(*mode));
  CUresult r = rgpu::call(rgpu::API_rgpu_capture_mode, req, &rsp);
  if (r != CUDA_SUCCESS) return r;
  int32_t previous = 0;
  if (!rsp.get(&previous)) return CUDA_ERROR_UNKNOWN;
  *mode = static_cast<CUstreamCaptureMode>(previous);
  return CUDA_SUCCESS;
}

// --- the primary context ---------------------------------------------------
//
// PyTorch asks whether a device's primary context exists before a great many
// operations: 289 times per ResNet-18 inference, measured, which was 45% of
// every round trip the shim made. The answer only changes when the primary
// context is retained, released or reset, and every one of those goes through
// here, so while we hold a reference the answer is ours to give.
//
// ponytail: the cache is only trusted while our own retain count is above
// zero. Another client of the same server could change the state underneath
// us otherwise, and one process per server is not a promise this makes.

namespace {

struct PrimaryCtx {
  int retained = 0;        // our own references, not the driver's
  unsigned int flags = 0;
  bool flags_known = false;
};

auto& g_primary_mu = *new std::mutex();
auto& g_primary = *new std::map<CUdevice, PrimaryCtx>();

}  // namespace

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  rgpu::Buffer req, rsp;
  req.put<uint8_t>(pctx ? 1 : 0);
  req.put<CUdevice>(dev);
  CUresult r = rgpu::call(rgpu::API_cuDevicePrimaryCtxRetain, req, &rsp);
  if (r != CUDA_SUCCESS) return r;
  if (pctx) {
    uint64_t h = 0;
    if (!rsp.get(&h)) return CUDA_ERROR_UNKNOWN;
    *pctx = reinterpret_cast<CUcontext>(h);
  }
  std::lock_guard<std::mutex> lk(g_primary_mu);
  g_primary[dev].retained++;
  return r;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  rgpu::Buffer req, rsp;
  req.put<CUdevice>(dev);
  CUresult r = rgpu::call(rgpu::API_cuDevicePrimaryCtxRelease_v2, req, &rsp);
  {
    std::lock_guard<std::mutex> lk(g_primary_mu);
    auto& p = g_primary[dev];
    if (p.retained > 0) p.retained--;
    if (p.retained == 0) p.flags_known = false;
  }
  return r;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  rgpu::Buffer req, rsp;
  req.put<CUdevice>(dev);
  CUresult r = rgpu::call(rgpu::API_cuDevicePrimaryCtxReset_v2, req, &rsp);
  // A reset does not release the primary context ("Resetting the primary
  // context does not release it"), so every retain this process holds is
  // still held afterwards, on the server as in CUDA. What the reset does end is
  // what the record says about the context: that it is active, and its flags.
  // So a reset that may have run forgets the record, and the next state query
  // asks the server. `retained` goes to zero with it - it is this cache's
  // licence to answer, not the driver's count - which costs round trips, not
  // correctness: a later release of a retain still held finds zero and leaves
  // it there.
  //
  // Only a refusal leaves the record alone. The server refuses a reset while
  // other sessions are live, and answers CUDA_ERROR_NOT_SUPPORTED without
  // touching the device, so the record is still true. Any other answer may
  // follow a reset that ran: the server hands an error held from an earlier
  // call that had no reply to the next call that succeeds, and a connection
  // that died before the answer says nothing about whether the call ran.
  // Known gap: NOT_SUPPORTED is not proof of a refusal. A held no-reply
  // failure with that code, folded into a reset that ran, looks the same and
  // keeps the stale record.
  if (r != CUDA_ERROR_NOT_SUPPORTED) {
    std::lock_guard<std::mutex> lk(g_primary_mu);
    g_primary[dev] = PrimaryCtx{};
  }
  return r;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
  rgpu::Buffer req, rsp;
  req.put<CUdevice>(dev);
  req.put<unsigned int>(flags);
  CUresult r = rgpu::call(rgpu::API_cuDevicePrimaryCtxSetFlags_v2, req, &rsp);
  {
    std::lock_guard<std::mutex> lk(g_primary_mu);
    auto& p = g_primary[dev];
    if (r == CUDA_SUCCESS) {
      p.flags = flags;
      p.flags_known = true;
    } else {
      p.flags_known = false;
    }
  }
  return r;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags,
                                    int* active) {
  {
    std::lock_guard<std::mutex> lk(g_primary_mu);
    auto it = g_primary.find(dev);
    // Holding a reference means the context is active; that much needs no
    // asking. The flags still do, once, unless we set them ourselves.
    if (it != g_primary.end() && it->second.retained > 0 &&
        (!flags || it->second.flags_known)) {
      if (flags) *flags = it->second.flags;
      if (active) *active = 1;
      return CUDA_SUCCESS;
    }
  }

  rgpu::Buffer req, rsp;
  req.put<CUdevice>(dev);
  req.put<uint8_t>(flags ? 1 : 0);
  req.put<uint8_t>(active ? 1 : 0);
  CUresult r = rgpu::call(rgpu::API_cuDevicePrimaryCtxGetState, req, &rsp);
  if (r != CUDA_SUCCESS) return r;
  unsigned int got_flags = 0;
  int got_active = 0;
  if (flags && !rsp.get(&got_flags)) return CUDA_ERROR_UNKNOWN;
  if (active && !rsp.get(&got_active)) return CUDA_ERROR_UNKNOWN;
  if (flags) *flags = got_flags;
  if (active) *active = got_active;
  if (flags) {
    std::lock_guard<std::mutex> lk(g_primary_mu);
    auto& p = g_primary[dev];
    p.flags = got_flags;
    p.flags_known = true;
  }
  return r;
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

// Refused rather than forwarded. The launch message carries no way to say
// "cooperative", so the server would run this through the ordinary
// cuLaunchKernel and the co-residency the kernel was written around would not
// be there: a grid-wide barrier would hang or, worse, return wrong numbers
// that look right. An error the caller can see beats an answer it cannot check.
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void** kernelParams) {
  (void)f; (void)gridDimX; (void)gridDimY; (void)gridDimZ;
  (void)blockDimX; (void)blockDimY; (void)blockDimZ;
  (void)sharedMemBytes; (void)hStream; (void)kernelParams;
  return rgpu::unimplemented(
      "cuLaunchCooperativeKernel",
      "cooperative launch guarantees cannot be preserved across the wire");
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
