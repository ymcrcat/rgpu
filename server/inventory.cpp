#include "server/inventory.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

#include "server/driver_syms.h"

namespace rgpu {
namespace {

// The session this thread is serving. One thread serves one session for as
// long as the session lives, and the same thread releases the inventory at the
// end, so this is the whole of the plumbing between a call and the list it
// belongs on.
thread_local Inventory* t_inv = nullptr;

// How many sessions are being served. A call that reaches past its own session
// - resetting a device is the one that does - has to know whether anybody else
// would be caught by it.
std::atomic<int> g_live_sessions{0};

void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[rgpu-server] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

// Resolves an entry point without the tracking wrapper. Everything in this
// file is either recording a call it is about to make or undoing one, so going
// back through driver_sym() would record the cleanup as well.
#define REAL(name, ...)                                                       \
  static auto fn =                                                            \
      reinterpret_cast<CUresult (*)(__VA_ARGS__)>(driver_sym_raw(name));      \
  if (!fn) return CUDA_ERROR_NOT_SUPPORTED

CUcontext current_ctx() {
  static auto fn = reinterpret_cast<CUresult (*)(CUcontext*)>(
      driver_sym_raw("cuCtxGetCurrent"));
  CUcontext ctx = nullptr;
  if (fn) fn(&ctx);
  return ctx;
}

// --- recording ------------------------------------------------------------

void note(Inventory::Items Inventory::*which, uint64_t handle) {
  Inventory* inv = t_inv;
  if (!inv || !handle) return;
  // Read outside the lock: it is a thread-local lookup in the driver, and the
  // only thread that could be holding this lock is this one.
  const CUcontext ctx = current_ctx();
  std::lock_guard<std::mutex> lk(inv->mu);
  (inv->*which)[handle] = Inventory::Item{ctx};
}

void forget(Inventory::Items Inventory::*which, uint64_t handle) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  (inv->*which).erase(handle);
}

void note_primary_retain(CUdevice dev, int delta, CUcontext ctx) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  if (ctx) inv->primary_ctx[dev] = ctx;
  int& n = inv->primary_retains[dev];
  n += delta;
  // A client that releases more than it retained has taken a retain off
  // somebody else, which the driver either refused or charged to the process.
  // Either way this session is not owed a release for it.
  if (n < 0) n = 0;
}

// Everything recorded under a context that no longer holds it. Called with
// the inventory locked.
void forget_under(Inventory* inv, CUcontext ctx) {
  Inventory::Items* maps[] = {&inv->allocs,  &inv->modules, &inv->streams,
                              &inv->events,  &inv->graphs,  &inv->graph_execs};
  for (Inventory::Items* m : maps) {
    for (auto it = m->begin(); it != m->end();) {
      it = it->second.ctx == ctx ? m->erase(it) : std::next(it);
    }
  }
  for (auto it = inv->handles.begin(); it != inv->handles.end();) {
    it = it->second.ctx == ctx ? inv->handles.erase(it) : std::next(it);
  }
}

// A destroyed context takes everything inside it with it, so those entries
// stop being this session's to give back. Freeing them afterwards would be
// handing the driver values that no longer mean anything - and, once the
// driver reuses an address, could mean somebody else's memory.
void forget_context(CUcontext ctx) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  inv->contexts.erase(reinterpret_cast<uint64_t>(ctx));
  forget_under(inv, ctx);
}

// Resetting a device destroys everything in its primary context without
// destroying the context, so the session's record of what is in there is
// suddenly wrong. Only this session's record can be corrected here; a reset is
// a process-wide act, and a client that makes one while another session is
// using the device has already broken that session, driver or no driver.
void forget_primary(CUdevice dev) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  auto it = inv->primary_ctx.find(dev);
  if (it == inv->primary_ctx.end()) return;
  forget_under(inv, it->second);
  // The retains go too. A reset destroys the primary context and all its
  // state, and the driver is entitled to have dropped the reference count with
  // it; owing releases against a count that no longer exists would take a
  // retain off whoever retains it next. Forgetting them can at worst leak one
  // retain, which is the safe way to be wrong.
  inv->primary_retains.erase(dev);
  inv->primary_ctx.erase(it);
}

// --- the wrappers ---------------------------------------------------------
//
// Each one is the driver's own call with a line of bookkeeping after it. They
// are what driver_sym() hands to the generated dispatch, so a call is recorded
// exactly where it happens, with the driver's own types, and no generated code
// has to know.

CUresult w_cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytes) {
  REAL("cuMemAlloc_v2", CUdeviceptr*, size_t);
  CUresult r = fn(dptr, bytes);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr);
  return r;
}

CUresult w_cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pitch, size_t width,
                              size_t height, unsigned int elem) {
  REAL("cuMemAllocPitch_v2", CUdeviceptr*, size_t*, size_t, size_t,
       unsigned int);
  CUresult r = fn(dptr, pitch, width, height, elem);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr);
  return r;
}

CUresult w_cuMemAllocAsync(CUdeviceptr* dptr, size_t bytes, CUstream stream) {
  REAL("cuMemAllocAsync", CUdeviceptr*, size_t, CUstream);
  CUresult r = fn(dptr, bytes, stream);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr);
  return r;
}

CUresult w_cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytes,
                                   CUmemoryPool pool, CUstream stream) {
  REAL("cuMemAllocFromPoolAsync", CUdeviceptr*, size_t, CUmemoryPool, CUstream);
  CUresult r = fn(dptr, bytes, pool, stream);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr);
  return r;
}

CUresult w_cuMemFree_v2(CUdeviceptr dptr) {
  REAL("cuMemFree_v2", CUdeviceptr);
  CUresult r = fn(dptr);
  if (r == CUDA_SUCCESS) forget(&Inventory::allocs, dptr);
  return r;
}

CUresult w_cuMemFreeAsync(CUdeviceptr dptr, CUstream stream) {
  REAL("cuMemFreeAsync", CUdeviceptr, CUstream);
  CUresult r = fn(dptr, stream);
  if (r == CUDA_SUCCESS) forget(&Inventory::allocs, dptr);
  return r;
}

CUresult w_cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  REAL("cuDevicePrimaryCtxRetain", CUcontext*, CUdevice);
  CUresult r = fn(pctx, dev);
  if (r == CUDA_SUCCESS) note_primary_retain(dev, 1, pctx ? *pctx : nullptr);
  return r;
}

CUresult w_cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  REAL("cuDevicePrimaryCtxRelease_v2", CUdevice);
  CUresult r = fn(dev);
  if (r == CUDA_SUCCESS) note_primary_retain(dev, -1, nullptr);
  return r;
}

// Resetting a device destroys everything in its primary context, for every
// session in this process, and only the session that asked can have its record
// corrected. One client must not be able to throw another tenant's memory away
// - the peer is not authenticated, so it must not be trusted with it - so this
// is refused outright while anybody else is connected, in the same spirit as
// the launch types the shim will not carry. Alone on the server it is allowed,
// because then there is nobody else to harm.
CUresult w_cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  REAL("cuDevicePrimaryCtxReset_v2", CUdevice);
  if (g_live_sessions.load() > 1) {
    logf("refusing cuDevicePrimaryCtxReset on device %d: %d sessions are live "
         "and it would destroy what all of them are holding",
         dev, g_live_sessions.load());
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult r = fn(dev);
  if (r == CUDA_SUCCESS) forget_primary(dev);
  return r;
}

CUresult w_cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  REAL("cuCtxCreate_v2", CUcontext*, unsigned int, CUdevice);
  CUresult r = fn(pctx, flags, dev);
  // Creating a context makes it current, so it is its own context: recorded
  // that way, destroying it is enough to account for everything in it.
  if (r == CUDA_SUCCESS && pctx) {
    note(&Inventory::contexts, reinterpret_cast<uint64_t>(*pctx));
  }
  return r;
}

CUresult w_cuCtxDestroy_v2(CUcontext ctx) {
  REAL("cuCtxDestroy_v2", CUcontext);
  CUresult r = fn(ctx);
  if (r == CUDA_SUCCESS) forget_context(ctx);
  return r;
}

CUresult w_cuModuleLoad(CUmodule* mod, const char* path) {
  REAL("cuModuleLoad", CUmodule*, const char*);
  CUresult r = fn(mod, path);
  if (r == CUDA_SUCCESS && mod) {
    note(&Inventory::modules, reinterpret_cast<uint64_t>(*mod));
  }
  return r;
}

CUresult w_cuModuleLoadData(CUmodule* mod, const void* image) {
  REAL("cuModuleLoadData", CUmodule*, const void*);
  CUresult r = fn(mod, image);
  if (r == CUDA_SUCCESS && mod) {
    note(&Inventory::modules, reinterpret_cast<uint64_t>(*mod));
  }
  return r;
}

CUresult w_cuModuleLoadFatBinary(CUmodule* mod, const void* image) {
  REAL("cuModuleLoadFatBinary", CUmodule*, const void*);
  CUresult r = fn(mod, image);
  if (r == CUDA_SUCCESS && mod) {
    note(&Inventory::modules, reinterpret_cast<uint64_t>(*mod));
  }
  return r;
}

CUresult w_cuModuleUnload(CUmodule mod) {
  REAL("cuModuleUnload", CUmodule);
  CUresult r = fn(mod);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::modules, reinterpret_cast<uint64_t>(mod));
  }
  return r;
}

CUresult w_cuStreamCreate(CUstream* stream, unsigned int flags) {
  REAL("cuStreamCreate", CUstream*, unsigned int);
  CUresult r = fn(stream, flags);
  if (r == CUDA_SUCCESS && stream) {
    note(&Inventory::streams, reinterpret_cast<uint64_t>(*stream));
  }
  return r;
}

CUresult w_cuStreamCreateWithPriority(CUstream* stream, unsigned int flags,
                                      int priority) {
  REAL("cuStreamCreateWithPriority", CUstream*, unsigned int, int);
  CUresult r = fn(stream, flags, priority);
  if (r == CUDA_SUCCESS && stream) {
    note(&Inventory::streams, reinterpret_cast<uint64_t>(*stream));
  }
  return r;
}

CUresult w_cuStreamDestroy_v2(CUstream stream) {
  REAL("cuStreamDestroy_v2", CUstream);
  CUresult r = fn(stream);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::streams, reinterpret_cast<uint64_t>(stream));
  }
  return r;
}

CUresult w_cuEventCreate(CUevent* event, unsigned int flags) {
  REAL("cuEventCreate", CUevent*, unsigned int);
  CUresult r = fn(event, flags);
  if (r == CUDA_SUCCESS && event) {
    note(&Inventory::events, reinterpret_cast<uint64_t>(*event));
  }
  return r;
}

CUresult w_cuEventDestroy_v2(CUevent event) {
  REAL("cuEventDestroy_v2", CUevent);
  CUresult r = fn(event);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::events, reinterpret_cast<uint64_t>(event));
  }
  return r;
}

CUresult w_cuGraphCreate(CUgraph* graph, unsigned int flags) {
  REAL("cuGraphCreate", CUgraph*, unsigned int);
  CUresult r = fn(graph, flags);
  if (r == CUDA_SUCCESS && graph) {
    note(&Inventory::graphs, reinterpret_cast<uint64_t>(*graph));
  }
  return r;
}

// A capture hands the finished graph to the client, which owns it from here
// exactly as if it had created one.
CUresult w_cuStreamEndCapture(CUstream stream, CUgraph* graph) {
  REAL("cuStreamEndCapture", CUstream, CUgraph*);
  CUresult r = fn(stream, graph);
  if (r == CUDA_SUCCESS && graph) {
    note(&Inventory::graphs, reinterpret_cast<uint64_t>(*graph));
  }
  return r;
}

CUresult w_cuGraphDestroy(CUgraph graph) {
  REAL("cuGraphDestroy", CUgraph);
  CUresult r = fn(graph);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::graphs, reinterpret_cast<uint64_t>(graph));
  }
  return r;
}

CUresult w_cuGraphInstantiateWithFlags(CUgraphExec* exec, CUgraph graph,
                                       unsigned long long flags) {
  REAL("cuGraphInstantiateWithFlags", CUgraphExec*, CUgraph,
       unsigned long long);
  CUresult r = fn(exec, graph, flags);
  if (r == CUDA_SUCCESS && exec) {
    note(&Inventory::graph_execs, reinterpret_cast<uint64_t>(*exec));
  }
  return r;
}

CUresult w_cuGraphExecDestroy(CUgraphExec exec) {
  REAL("cuGraphExecDestroy", CUgraphExec);
  CUresult r = fn(exec);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::graph_execs, reinterpret_cast<uint64_t>(exec));
  }
  return r;
}

// A green context is a slice of a device, and nothing about it is tied to a
// CUcontext, so it outlives everything else a session holds.
CUresult w_cuGreenCtxCreate(CUgreenCtx* pctx, CUdevResourceDesc desc,
                            CUdevice dev, unsigned int flags) {
  REAL("cuGreenCtxCreate", CUgreenCtx*, CUdevResourceDesc, CUdevice,
       unsigned int);
  CUresult r = fn(pctx, desc, dev, flags);
  if (r == CUDA_SUCCESS && pctx) {
    note(&Inventory::green_ctxs, reinterpret_cast<uint64_t>(*pctx));
  }
  return r;
}

CUresult w_cuGreenCtxDestroy(CUgreenCtx ctx) {
  REAL("cuGreenCtxDestroy", CUgreenCtx);
  CUresult r = fn(ctx);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::green_ctxs, reinterpret_cast<uint64_t>(ctx));
  }
  return r;
}

// An ordinary stream, however it was made, so it is released like one.
CUresult w_cuGreenCtxStreamCreate(CUstream* stream, CUgreenCtx ctx,
                                  unsigned int flags, int priority) {
  REAL("cuGreenCtxStreamCreate", CUstream*, CUgreenCtx, unsigned int, int);
  CUresult r = fn(stream, ctx, flags, priority);
  if (r == CUDA_SUCCESS && stream) {
    note(&Inventory::streams, reinterpret_cast<uint64_t>(*stream));
  }
  return r;
}

CUresult w_cuLibraryLoadData(CUlibrary* library, const void* code,
                             CUjit_option* jit_options, void** jit_values,
                             unsigned int num_jit, CUlibraryOption* lib_options,
                             void** lib_values, unsigned int num_lib) {
  REAL("cuLibraryLoadData", CUlibrary*, const void*, CUjit_option*, void**,
       unsigned int, CUlibraryOption*, void**, unsigned int);
  CUresult r = fn(library, code, jit_options, jit_values, num_jit, lib_options,
                  lib_values, num_lib);
  if (r == CUDA_SUCCESS && library) {
    note(&Inventory::libraries, reinterpret_cast<uint64_t>(*library));
  }
  return r;
}

CUresult w_cuLibraryUnload(CUlibrary library) {
  REAL("cuLibraryUnload", CUlibrary);
  CUresult r = fn(library);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::libraries, reinterpret_cast<uint64_t>(library));
  }
  return r;
}

CUresult w_cuGraphClone(CUgraph* clone, CUgraph original) {
  REAL("cuGraphClone", CUgraph*, CUgraph);
  CUresult r = fn(clone, original);
  if (r == CUDA_SUCCESS && clone) {
    note(&Inventory::graphs, reinterpret_cast<uint64_t>(*clone));
  }
  return r;
}

CUresult w_cuTexRefCreate(CUtexref* texref) {
  REAL("cuTexRefCreate", CUtexref*);
  CUresult r = fn(texref);
  if (r == CUDA_SUCCESS && texref) {
    note(&Inventory::texrefs, reinterpret_cast<uint64_t>(*texref));
  }
  return r;
}

CUresult w_cuTexRefDestroy(CUtexref texref) {
  REAL("cuTexRefDestroy", CUtexref);
  CUresult r = fn(texref);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::texrefs, reinterpret_cast<uint64_t>(texref));
  }
  return r;
}

struct Tracked {
  const char* name;
  void* fn;
};

// Everything whose result a client can still be holding when it dies. Calls
// that are not here are not recorded, which is only safe because they create
// nothing that outlives the call: a memcpy, a query, a launch.
const Tracked kTracked[] = {
    {"cuMemAlloc_v2", reinterpret_cast<void*>(&w_cuMemAlloc_v2)},
    {"cuMemAllocPitch_v2", reinterpret_cast<void*>(&w_cuMemAllocPitch_v2)},
    {"cuMemAllocAsync", reinterpret_cast<void*>(&w_cuMemAllocAsync)},
    {"cuMemAllocFromPoolAsync",
     reinterpret_cast<void*>(&w_cuMemAllocFromPoolAsync)},
    {"cuMemFree_v2", reinterpret_cast<void*>(&w_cuMemFree_v2)},
    {"cuMemFreeAsync", reinterpret_cast<void*>(&w_cuMemFreeAsync)},
    {"cuDevicePrimaryCtxRetain",
     reinterpret_cast<void*>(&w_cuDevicePrimaryCtxRetain)},
    {"cuDevicePrimaryCtxRelease_v2",
     reinterpret_cast<void*>(&w_cuDevicePrimaryCtxRelease_v2)},
    {"cuDevicePrimaryCtxReset_v2",
     reinterpret_cast<void*>(&w_cuDevicePrimaryCtxReset_v2)},
    {"cuCtxCreate_v2", reinterpret_cast<void*>(&w_cuCtxCreate_v2)},
    {"cuCtxDestroy_v2", reinterpret_cast<void*>(&w_cuCtxDestroy_v2)},
    {"cuModuleLoad", reinterpret_cast<void*>(&w_cuModuleLoad)},
    {"cuModuleLoadData", reinterpret_cast<void*>(&w_cuModuleLoadData)},
    {"cuModuleLoadFatBinary",
     reinterpret_cast<void*>(&w_cuModuleLoadFatBinary)},
    {"cuModuleUnload", reinterpret_cast<void*>(&w_cuModuleUnload)},
    {"cuStreamCreate", reinterpret_cast<void*>(&w_cuStreamCreate)},
    {"cuStreamCreateWithPriority",
     reinterpret_cast<void*>(&w_cuStreamCreateWithPriority)},
    {"cuStreamDestroy_v2", reinterpret_cast<void*>(&w_cuStreamDestroy_v2)},
    {"cuEventCreate", reinterpret_cast<void*>(&w_cuEventCreate)},
    {"cuEventDestroy_v2", reinterpret_cast<void*>(&w_cuEventDestroy_v2)},
    {"cuGraphCreate", reinterpret_cast<void*>(&w_cuGraphCreate)},
    {"cuStreamEndCapture", reinterpret_cast<void*>(&w_cuStreamEndCapture)},
    {"cuGraphDestroy", reinterpret_cast<void*>(&w_cuGraphDestroy)},
    {"cuGraphInstantiateWithFlags",
     reinterpret_cast<void*>(&w_cuGraphInstantiateWithFlags)},
    {"cuGraphExecDestroy", reinterpret_cast<void*>(&w_cuGraphExecDestroy)},
    {"cuGraphClone", reinterpret_cast<void*>(&w_cuGraphClone)},
    {"cuGreenCtxCreate", reinterpret_cast<void*>(&w_cuGreenCtxCreate)},
    {"cuGreenCtxDestroy", reinterpret_cast<void*>(&w_cuGreenCtxDestroy)},
    {"cuGreenCtxStreamCreate",
     reinterpret_cast<void*>(&w_cuGreenCtxStreamCreate)},
    {"cuLibraryLoadData", reinterpret_cast<void*>(&w_cuLibraryLoadData)},
    {"cuLibraryUnload", reinterpret_cast<void*>(&w_cuLibraryUnload)},
    {"cuTexRefCreate", reinterpret_cast<void*>(&w_cuTexRefCreate)},
    {"cuTexRefDestroy", reinterpret_cast<void*>(&w_cuTexRefDestroy)},
};

// --- releasing ------------------------------------------------------------

CUresult destroy_alloc(uint64_t h) {
  REAL("cuMemFree_v2", CUdeviceptr);
  return fn(static_cast<CUdeviceptr>(h));
}
CUresult destroy_module(uint64_t h) {
  REAL("cuModuleUnload", CUmodule);
  return fn(reinterpret_cast<CUmodule>(h));
}
CUresult destroy_stream(uint64_t h) {
  REAL("cuStreamDestroy_v2", CUstream);
  return fn(reinterpret_cast<CUstream>(h));
}
CUresult destroy_event(uint64_t h) {
  REAL("cuEventDestroy_v2", CUevent);
  return fn(reinterpret_cast<CUevent>(h));
}
CUresult destroy_graph(uint64_t h) {
  REAL("cuGraphDestroy", CUgraph);
  return fn(reinterpret_cast<CUgraph>(h));
}
CUresult destroy_graph_exec(uint64_t h) {
  REAL("cuGraphExecDestroy", CUgraphExec);
  return fn(reinterpret_cast<CUgraphExec>(h));
}
CUresult destroy_green_ctx(uint64_t h) {
  REAL("cuGreenCtxDestroy", CUgreenCtx);
  return fn(reinterpret_cast<CUgreenCtx>(h));
}
CUresult destroy_library(uint64_t h) {
  REAL("cuLibraryUnload", CUlibrary);
  return fn(reinterpret_cast<CUlibrary>(h));
}
CUresult destroy_texref(uint64_t h) {
  REAL("cuTexRefDestroy", CUtexref);
  return fn(reinterpret_cast<CUtexref>(h));
}
CUresult destroy_context(uint64_t h) {
  REAL("cuCtxDestroy_v2", CUcontext);
  return fn(reinterpret_cast<CUcontext>(h));
}
CUresult release_primary(CUdevice dev) {
  REAL("cuDevicePrimaryCtxRelease_v2", CUdevice);
  return fn(dev);
}
CUresult set_context(CUcontext ctx) {
  REAL("cuCtxSetCurrent", CUcontext);
  return fn(ctx);
}
CUresult sync_context() {
  REAL("cuCtxSynchronize", void);
  return fn();
}

// Makes a context current before something inside it is destroyed, and
// remembers which one is current so a run of resources from the same context
// costs one call. A context that cannot be made current is still worth trying
// the destroy under, in case the driver does not need it.
class Current {
 public:
  // A null context means no context, not "leave whatever the last one was":
  // a resource recorded without one was made without one, and destroying it
  // under a context it never belonged to is at best luck.
  void use(CUcontext ctx) {
    if (known_ && ctx == ctx_) return;
    CUresult r = set_context(ctx);
    if (r != CUDA_SUCCESS) {
      logf("session cleanup: could not make context %p current (%d); "
           "releasing what is in it anyway",
           (void*)ctx, r);
    }
    ctx_ = ctx;
    known_ = true;
  }
  void none() { use(nullptr); }

 private:
  CUcontext ctx_ = nullptr;
  bool known_ = false;
};

unsigned release_items(Inventory::Items& items, const char* what,
                       CUresult (*destroy)(uint64_t), Current* current,
                       unsigned* failed) {
  unsigned released = 0;
  for (const auto& entry : items) {
    current->use(entry.second.ctx);
    CUresult r = destroy(entry.first);
    if (r == CUDA_SUCCESS) {
      released++;
    } else {
      (*failed)++;
      logf("session cleanup: %s %llx was not released (%d)", what,
           (unsigned long long)entry.first, r);
    }
  }
  return released;
}

// The same, for what does not live in a context and must not be destroyed
// under one.
unsigned release_detached(Inventory::Items& items, const char* what,
                          CUresult (*destroy)(uint64_t), unsigned* failed) {
  unsigned released = 0;
  for (const auto& entry : items) {
    CUresult r = destroy(entry.first);
    if (r == CUDA_SUCCESS) {
      released++;
    } else {
      (*failed)++;
      logf("session cleanup: %s %llx was not released (%d)", what,
           (unsigned long long)entry.first, r);
    }
  }
  return released;
}

std::string plural(unsigned n, const char* one, const char* many = nullptr) {
  if (n == 1) return std::to_string(n) + " " + one;
  return std::to_string(n) + " " + (many ? std::string(many)
                                         : std::string(one) + "s");
}

}  // namespace

void inventory_bind(Inventory* inv) {
  if (inv && !t_inv) g_live_sessions.fetch_add(1);
  if (!inv && t_inv) g_live_sessions.fetch_sub(1);
  t_inv = inv;
}

void inventory_note_handle(uint64_t handle, const char* what,
                           CUresult (*destroy)(uint64_t)) {
  Inventory* inv = t_inv;
  if (!inv || !handle) return;
  const CUcontext ctx = current_ctx();
  std::lock_guard<std::mutex> lk(inv->mu);
  inv->handles[handle] = Inventory::LibHandle{ctx, what, destroy};
}

void inventory_forget_handle(uint64_t handle) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  inv->handles.erase(handle);
}

void* driver_wrapper(const char* name) {
  for (const Tracked& t : kTracked) {
    if (std::strcmp(t.name, name) == 0) return t.fn;
  }
  return nullptr;
}

std::string release_inventory(Inventory& inv) {
  // Taken out whole, so nothing is held while the driver works and anything
  // that somehow arrives afterwards is recorded against an empty list rather
  // than freed twice.
  Inventory::Items allocs, contexts, modules, streams, events, graphs, execs;
  Inventory::Items green_ctxs, libraries, texrefs;
  std::unordered_map<uint64_t, Inventory::LibHandle> handles;
  std::unordered_map<int, int> retains;
  {
    std::lock_guard<std::mutex> lk(inv.mu);
    allocs.swap(inv.allocs);
    contexts.swap(inv.contexts);
    modules.swap(inv.modules);
    streams.swap(inv.streams);
    events.swap(inv.events);
    graphs.swap(inv.graphs);
    execs.swap(inv.graph_execs);
    green_ctxs.swap(inv.green_ctxs);
    libraries.swap(inv.libraries);
    texrefs.swap(inv.texrefs);
    handles.swap(inv.handles);
    retains.swap(inv.primary_retains);
  }

  unsigned failed = 0;
  Current current;

  // Work still running would be reading the memory about to be freed. The
  // contexts are synchronised first, once each, rather than per resource.
  {
    Inventory::Items* maps[] = {&allocs, &modules,   &streams, &events,
                                &graphs, &execs,     &libraries, &texrefs};
    std::vector<CUcontext> seen;
    for (const Inventory::Items* m : maps) {
      for (const auto& entry : *m) {
        CUcontext ctx = entry.second.ctx;
        if (!ctx) continue;
        bool known = false;
        for (CUcontext c : seen) known = known || c == ctx;
        if (known) continue;
        seen.push_back(ctx);
        current.use(ctx);
        CUresult r = sync_context();
        if (r != CUDA_SUCCESS) {
          logf("session cleanup: context %p did not synchronise (%d); "
               "releasing it anyway",
               (void*)ctx, r);
        }
      }
    }
  }

  // Order matters. A library handle owns streams and workspaces of its own, so
  // it goes before the memory; what lives in a context goes before the
  // context; and the primary context, which is shared with every other
  // session, goes last of all.
  unsigned handles_released = 0;
  for (const auto& entry : handles) {
    current.use(entry.second.ctx);
    CUresult r = entry.second.destroy
                     ? entry.second.destroy(entry.first)
                     : CUDA_ERROR_NOT_SUPPORTED;
    if (r == CUDA_SUCCESS) {
      handles_released++;
    } else {
      failed++;
      logf("session cleanup: %s handle %llx was not destroyed (%d)",
           entry.second.what, (unsigned long long)entry.first, r);
    }
  }

  const unsigned execs_released =
      release_items(execs, "graph exec", destroy_graph_exec, &current, &failed);
  const unsigned graphs_released =
      release_items(graphs, "graph", destroy_graph, &current, &failed);
  const unsigned events_released =
      release_items(events, "event", destroy_event, &current, &failed);
  const unsigned streams_released =
      release_items(streams, "stream", destroy_stream, &current, &failed);
  const unsigned modules_released =
      release_items(modules, "module", destroy_module, &current, &failed);
  const unsigned libraries_released =
      release_items(libraries, "library", destroy_library, &current, &failed);
  const unsigned texrefs_released = release_items(
      texrefs, "texture reference", destroy_texref, &current, &failed);
  const unsigned allocs_released =
      release_items(allocs, "allocation", destroy_alloc, &current, &failed);

  // Nothing below belongs to a context. Destroying the one this thread is
  // pointing at would leave it pointing at something that is gone, and a green
  // context is a slice of the device rather than something inside a context,
  // so both are destroyed by handle with nothing current.
  current.none();
  const unsigned green_released =
      release_detached(green_ctxs, "green context", destroy_green_ctx, &failed);
  const unsigned contexts_released =
      release_detached(contexts, "context", destroy_context, &failed);

  // Exactly as many releases as this session took retains, and no more: the
  // count above is already net of the releases the client made itself. One
  // release too many here would take the primary context away from a session
  // that is still using it.
  unsigned retains_released = 0;
  for (const auto& entry : retains) {
    for (int i = 0; i < entry.second; i++) {
      CUresult r = release_primary(entry.first);
      if (r == CUDA_SUCCESS) {
        retains_released++;
        continue;
      }
      // The next one would fail the same way, and saying so once with the
      // count is more use than the same line repeated.
      failed++;
      logf("session cleanup: releasing the primary context on device %d "
           "failed (%d); %d of this session's retains are still held",
           entry.first, r, entry.second - i);
      break;
    }
  }

  std::vector<std::string> parts;
  if (allocs_released) parts.push_back(plural(allocs_released, "allocation"));
  if (retains_released) {
    parts.push_back(plural(retains_released, "primary-context retain"));
  }
  if (contexts_released) parts.push_back(plural(contexts_released, "context"));
  if (handles_released) {
    parts.push_back(plural(handles_released, "library handle"));
  }
  if (modules_released) parts.push_back(plural(modules_released, "module"));
  if (streams_released) parts.push_back(plural(streams_released, "stream"));
  if (events_released) parts.push_back(plural(events_released, "event"));
  if (graphs_released) parts.push_back(plural(graphs_released, "graph"));
  if (execs_released) parts.push_back(plural(execs_released, "graph exec"));
  if (green_released) parts.push_back(plural(green_released, "green context"));
  if (libraries_released) {
    parts.push_back(plural(libraries_released, "library", "libraries"));
  }
  if (texrefs_released) {
    parts.push_back(plural(texrefs_released, "texture reference"));
  }

  std::string summary;
  if (parts.empty()) {
    // Only ever a claim about the list, which is not everything a driver can
    // hand out. See the tracked table above for what is on it.
    summary = "it held nothing this server tracks";
  } else {
    summary = "released ";
    for (size_t i = 0; i < parts.size(); i++) {
      if (i) summary += i + 1 == parts.size() ? " and " : ", ";
      summary += parts[i];
    }
  }
  if (failed) {
    summary += "; " + plural(failed, "resource") + " could not be released";
  }
  return summary;
}

}  // namespace rgpu
