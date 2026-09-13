#include "server/inventory.h"

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
// would be caught by it, and has to go on knowing until it has finished: the
// mutex is held across the check and the reset, and binding takes it too, so
// no session can start between the two and have its resources destroyed
// behind its back.
std::mutex g_live_mu;
int g_live_sessions = 0;

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
//
// Only the resources go. The retains stay: the driver says "Resetting the
// primary context does not release it, an application that has retained the
// primary context should explicitly release its usage", and that "it is safe
// for other modules to call cuDevicePrimaryCtxRelease() even after resetting
// the device". So every retain this session held is still held, and expiry
// owes exactly that many releases; forgetting them would leak them for the
// life of the server. The handle stays recorded too, since the context was not
// released: a later reset has to find what was made in it afterwards, and a
// retain that hands back a new handle overwrites it.
//
// The last release of a primary context in the process destroys what is in it
// the same way, and comes here too. The handle is left recorded then as well;
// if the driver hands out a new one when the context is next retained, that
// retain overwrites it.
void forget_primary(CUdevice dev) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  auto it = inv->primary_ctx.find(dev);
  if (it == inv->primary_ctx.end()) return;
  forget_under(inv, it->second);
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

// Held across a session's retain, and across a session's release together with
// the question that follows it, so that no other session's retain can bring a
// primary context back to life between a release that ended it and the check
// that notices. Retains and releases are rare next to everything else, so
// making them wait for each other costs nothing that shows. Lock order is
// this, then the inventory's own mutex; nothing takes them the other way round.
std::mutex g_primary_mu;

CUresult w_cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  REAL("cuDevicePrimaryCtxRetain", CUcontext*, CUdevice);
  std::lock_guard<std::mutex> lk(g_primary_mu);
  CUresult r = fn(pctx, dev);
  if (r == CUDA_SUCCESS) note_primary_retain(dev, 1, pctx ? *pctx : nullptr);
  return r;
}

// Releasing the last retain in the process destroys the primary context and
// everything in it - "The context is automatically reset once the last
// reference to it is released" - so whatever this session recorded in there is
// no longer its to free. Its own count cannot say whether this release was the
// last one, because other sessions may hold retains on the same context, so
// the driver is asked instead. If the state cannot be read, the record is left
// as it is: the driver will refuse frees in a context that is gone.
CUresult w_cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  REAL("cuDevicePrimaryCtxRelease_v2", CUdevice);
  static auto get_state =
      reinterpret_cast<CUresult (*)(CUdevice, unsigned int*, int*)>(
          driver_sym_raw("cuDevicePrimaryCtxGetState"));
  std::lock_guard<std::mutex> lk(g_primary_mu);
  CUresult r = fn(dev);
  if (r != CUDA_SUCCESS) return r;
  note_primary_retain(dev, -1, nullptr);
  unsigned int flags = 0;
  int active = 1;
  if (get_state && get_state(dev, &flags, &active) == CUDA_SUCCESS &&
      !active) {
    forget_primary(dev);
  }
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
  // Held across the driver call below, deliberately, not just across the
  // check. Letting go in between would let a session bind in the gap - binding
  // takes this lock - start using the device, and then have the reset destroy
  // what it had just been given. The price is that a new session waits for one
  // reset to finish before it can start, which is rare and bounded. Lock order
  // is this, then the inventory's own mutex in forget_primary; nothing takes
  // them the other way round.
  std::lock_guard<std::mutex> lk(g_live_mu);
  const int live = g_live_sessions;
  if (live > 1) {
    logf("refusing cuDevicePrimaryCtxReset on device %d: %d sessions are live "
         "and it would destroy what all of them are holding",
         dev, live);
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

CUresult w_cuGraphClone(CUgraph* clone, CUgraph original) {
  REAL("cuGraphClone", CUgraph*, CUgraph);
  CUresult r = fn(clone, original);
  if (r == CUDA_SUCCESS && clone) {
    note(&Inventory::graphs, reinterpret_cast<uint64_t>(*clone));
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

// The same, for contexts themselves, which must not be destroyed while current.
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

std::string plural(unsigned n, const char* what) {
  return std::to_string(n) + " " + what + (n == 1 ? "" : "s");
}

}  // namespace

void inventory_bind(Inventory* inv) {
  std::lock_guard<std::mutex> lk(g_live_mu);
  if (inv && !t_inv) g_live_sessions++;
  if (!inv && t_inv) g_live_sessions--;
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
    handles.swap(inv.handles);
    retains.swap(inv.primary_retains);
  }

  unsigned failed = 0;
  Current current;

  // Work still running would be reading the memory about to be freed. The
  // contexts are synchronised first, once each, rather than per resource.
  {
    Inventory::Items* maps[] = {&allocs, &modules, &streams,
                                &events, &graphs,  &execs};
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
  const unsigned allocs_released =
      release_items(allocs, "allocation", destroy_alloc, &current, &failed);

  // Nothing below runs inside a context, and destroying the one this thread is
  // pointing at would leave it pointing at something that is gone. Contexts
  // are destroyed by handle, with none of them current.
  current.none();
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
