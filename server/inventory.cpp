#include "server/inventory.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

#include "server/client_threads.h"
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

// --- primary-context generations -------------------------------------------
//
// A primary context is shared by every session in the process, and it is
// destroyed with everything in it by the last release anywhere in the process,
// or by a reset. The session that made that release can correct its own
// record, but other sessions may still list things they made in the context
// before releasing their own retains, and if one of them freed those at expiry
// it would be freeing destroyed state - or, once a third session has retained
// the context again and allocated, that session's memory.
//
// So each device's primary context has a generation, bumped whenever the
// server sees it destroyed, and every entry made in a primary context records
// the generation it was made in. At expiry an entry from an older generation is
// skipped, not freed.

// Serialises everything that can end a primary context's generation or start a
// new one - a session's retain, a release together with the question whether
// it was the last, a reset, and the releases and generation-checked frees at
// expiry - so that nothing can revive a context between its destruction and
// the bump that records it, or destroy one between the check and a free. It is
// held across those driver calls, which is the point. Lock order: g_live_mu,
// then this, then g_gen_mu, then an inventory's own mutex.
std::mutex g_primary_mu;

// The tables themselves. Their own lock, and never held across a driver call,
// because they are read on every allocation, which must not wait for another
// session's retain or release.
std::mutex g_gen_mu;
std::unordered_map<CUcontext, int> g_primary_dev;  // handle -> device
std::unordered_map<int, uint64_t> g_primary_gen;   // device -> generation

void learn_primary(CUcontext ctx, int dev) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(g_gen_mu);
  g_primary_dev[ctx] = dev;
}

uint64_t generation(int dev) {
  std::lock_guard<std::mutex> lk(g_gen_mu);
  auto it = g_primary_gen.find(dev);
  return it == g_primary_gen.end() ? 0 : it->second;
}

// A generation no primary context ever reaches, so one that has always already
// ended. It marks an entry whose placement is not known: see stamp_of.
constexpr uint64_t kEndedGeneration = ~uint64_t{0};

// Called with g_primary_mu held, straight after the driver call that destroyed
// the context.
void bump_generation(int dev) {
  std::lock_guard<std::mutex> lk(g_gen_mu);
  g_primary_gen[dev]++;
}

// Where something is about to be made: the current context and, if that is a
// primary context, its device and generation. Taken before the driver call, so
// that a generation that ends while the call runs is charged to the entry. The
// wrong way round, an entry made in a context that was then destroyed would
// carry the new generation and be freed at expiry; this way the worst case is
// that something made in the new generation is skipped and leaks.
using Stamp = InventoryStamp;  // see inventory.h

Stamp stamp_context(CUcontext ctx) {
  Stamp st;
  st.ctx = ctx;
  std::lock_guard<std::mutex> lk(g_gen_mu);
  auto d = g_primary_dev.find(st.ctx);
  if (d != g_primary_dev.end()) {
    st.dev = d->second;
    auto g = g_primary_gen.find(st.dev);
    st.gen = g == g_primary_gen.end() ? 0 : g->second;
  }
  return st;
}

Stamp stamp() { return stamp_context(current_ctx()); }

// Whether an entry is still what it was when it was made: always, unless it
// was made in a primary context whose generation has since ended, or it was
// stamped as already ended because nobody knows where it was made.
bool still_current(int dev, uint64_t gen) {
  if (gen == kEndedGeneration) return false;
  return dev < 0 || generation(dev) == gen;
}

// --- recording ------------------------------------------------------------

// Where something made from another object is about to be made: where that
// object was recorded. A graph captured on a stream, a clone of a graph and an
// executable instantiated from one live in their source's context, not in
// whatever context happens to be current, so they have to be recorded there -
// otherwise the destruction of the source's primary context would go unseen
// for them, and they would be freed again at expiry, while the destruction of
// the current one would skip them and leak them. That they live in the
// source's context is inferred, not documented; the fake driver models it the
// same way.
//
// A null source is the current context's default stream, so it is recorded in
// the current context. A non-null source this session has no record of - a
// handle it never made, such as another session's - lives who knows where, and
// whatever destroys that context will not be seen from here. Recorded in the
// current context, the object would be freed at expiry even if its real
// context had long since been destroyed with it: a stale free, of what may by
// then be somebody else's. So it is recorded as made in a generation that has
// already ended, and expiry skips it. The worst case is that the object leaks,
// which is the direction every uncertainty in this file is made to fail in.
Stamp stamp_of(Inventory::Items Inventory::*which, uint64_t source) {
  Inventory* inv = t_inv;
  if (!inv || !source) return stamp();
  {
    std::lock_guard<std::mutex> lk(inv->mu);
    auto it = (inv->*which).find(source);
    if (it != (inv->*which).end()) {
      Stamp st;
      st.ctx = it->second.ctx;
      st.dev = it->second.dev;
      st.gen = it->second.gen;
      return st;
    }
  }
  Stamp st = stamp();
  st.gen = kEndedGeneration;
  return st;
}

void note(Inventory::Items Inventory::*which, uint64_t handle,
          const Stamp& st) {
  Inventory* inv = t_inv;
  if (!inv || !handle) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  (inv->*which)[handle] = Inventory::Item{st.ctx, st.dev, st.gen};
}

void forget(Inventory::Items Inventory::*which, uint64_t handle) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  (inv->*which).erase(handle);
}

void note_primary_retain(CUdevice dev, int delta) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
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
  for (auto it = inv->captures.begin(); it != inv->captures.end();) {
    it = it->where.ctx == ctx ? inv->captures.erase(it) : std::next(it);
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
// destroying the context, and the last release of a primary context in the
// process destroys it outright; either way the session's record of what was in
// there is suddenly wrong. Called with g_primary_mu held, after the generation
// has been bumped: everything this session made on the device in an earlier
// generation is forgotten here, so the client freeing it later is not tracked
// and expiry does not have to skip it. Other sessions' records are left as
// they are, and expiry skips their entries by generation.
//
// Only the resources go. The retains stay: the driver says "Resetting the
// primary context does not release it, an application that has retained the
// primary context should explicitly release its usage", and that "it is safe
// for other modules to call cuDevicePrimaryCtxRelease() even after resetting
// the device". So every retain this session held is still held, and expiry
// owes exactly that many releases; forgetting them would leak them for the
// life of the server. After a last release the session holds none anyway.
void forget_primary(CUdevice dev) {
  Inventory* inv = t_inv;
  if (!inv) return;
  const uint64_t now = generation(dev);
  std::lock_guard<std::mutex> lk(inv->mu);
  Inventory::Items* maps[] = {&inv->allocs,  &inv->modules, &inv->streams,
                              &inv->events,  &inv->graphs,  &inv->graph_execs};
  for (Inventory::Items* m : maps) {
    for (auto it = m->begin(); it != m->end();) {
      const bool gone = it->second.dev == dev && it->second.gen != now;
      it = gone ? m->erase(it) : std::next(it);
    }
  }
  for (auto it = inv->handles.begin(); it != inv->handles.end();) {
    const bool gone = it->second.dev == dev && it->second.gen != now;
    it = gone ? inv->handles.erase(it) : std::next(it);
  }
  for (auto it = inv->captures.begin(); it != inv->captures.end();) {
    const bool gone = it->where.dev == dev && it->where.gen != now;
    it = gone ? inv->captures.erase(it) : std::next(it);
  }
}

// --- the wrappers ---------------------------------------------------------
//
// Each one is the driver's own call with a line of bookkeeping after it. They
// are what driver_sym() hands to the generated dispatch, so a call is recorded
// exactly where it happens, with the driver's own types, and no generated code
// has to know.

CUresult w_cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytes) {
  REAL("cuMemAlloc_v2", CUdeviceptr*, size_t);
  const Stamp st = stamp();
  CUresult r = fn(dptr, bytes);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr, st);
  return r;
}

CUresult w_cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pitch, size_t width,
                              size_t height, unsigned int elem) {
  REAL("cuMemAllocPitch_v2", CUdeviceptr*, size_t*, size_t, size_t,
       unsigned int);
  const Stamp st = stamp();
  CUresult r = fn(dptr, pitch, width, height, elem);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr, st);
  return r;
}

CUresult w_cuMemAllocAsync(CUdeviceptr* dptr, size_t bytes, CUstream stream) {
  REAL("cuMemAllocAsync", CUdeviceptr*, size_t, CUstream);
  const Stamp st = stamp();
  CUresult r = fn(dptr, bytes, stream);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr, st);
  return r;
}

CUresult w_cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytes,
                                   CUmemoryPool pool, CUstream stream) {
  REAL("cuMemAllocFromPoolAsync", CUdeviceptr*, size_t, CUmemoryPool, CUstream);
  const Stamp st = stamp();
  CUresult r = fn(dptr, bytes, pool, stream);
  if (r == CUDA_SUCCESS && dptr) note(&Inventory::allocs, *dptr, st);
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

// A retain can bring a destroyed primary context back, so it takes
// g_primary_mu: a release that has just destroyed the context gets to record
// that before anything can start a new generation.
CUresult w_cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  REAL("cuDevicePrimaryCtxRetain", CUcontext*, CUdevice);
  std::lock_guard<std::mutex> lk(g_primary_mu);
  CUresult r = fn(pctx, dev);
  if (r == CUDA_SUCCESS) {
    learn_primary(pctx ? *pctx : nullptr, dev);
    note_primary_retain(dev, 1);
  }
  return r;
}

// Whether the device's primary context is inactive, as the driver reports it.
// Called with g_primary_mu held, straight after a release, so the answer is
// about that release. If the state cannot be read the answer is no, and
// nothing is forgotten or skipped: a free the driver refuses is the lesser
// mistake next to skipping everything a live context still holds.
bool primary_inactive(CUdevice dev) {
  static auto get_state =
      reinterpret_cast<CUresult (*)(CUdevice, unsigned int*, int*)>(
          driver_sym_raw("cuDevicePrimaryCtxGetState"));
  unsigned int flags = 0;
  int active = 1;
  return get_state && get_state(dev, &flags, &active) == CUDA_SUCCESS &&
         !active;
}

// Releasing the last retain in the process destroys the primary context and
// everything in it - "The context is automatically reset once the last
// reference to it is released". This session's own count cannot say whether
// this release was the last one, because other sessions may hold retains on
// the same context, so the driver is asked instead, before anything else can
// retain it.
CUresult w_cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  REAL("cuDevicePrimaryCtxRelease_v2", CUdevice);
  std::lock_guard<std::mutex> lk(g_primary_mu);
  CUresult r = fn(dev);
  if (r != CUDA_SUCCESS) return r;
  note_primary_retain(dev, -1);
  if (primary_inactive(dev)) {
    bump_generation(dev);
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
  // is this, then g_primary_mu, then the inventory's own mutex in
  // forget_primary; nothing takes them the other way round.
  std::lock_guard<std::mutex> lk(g_live_mu);
  const int live = g_live_sessions;
  if (live > 1) {
    logf("refusing cuDevicePrimaryCtxReset on device %d: %d sessions are live "
         "and it would destroy what all of them are holding",
         dev, live);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  std::lock_guard<std::mutex> primary(g_primary_mu);
  CUresult r = fn(dev);
  if (r == CUDA_SUCCESS) {
    bump_generation(dev);
    forget_primary(dev);
  }
  return r;
}

CUresult w_cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  REAL("cuCtxCreate_v2", CUcontext*, unsigned int, CUdevice);
  // Creating pushes, so a thread whose stack is full is refused before the
  // driver makes anything, with a code the call documents.
  if (!client_threads_may_create()) return CUDA_ERROR_INVALID_VALUE;
  CUresult r = fn(pctx, flags, dev);
  // Creating a context makes it current, so it is its own context: recorded
  // that way, destroying it is enough to account for everything in it.
  if (r == CUDA_SUCCESS && pctx) {
    note(&Inventory::contexts, reinterpret_cast<uint64_t>(*pctx), stamp());
    client_threads_created(*pctx);
  }
  return r;
}

CUresult w_cuCtxDestroy_v2(CUcontext ctx) {
  REAL("cuCtxDestroy_v2", CUcontext);
  CUresult r = fn(ctx);
  if (r == CUDA_SUCCESS) {
    forget_context(ctx);
    client_threads_destroyed(ctx);
  }
  return r;
}

// Deprecated, and a destroy: "Decrements the usage count of the context ctx,
// and destroys the context if the usage count goes to 0". A created context's
// count is 1, and cuCtxAttach, the only call that raises it, is refused below,
// so every detach that succeeds destroyed the context. It is accounted for
// exactly as cuCtxDestroy is: left on the list, the context would be destroyed
// again at expiry, by which time its address may be another session's
// context; left in the client threads' stacks, it would be made current again
// at that address.
CUresult w_cuCtxDetach(CUcontext ctx) {
  REAL("cuCtxDetach", CUcontext);
  CUresult r = fn(ctx);
  if (r == CUDA_SUCCESS) {
    forget_context(ctx);
    client_threads_destroyed(ctx);
  }
  return r;
}

// --- refused ----------------------------------------------------------------
//
// Calls that would hand a client a context the server cannot account for.
// Refused with CUDA_ERROR_NOT_SUPPORTED, and said once, in the same spirit as
// the reset refusal above.
//
//   - cuCtxAttach (deprecated) raises a context's usage count, which would
//     make cuCtxDetach something other than a destroy, and whether one was
//     could not be seen.
//   - Green contexts. cuCtxFromGreenCtx hands out a CUcontext that can be made
//     current and carries no record here, and cuGreenCtxDestroy destroys it -
//     "releasing the primary context of the device that this green context
//     was created for" - with no sweep of the client threads' stacks and no
//     new generation for that primary context. The whole family is refused,
//     so that no green context exists to use.

void say_refused(std::atomic<bool>& said, const char* what) {
  if (!said.exchange(true)) logf("%s", what);
}

std::atomic<bool> g_said_attach{false};
std::atomic<bool> g_said_green{false};

void say_green_refused() {
  say_refused(g_said_green,
              "refusing green contexts (cuGreenCtx*, cuCtxFromGreenCtx): a "
              "context made from one is not tracked, and destroying one "
              "releases a primary context unseen");
}

CUresult w_cuCtxAttach(CUcontext*, unsigned int) {
  say_refused(g_said_attach,
              "refusing cuCtxAttach: it is deprecated, and a context whose "
              "usage count it raised would not be destroyed by cuCtxDetach, "
              "which the server could not tell");
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult w_cuGreenCtxCreate(CUgreenCtx*, CUdevResourceDesc, CUdevice,
                            unsigned int) {
  say_green_refused();
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult w_cuGreenCtxDestroy(CUgreenCtx) {
  say_green_refused();
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult w_cuCtxFromGreenCtx(CUcontext*, CUgreenCtx) {
  say_green_refused();
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult w_cuGreenCtxStreamCreate(CUstream*, CUgreenCtx, unsigned int, int) {
  say_green_refused();
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult w_cuGreenCtxRecordEvent(CUgreenCtx, CUevent) {
  say_green_refused();
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult w_cuGreenCtxWaitEvent(CUgreenCtx, CUevent) {
  say_green_refused();
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult w_cuModuleLoad(CUmodule* mod, const char* path) {
  REAL("cuModuleLoad", CUmodule*, const char*);
  const Stamp st = stamp();
  CUresult r = fn(mod, path);
  if (r == CUDA_SUCCESS && mod) {
    note(&Inventory::modules, reinterpret_cast<uint64_t>(*mod), st);
  }
  return r;
}

CUresult w_cuModuleLoadData(CUmodule* mod, const void* image) {
  REAL("cuModuleLoadData", CUmodule*, const void*);
  const Stamp st = stamp();
  CUresult r = fn(mod, image);
  if (r == CUDA_SUCCESS && mod) {
    note(&Inventory::modules, reinterpret_cast<uint64_t>(*mod), st);
  }
  return r;
}

CUresult w_cuModuleLoadFatBinary(CUmodule* mod, const void* image) {
  REAL("cuModuleLoadFatBinary", CUmodule*, const void*);
  const Stamp st = stamp();
  CUresult r = fn(mod, image);
  if (r == CUDA_SUCCESS && mod) {
    note(&Inventory::modules, reinterpret_cast<uint64_t>(*mod), st);
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
  const Stamp st = stamp();
  CUresult r = fn(stream, flags);
  if (r == CUDA_SUCCESS && stream) {
    note(&Inventory::streams, reinterpret_cast<uint64_t>(*stream), st);
  }
  return r;
}

CUresult w_cuStreamCreateWithPriority(CUstream* stream, unsigned int flags,
                                      int priority) {
  REAL("cuStreamCreateWithPriority", CUstream*, unsigned int, int);
  const Stamp st = stamp();
  CUresult r = fn(stream, flags, priority);
  if (r == CUDA_SUCCESS && stream) {
    note(&Inventory::streams, reinterpret_cast<uint64_t>(*stream), st);
  }
  return r;
}

// Forgets the open capture on `stream`: for the default stream (0), the one in
// context `ctx`.
void forget_capture(uint64_t stream, CUcontext ctx) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  for (auto it = inv->captures.begin(); it != inv->captures.end(); ++it) {
    if (it->stream == stream && (stream != 0 || it->where.ctx == ctx)) {
      inv->captures.erase(it);
      return;
    }
  }
}

CUresult w_cuStreamDestroy_v2(CUstream stream) {
  REAL("cuStreamDestroy_v2", CUstream);
  CUresult r = fn(stream);
  if (r == CUDA_SUCCESS) {
    forget(&Inventory::streams, reinterpret_cast<uint64_t>(stream));
    // A stream that is gone has no capture to end. Destroying one in capture
    // is not documented either way; this assumes the capture goes with it.
    if (stream) forget_capture(reinterpret_cast<uint64_t>(stream), nullptr);
  }
  return r;
}

CUresult w_cuEventCreate(CUevent* event, unsigned int flags) {
  REAL("cuEventCreate", CUevent*, unsigned int);
  const Stamp st = stamp();
  CUresult r = fn(event, flags);
  if (r == CUDA_SUCCESS && event) {
    note(&Inventory::events, reinterpret_cast<uint64_t>(*event), st);
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
  const Stamp st = stamp();
  CUresult r = fn(graph, flags);
  if (r == CUDA_SUCCESS && graph) {
    note(&Inventory::graphs, reinterpret_cast<uint64_t>(*graph), st);
  }
  return r;
}

// A capture begun is recorded where its stream is, so that expiry can end it
// if the client never does.
CUresult w_cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode) {
  REAL("cuStreamBeginCapture_v2", CUstream, CUstreamCaptureMode);
  const Stamp st =
      stamp_of(&Inventory::streams, reinterpret_cast<uint64_t>(stream));
  CUresult r = fn(stream, mode);
  Inventory* inv = t_inv;
  if (r == CUDA_SUCCESS && inv) {
    std::lock_guard<std::mutex> lk(inv->mu);
    Inventory::OpenCapture c;
    c.stream = reinterpret_cast<uint64_t>(stream);
    c.where = Inventory::Item{st.ctx, st.dev, st.gen};
    inv->captures.push_back(c);
  }
  return r;
}

// A capture hands the finished graph to the client, which owns it from here
// exactly as if it had created one.
CUresult w_cuStreamEndCapture(CUstream stream, CUgraph* graph) {
  REAL("cuStreamEndCapture", CUstream, CUgraph*);
  const Stamp st =
      stamp_of(&Inventory::streams, reinterpret_cast<uint64_t>(stream));
  CUresult r = fn(stream, graph);
  if (r == CUDA_SUCCESS) {
    forget_capture(reinterpret_cast<uint64_t>(stream), st.ctx);
    if (graph) {
      note(&Inventory::graphs, reinterpret_cast<uint64_t>(*graph), st);
    }
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
  const Stamp st =
      stamp_of(&Inventory::graphs, reinterpret_cast<uint64_t>(graph));
  CUresult r = fn(exec, graph, flags);
  if (r == CUDA_SUCCESS && exec) {
    note(&Inventory::graph_execs, reinterpret_cast<uint64_t>(*exec), st);
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
  const Stamp st =
      stamp_of(&Inventory::graphs, reinterpret_cast<uint64_t>(original));
  CUresult r = fn(clone, original);
  if (r == CUDA_SUCCESS && clone) {
    note(&Inventory::graphs, reinterpret_cast<uint64_t>(*clone), st);
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
    {"cuCtxDetach", reinterpret_cast<void*>(&w_cuCtxDetach)},
    // Refused, not tracked; see above.
    {"cuCtxAttach", reinterpret_cast<void*>(&w_cuCtxAttach)},
    {"cuGreenCtxCreate", reinterpret_cast<void*>(&w_cuGreenCtxCreate)},
    {"cuGreenCtxDestroy", reinterpret_cast<void*>(&w_cuGreenCtxDestroy)},
    {"cuCtxFromGreenCtx", reinterpret_cast<void*>(&w_cuCtxFromGreenCtx)},
    {"cuGreenCtxStreamCreate",
     reinterpret_cast<void*>(&w_cuGreenCtxStreamCreate)},
    {"cuGreenCtxRecordEvent", reinterpret_cast<void*>(&w_cuGreenCtxRecordEvent)},
    {"cuGreenCtxWaitEvent", reinterpret_cast<void*>(&w_cuGreenCtxWaitEvent)},
    {"cuModuleLoad",reinterpret_cast<void*>(&w_cuModuleLoad)},
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
    {"cuStreamBeginCapture_v2",
     reinterpret_cast<void*>(&w_cuStreamBeginCapture_v2)},
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
CUresult end_capture(uint64_t stream, CUgraph* graph) {
  REAL("cuStreamEndCapture", CUstream, CUgraph*);
  return fn(reinterpret_cast<CUstream>(stream), graph);
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

// Something made in a primary context whose generation has ended: destroyed
// with the context, by another session's last release or expiry if not this
// one's, so it is not this session's to free. Logged, and counted apart from
// failures, because nothing went wrong.
void log_skipped(const char* what, uint64_t handle, int dev, uint64_t gen) {
  if (gen == kEndedGeneration) {
    logf("session cleanup: %s %llx was not released: it was made from an "
         "object this session did not make, so where it lives is not known",
         what, (unsigned long long)handle);
    return;
  }
  logf("session cleanup: %s %llx was not released: device %d's primary "
       "context was destroyed after it was made",
       what, (unsigned long long)handle, dev);
}

// Runs `destroy` for an entry unless its primary context's generation has
// ended, in which case it returns false. For an entry made in a primary
// context the check and the destroy happen under g_primary_mu, so no release
// or reset can destroy the context between them, and no retain can recreate it
// with somebody else's memory at the same address.
template <typename F>
bool unless_destroyed(int dev, uint64_t gen, F destroy) {
  if (gen == kEndedGeneration) return false;
  if (dev < 0) {
    destroy();
    return true;
  }
  std::lock_guard<std::mutex> lk(g_primary_mu);
  if (!still_current(dev, gen)) return false;
  destroy();
  return true;
}

unsigned release_items(Inventory::Items& items, const char* what,
                       CUresult (*destroy)(uint64_t), Current* current,
                       unsigned* failed, unsigned* skipped) {
  unsigned released = 0;
  for (const auto& entry : items) {
    CUresult r = CUDA_SUCCESS;
    const bool ran =
        unless_destroyed(entry.second.dev, entry.second.gen, [&] {
          current->use(entry.second.ctx);
          r = destroy(entry.first);
        });
    if (!ran) {
      (*skipped)++;
      log_skipped(what, entry.first, entry.second.dev, entry.second.gen);
      continue;
    }
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

InventoryStamp inventory_stamp() { return stamp(); }

void inventory_note_handle(uint64_t handle, const InventoryStamp& made,
                           const char* what, CUresult (*destroy)(uint64_t)) {
  Inventory* inv = t_inv;
  if (!inv || !handle) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  inv->handles[handle] =
      Inventory::LibHandle{made.ctx, made.dev, made.gen, what, destroy};
}

void inventory_forget_handle(uint64_t handle) {
  Inventory* inv = t_inv;
  if (!inv) return;
  std::lock_guard<std::mutex> lk(inv->mu);
  inv->handles.erase(handle);
}

void* driver_wrapper(const char* name) {
  // Calls that change which context a client thread has current, carried out
  // on that thread's own stack. None of them is also tracked here.
  if (void* fn = client_threads_wrapper(name)) return fn;
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
  std::vector<Inventory::OpenCapture> captures;
  {
    std::lock_guard<std::mutex> lk(inv.mu);
    captures.swap(inv.captures);
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
  unsigned skipped = 0;
  Current current;

  // Captures the client left open are ended before anything else: until they
  // are, one begun other than RELAXED restricts this thread, which began it,
  // and the frees below could be refused. Each is ended where its stream is,
  // and the graph that comes out is the session's to release like any other.
  // One on a stream of unknown placement - another session's - is left alone,
  // as everything made from such a stream is.
  unsigned captures_ended = 0;
  for (const Inventory::OpenCapture& c : captures) {
    CUresult r = CUDA_SUCCESS;
    CUgraph graph = nullptr;
    const bool ran = unless_destroyed(c.where.dev, c.where.gen, [&] {
      current.use(c.where.ctx);
      r = end_capture(c.stream, &graph);
    });
    if (!ran) {
      skipped++;
      log_skipped("stream capture on stream", c.stream, c.where.dev,
                  c.where.gen);
      continue;
    }
    if (r != CUDA_SUCCESS) {
      failed++;
      logf("session cleanup: the stream capture on stream %llx could not be "
           "ended (%d)",
           (unsigned long long)c.stream, r);
      continue;
    }
    captures_ended++;
    if (graph) graphs[reinterpret_cast<uint64_t>(graph)] = c.where;
  }

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
        // A destroyed primary context has nothing of this session's to wait
        // for.
        if (!still_current(entry.second.dev, entry.second.gen)) continue;
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
    CUresult r = CUDA_SUCCESS;
    const bool ran =
        unless_destroyed(entry.second.dev, entry.second.gen, [&] {
          current.use(entry.second.ctx);
          r = entry.second.destroy ? entry.second.destroy(entry.first)
                                   : CUDA_ERROR_NOT_SUPPORTED;
        });
    if (!ran) {
      skipped++;
      log_skipped(entry.second.what, entry.first, entry.second.dev,
                  entry.second.gen);
      continue;
    }
    if (r == CUDA_SUCCESS) {
      handles_released++;
    } else {
      failed++;
      logf("session cleanup: %s handle %llx was not destroyed (%d)",
           entry.second.what, (unsigned long long)entry.first, r);
    }
  }

  const unsigned execs_released =
      release_items(execs, "graph exec", destroy_graph_exec, &current, &failed,
                    &skipped);
  const unsigned graphs_released =
      release_items(graphs, "graph", destroy_graph, &current, &failed,
                    &skipped);
  const unsigned events_released =
      release_items(events, "event", destroy_event, &current, &failed,
                    &skipped);
  const unsigned streams_released =
      release_items(streams, "stream", destroy_stream, &current, &failed,
                    &skipped);
  const unsigned modules_released =
      release_items(modules, "module", destroy_module, &current, &failed,
                    &skipped);
  const unsigned allocs_released =
      release_items(allocs, "allocation", destroy_alloc, &current, &failed,
                    &skipped);

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
      CUresult r;
      {
        // This release may be the last in the process, which destroys the
        // context and everything any session made in it, so it is recorded
        // the same way as a client's own release.
        std::lock_guard<std::mutex> lk(g_primary_mu);
        r = release_primary(entry.first);
        if (r == CUDA_SUCCESS && primary_inactive(entry.first)) {
          bump_generation(entry.first);
        }
      }
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
  if (captures_ended) {
    summary = "ended " + plural(captures_ended, "open stream capture") + "; " +
              summary;
  }
  if (skipped) {
    summary += "; " + plural(skipped, "resource") +
               " skipped, destroyed with a primary context or of unknown "
               "placement";
  }
  if (failed) {
    summary += "; " + plural(failed, "resource") + " could not be released";
  }
  return summary;
}

}  // namespace rgpu
