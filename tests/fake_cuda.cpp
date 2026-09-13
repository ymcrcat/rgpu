// A driver that pretends to be a GPU, backed by host memory.
//
// Linked into rgpu-server-fake so the whole stack can be exercised without a
// GPU: real client stubs, real wire format, real server dispatch. Only the
// bottom of the stack is fake.
//
// These definitions are strong and override the weak ones in
// tests/generated/fake_driver.cpp.
//
// Contexts are modelled the way the driver documents them, because a fake that
// is loose about them hides exactly the class of bug that issue #2 is:
//
//   - There are RGPU_FAKE_DEVICES devices (default 1), each with a primary
//     context of its own, with its own retain count, released and reset
//     independently of every other device.
//   - The current context is state of the calling OS thread, kept as a stack,
//     as the real driver keeps it. In the server one thread serves a session,
//     so that is where this state lives there too; nothing here is shortened
//     to "one current context per process".
//   - Everything the fake hands out - memory, modules, streams, events, graphs
//     - belongs to the context that was current when it was made, and carries
//     that context with it: a call on it acts in its own context whichever
//     one is current, as the driver infers placement from a pointer's value.
//     Destroying the context, or resetting a primary context, destroys what
//     is in it. Which context and device own an allocation can be read back
//     through the pointer attributes.
//
// What is deliberately not modelled: peer access between contexts, and
// cuMemAllocAsync-style allocations that belong to no context.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "tests/fake_stats.h"

namespace {

std::mutex g_mu;

// --- devices ---------------------------------------------------------------

constexpr int kMaxDevices = 16;
constexpr int kFakeDriverVersion = 12080;

// Read once. A value that does not parse is a broken test setup, and running
// on with one device instead would turn it into a confusing pass or fail
// somewhere else, so it stops the process instead.
int device_count() {
  static const int n = [] {
    const char* s = std::getenv("RGPU_FAKE_DEVICES");
    if (!s || !*s) return 1;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (*end != '\0' || v < 1 || v > kMaxDevices) {
      std::fprintf(stderr,
                   "[fake cuda] RGPU_FAKE_DEVICES=%s is not a device count "
                   "between 1 and %d\n",
                   s, kMaxDevices);
      std::abort();
    }
    return static_cast<int>(v);
  }();
  return n;
}

bool valid_device(CUdevice dev) { return dev >= 0 && dev < device_count(); }

// --- contexts --------------------------------------------------------------

// A primary context's handle. One per device and stable for the life of the
// process, which is what the driver does too: retaining again after a release
// or a reset gives back the same handle.
constexpr unsigned long long kPrimaryBase = 0xC0FFEE01ull;

CUcontext primary_token(CUdevice dev) {
  return reinterpret_cast<CUcontext>(kPrimaryBase + static_cast<unsigned>(dev));
}

// The device whose primary context this is, or -1 for anything else.
int primary_device(CUcontext ctx) {
  const auto v = reinterpret_cast<unsigned long long>(ctx);
  if (v < kPrimaryBase || v >= kPrimaryBase + device_count()) return -1;
  return static_cast<int>(v - kPrimaryBase);
}

// Per device, how many retains the whole process holds. A primary context is
// shared, so this is the count that must not go negative, and while it is zero
// the context is not initialised and nothing can run in it. Only retains and
// releases move it; a reset does not.
std::map<int, int> g_primary_retains;

// Contexts the client created, and the device each is on.
std::map<unsigned long long, CUdevice> g_contexts;

unsigned long long g_next_handle = 1;

// Handles of destroyed contexts, most recent last, for RGPU_FAKE_REUSE_CONTEXTS.
//
// A CUcontext is a pointer to driver heap state, and nothing in the header
// promises that a destroyed context's address is never handed out again: the
// next cuCtxCreate anywhere in the process may well get it. By default this
// fake never reuses one, so that a stale handle is always recognisable; with
// RGPU_FAKE_REUSE_CONTEXTS=1 a create takes the most recently destroyed handle
// instead, so a test can show that a stale handle is never mistaken for the
// new context that now has its address.
std::vector<unsigned long long> g_destroyed_contexts;

bool reuse_contexts() {
  static const bool on = [] {
    const char* s = std::getenv("RGPU_FAKE_REUSE_CONTEXTS");
    return s && std::strcmp(s, "1") == 0;
  }();
  return on;
}

// The calling thread's context stack; the top is the current context. A
// thread starts with none. Only the thread itself touches this, so it needs no
// lock, but the contexts named in it can be destroyed by any thread, which is
// why every use checks them against the tables above.
thread_local std::vector<CUcontext> t_stack;

CUcontext current() { return t_stack.empty() ? nullptr : t_stack.back(); }

// Whether a handle names a context at all, initialised or not. This is the
// test for binding one to a thread. Called with g_mu held.
bool bindable_locked(CUcontext ctx) {
  if (primary_device(ctx) >= 0) return true;
  return g_contexts.count(reinterpret_cast<unsigned long long>(ctx)) != 0;
}

// The context a call runs in, or why it cannot run. Called with g_mu held.
//
// Two different failures, as the driver documents them: no context at all is
// CUDA_ERROR_INVALID_CONTEXT, and a current context that has since been
// destroyed - by another thread, which leaves it current here - is
// CUDA_ERROR_CONTEXT_IS_DESTROYED. A primary
// context nobody holds a retain on is "not yet initialised", which the driver
// reports the same way.
CUresult enter_locked(CUcontext* ctx) {
  *ctx = current();
  if (!*ctx) return CUDA_ERROR_INVALID_CONTEXT;
  const int dev = primary_device(*ctx);
  if (dev >= 0) {
    auto it = g_primary_retains.find(dev);
    return it != g_primary_retains.end() && it->second > 0
               ? CUDA_SUCCESS
               : CUDA_ERROR_CONTEXT_IS_DESTROYED;
  }
  return g_contexts.count(reinterpret_cast<unsigned long long>(*ctx))
             ? CUDA_SUCCESS
             : CUDA_ERROR_CONTEXT_IS_DESTROYED;
}

CUresult need_context() {
  std::lock_guard<std::mutex> lk(g_mu);
  CUcontext ctx = nullptr;
  return enter_locked(&ctx);
}

// --- what lives in a context -------------------------------------------------
//
// A real driver knows which of its handles are still alive and which context
// each belongs to; this one has to as well, or a test could not tell a handle
// that was released from one that was merely forgotten, or one used in the
// right place from one used in the wrong one.
//
// What a real driver does with an object that belongs to another context: it
// uses it there. The header's Unified Addressing overview says "Since pointers
// are unique, it is not necessary to specify information about the pointers
// specified to the various copy functions", and that the driver "should infer
// the location of the pointer from its value". Streams, events, modules and
// graphs are handles to one context's objects in the same way, and nothing in
// the header confines their use to the current context. So a copy, a free, a
// stream synchronise or a destroy of another context's object succeeds here.
// A fake stricter than the hardware would let a test pass for a reason the
// hardware does not share.
//
// cuModuleUnload reads "Unloads a module hmod from the current context", but
// on hardware it unloads a module loaded under another context while a
// different one is current (real-GPU probe, check 4), so the fake identifies a
// module by its handle the way it does an allocation, not by the current
// context. Its return list names CUDA_ERROR_INVALID_VALUE, which the fake
// gives for a handle that names no module.
//
// One call is stricter:
//   - cuLaunchKernel with a stream whose context is not the function's is
//     CUDA_ERROR_INVALID_HANDLE. The header says it for cuLaunchKernelEx:
//     "The CUDA context associated with this stream must match that
//     associated with function f". With a null stream the launch is refused
//     unless the function's context is current. That case is INFERRED: the
//     only text on it ("the context to launch the kernel on will either be
//     taken from the specified stream hStream or the current context in case
//     of NULL stream") is written about context-less CUkernel handles, and is
//     read here as saying a null stream means the current context's.
//
// Objects made from other objects - a graph captured on a stream, a clone of
// a graph, an executable instantiated from one - belong to no context at all.
// The real-GPU probe (check 6) confirmed that each survives the destruction of
// both the capture context and the current context: it lives until it is
// explicitly destroyed. So the fake records them under no context (nullptr),
// and a context's destruction or a primary-context reset never takes them.
// (A launch of such an object still needs some context current, as any call
// here does.)
//
// What still needs a context is unchanged: a call the header documents
// CUDA_ERROR_INVALID_CONTEXT for still needs one current and usable. A handle
// that names nothing - never issued, or destroyed with its context - is
// CUDA_ERROR_INVALID_VALUE for memory, graphs and module unloads, and
// CUDA_ERROR_INVALID_HANDLE for streams, events and other module calls.

// Device pointers are not host addresses. They are handed out from a range no
// host allocation can occupy - above 48 bits, so dereferencing one on the host
// faults instead of scribbling on the heap - and never handed out twice, so a
// free of an address that was freed before is always recognisable as stale.
// A malloc address would come back from the next malloc, and a stale free of
// it would free somebody else's allocation without a sound.
constexpr CUdeviceptr kDeviceBase = 0x00DE000000000000ull;
constexpr CUdeviceptr kPage = 4096;
CUdeviceptr g_next_ptr = kDeviceBase;

struct Alloc {
  size_t size;
  CUcontext ctx;
  void* host;  // the bytes behind it
};
std::map<CUdeviceptr, Alloc> g_allocs;

// Handle -> the context it lives in.
using Objects = std::map<unsigned long long, CUcontext>;
Objects g_modules, g_streams, g_events, g_graphs, g_graph_execs;

// Hands out a distinct token per creation, tagged so a value that turns up in
// the wrong place is recognisable in a log, and records it in `ctx`. Called
// with g_mu held.
unsigned long long mint_in_locked(Objects* into, unsigned tag, CUcontext ctx) {
  const unsigned long long h =
      (static_cast<unsigned long long>(tag) << 32) | g_next_handle++;
  (*into)[h] = ctx;
  return h;
}

// The same, in the current context.
CUresult mint(Objects* into, unsigned tag, unsigned long long* out) {
  std::lock_guard<std::mutex> lk(g_mu);
  CUcontext ctx = nullptr;
  CUresult r = enter_locked(&ctx);
  if (r != CUDA_SUCCESS) return r;
  *out = mint_in_locked(into, tag, ctx);
  return CUDA_SUCCESS;
}

// Whether `h` is a live object of this kind, and if so the context it lives
// in, which need not be the current one. A context has to be current all the
// same. `missing` is what this kind of call reports for a handle it does not
// know. Called with g_mu held.
CUresult owned_locked(const Objects& in, unsigned long long h,
                      CUresult missing, CUcontext* owner = nullptr) {
  CUcontext ctx = nullptr;
  CUresult r = enter_locked(&ctx);
  if (r != CUDA_SUCCESS) return r;
  auto it = in.find(h);
  if (it == in.end()) return missing;
  if (owner) *owner = it->second;
  return CUDA_SUCCESS;
}

CUresult owned(const Objects& in, unsigned long long h, CUresult missing,
               CUcontext* owner = nullptr) {
  std::lock_guard<std::mutex> lk(g_mu);
  return owned_locked(in, h, missing, owner);
}

// A null stream is the current context's default stream; any other is a live
// stream in whatever context it was made in. `ctx` gets the context the
// stream runs in.
CUresult stream_ok_locked(CUstream s, CUcontext* ctx = nullptr) {
  if (!s) {
    CUcontext cur = nullptr;
    CUresult r = enter_locked(&cur);
    if (r == CUDA_SUCCESS && ctx) *ctx = cur;
    return r;
  }
  return owned_locked(g_streams, reinterpret_cast<unsigned long long>(s),
                      CUDA_ERROR_INVALID_HANDLE, ctx);
}

CUresult stream_ok(CUstream s, CUcontext* ctx = nullptr) {
  std::lock_guard<std::mutex> lk(g_mu);
  return stream_ok_locked(s, ctx);
}

// Destroys an object, in whatever context it lives in - or, with
// `current_only`, only if that is the current context. Destroying something
// that was never created, or already destroyed, is an error here rather than a
// shrug, and it is counted: on a real driver the handle may belong to somebody
// else by now. Counted whether or not a context is current, because the counter
// is there to catch the mistake, not to mirror the return code.
CUresult retire(Objects* from, unsigned long long h, CUresult missing,
                bool current_only = false) {
  bool known;
  CUresult r;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = from->find(h);
    known = it != from->end();
    CUcontext ctx = nullptr;
    r = enter_locked(&ctx);
    if (r == CUDA_SUCCESS) {
      if (known && (!current_only || it->second == ctx)) {
        from->erase(it);
      } else {
        r = missing;
      }
    }
  }
  if (!known) rgpu_fake::count(rgpu_fake::kStale, 1);
  return r;
}

// The allocation containing device address p, or g_allocs.end(). Called with
// g_mu held.
std::map<CUdeviceptr, Alloc>::iterator containing_locked(CUdeviceptr p) {
  auto it = g_allocs.upper_bound(p);
  if (it == g_allocs.begin()) return g_allocs.end();
  --it;
  if (static_cast<size_t>(p - it->first) >= it->second.size) {
    return g_allocs.end();
  }
  return it;
}

// The device memory [p, p+n), in whichever context owns it, as host bytes. A
// context has to be current, as the copy calls document
// CUDA_ERROR_INVALID_CONTEXT, but it need not be the owner. Called with g_mu
// held; the caller copies under the same lock, so the allocation cannot be
// freed by another thread in between.
CUresult range_locked(CUdeviceptr p, size_t n, void** host) {
  CUcontext ctx = nullptr;
  CUresult r = enter_locked(&ctx);
  if (r != CUDA_SUCCESS) return r;
  auto it = containing_locked(p);
  if (it == g_allocs.end()) return CUDA_ERROR_INVALID_VALUE;
  const Alloc& a = it->second;
  const size_t off = static_cast<size_t>(p - it->first);
  if (n > a.size - off) return CUDA_ERROR_INVALID_VALUE;
  *host = static_cast<char*>(a.host) + off;
  // Allowed, and counted: the only trace a test has of work that ran under a
  // context other than the one its issuer selected. The counter's lock is its
  // own, and it takes no lock of ours.
  if (a.ctx != ctx) rgpu_fake::count(rgpu_fake::kCrossContextUse, 1);
  return CUDA_SUCCESS;
}

// Everything in one context, taken out of the tables. What destroying a
// context, resetting a primary context, or releasing its last retain does to
// the things inside it.
struct Contents {
  std::vector<void*> host;
  int modules = 0, streams = 0, events = 0, graphs = 0, execs = 0;
  int captures = 0;
};

// --- stream captures ---------------------------------------------------------
//
// Open captures, by stream: a created stream by its handle, and a null stream
// - the current context's default stream - by that context. Each remembers
// the mode it was begun in and the thread that began it, because both decide
// what the capture permits: "If mode is not CU_STREAM_CAPTURE_MODE_RELAXED,
// cuStreamEndCapture must be called on this stream from the same thread", and
// while it is open the thread that began it "is prohibited from potentially
// unsafe API calls" unless that thread's own mode is RELAXED.
//
// An open capture is counted ("captures"), as a resource is: one left open by
// a session that went away is exactly what a test has to be able to see.
struct Capture {
  CUstreamCaptureMode mode;
  std::thread::id owner;
};
using CaptureKey = std::pair<unsigned long long, CUcontext>;
std::map<CaptureKey, Capture> g_captures;

CaptureKey capture_key(CUstream stream, CUcontext stream_ctx) {
  if (stream) return {reinterpret_cast<unsigned long long>(stream), nullptr};
  return {0, stream_ctx};
}

// The calling thread's stream capture mode, which the header describes as the
// thread's own ("A thread's mode is one of the following"), starting at the
// mode it calls the default. Only the thread itself touches it.
thread_local CUstreamCaptureMode t_capture_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;

// Whether the calling thread may make a potentially unsafe call. The header's
// rule for a thread in GLOBAL or THREAD_LOCAL mode: not while it has a capture
// of its own open that was not begun RELAXED. A thread in RELAXED mode may.
// What counts as unsafe is not listed; the header's example is an allocation,
// so the fake refuses allocating and freeing device memory, and nothing else.
// The code is INFERRED: the header names none, and CUDA_ERROR_NOT_PERMITTED
// is the one that reads right. Not modelled: another thread's capture begun
// GLOBAL restricting a thread in GLOBAL mode. Called with g_mu held.
CUresult capture_permits_locked() {
  if (t_capture_mode == CU_STREAM_CAPTURE_MODE_RELAXED) return CUDA_SUCCESS;
  const std::thread::id self = std::this_thread::get_id();
  for (const auto& c : g_captures) {
    if (c.second.owner == self &&
        c.second.mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
      return CUDA_ERROR_NOT_PERMITTED;
    }
  }
  return CUDA_SUCCESS;
}

void take_objects_locked(Objects* from, CUcontext ctx, int* n) {
  for (auto it = from->begin(); it != from->end();) {
    if (it->second == ctx) {
      it = from->erase(it);
      ++*n;
    } else {
      ++it;
    }
  }
}

Contents take_contents_locked(CUcontext ctx) {
  Contents c;
  // A capture goes with the stream it is on: the context's default stream,
  // or a stream made in the context.
  for (auto it = g_captures.begin(); it != g_captures.end();) {
    const unsigned long long stream = it->first.first;
    const auto owner = stream ? g_streams.find(stream) : g_streams.end();
    const bool in_ctx = stream ? owner != g_streams.end() && owner->second == ctx
                               : it->first.second == ctx;
    if (in_ctx) {
      it = g_captures.erase(it);
      c.captures++;
    } else {
      ++it;
    }
  }
  for (auto a = g_allocs.begin(); a != g_allocs.end();) {
    if (a->second.ctx == ctx) {
      c.host.push_back(a->second.host);
      a = g_allocs.erase(a);
    } else {
      ++a;
    }
  }
  take_objects_locked(&g_modules, ctx, &c.modules);
  take_objects_locked(&g_streams, ctx, &c.streams);
  take_objects_locked(&g_events, ctx, &c.events);
  take_objects_locked(&g_graphs, ctx, &c.graphs);
  take_objects_locked(&g_graph_execs, ctx, &c.execs);
  return c;
}

// Outside the lock: the counters publish to a file.
void settle(const Contents& c) {
  for (void* h : c.host) std::free(h);
  if (!c.host.empty()) {
    rgpu_fake::count(rgpu_fake::kAlloc, -static_cast<int>(c.host.size()));
  }
  if (c.modules) rgpu_fake::count(rgpu_fake::kModule, -c.modules);
  if (c.streams) rgpu_fake::count(rgpu_fake::kStream, -c.streams);
  if (c.events) rgpu_fake::count(rgpu_fake::kEvent, -c.events);
  if (c.graphs) rgpu_fake::count(rgpu_fake::kGraph, -c.graphs);
  if (c.execs) rgpu_fake::count(rgpu_fake::kGraphExec, -c.execs);
  if (c.captures) rgpu_fake::count(rgpu_fake::kCapture, -c.captures);
}

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
  *count = device_count();
  return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (!valid_device(ordinal)) return CUDA_ERROR_INVALID_DEVICE;
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  std::snprintf(name, static_cast<size_t>(len), "rgpu fake device %d", dev);
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
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

// Counted, and slowed down by RGPU_FAKE_SLOW_TOTALMEM_MS (off by default), for
// tests/replay_smoke.cpp: a request still running when its client reconnects
// is the one a server must not run twice. Picked because nothing depends on
// how fast it is, and it has no effects to undo. Counted before the delay, so
// a test can see it has started; an invalid device fails after the delay, so
// a failing call is as slow as a succeeding one.
CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  rgpu_fake::count(rgpu_fake::kTotalMem, 1);
  static const long delay_ms = [] {
    const char* s = std::getenv("RGPU_FAKE_SLOW_TOTALMEM_MS");
    return s ? std::atol(s) : 0L;
  }();
  if (delay_ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  *bytes = size_t(24) << 30;
  return CUDA_SUCCESS;
}

// Contexts are opaque to everything above, but they are counted here: a
// context nobody destroyed is exactly the kind of leak this fake exists to
// make visible. Creating one pushes it onto the calling thread's stack, so it
// is current and whatever was current before comes back when it is popped.
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int, CUdevice dev) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    unsigned long long h = (0xC0FFEE00ull << 32) | g_next_handle++;
    if (reuse_contexts() && !g_destroyed_contexts.empty()) {
      h = g_destroyed_contexts.back();
      g_destroyed_contexts.pop_back();
    }
    g_contexts[h] = dev;
    *pctx = reinterpret_cast<CUcontext>(h);
  }
  t_stack.push_back(*pctx);
  rgpu_fake::count(rgpu_fake::kContext, 1);
  return CUDA_SUCCESS;
}

// Destroys the context and everything in it, from any thread. If it is current
// on this thread it is popped; on any other thread it stays current, and that
// thread's next call is told the context is destroyed.
CUresult cuCtxDestroy_v2(CUcontext ctx) {
  Contents gone;
  bool known;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    known = g_contexts.erase(reinterpret_cast<unsigned long long>(ctx)) != 0;
    if (known) {
      gone = take_contents_locked(ctx);
      if (reuse_contexts()) {
        g_destroyed_contexts.push_back(reinterpret_cast<unsigned long long>(ctx));
      }
    }
  }
  if (!known) {
    rgpu_fake::count(rgpu_fake::kStale, 1);
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  if (current() == ctx) t_stack.pop_back();
  settle(gone);
  rgpu_fake::count(rgpu_fake::kContext, -1);
  return CUDA_SUCCESS;
}

// Deprecated. "Decrements the usage count of the context ctx, and destroys the
// context if the usage count goes to 0. The context must be a handle that was
// passed back by cuCtxCreate() or cuCtxAttach(), and must be current to the
// calling thread." A created context's count is 1, and cuCtxAttach, the only
// thing that raises it, is not modelled, so a detach that is allowed at all is
// a destroy.
CUresult cuCtxDetach(CUcontext ctx) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ctx || current() != ctx ||
        !g_contexts.count(reinterpret_cast<unsigned long long>(ctx))) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
  }
  return cuCtxDestroy_v2(ctx);
}

CUresult cuCtxSynchronize(void) { return need_context(); }

// Binds a context to the calling thread by replacing the top of its stack, and
// null pops it, as documented. A handle that names no context - including one
// that was destroyed - is refused. A primary context is a valid handle whether
// or not anybody holds a retain on it; it is using it uninitialised that
// fails, not binding it.
CUresult cuCtxSetCurrent(CUcontext ctx) {
  rgpu_fake::count(rgpu_fake::kCtxSetCurrent, 1);
  if (!ctx) {
    if (!t_stack.empty()) t_stack.pop_back();
    return CUDA_SUCCESS;
  }
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!bindable_locked(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
  }
  if (t_stack.empty()) {
    t_stack.push_back(ctx);
  } else {
    t_stack.back() = ctx;
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = current();
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ctx || !bindable_locked(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
  }
  t_stack.push_back(ctx);
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx) {
  if (t_stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
  if (pctx) *pctx = t_stack.back();
  t_stack.pop_back();
  return CUDA_SUCCESS;
}

// The runtime binds a primary context per device, so the fake needs these too.
// One token per device however many times it is retained, which is the whole
// point of a primary context: the count is what is owned, not the handle.
// Retaining is not pushing: the caller still has to make it current.
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_primary_retains[dev]++;
  }
  *pctx = primary_token(dev);
  rgpu_fake::count(rgpu_fake::kPrimaryRetain, 1);
  return CUDA_SUCCESS;
}

// Releasing the last retain resets the context, as the driver documents ("The
// context is automatically reset once the last reference to it is released"),
// so what was in it goes with it.
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  Contents gone;
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
    if (--it->second == 0) gone = take_contents_locked(primary_token(dev));
  }
  settle(gone);
  rgpu_fake::count(rgpu_fake::kPrimaryRetain, -1);
  return CUDA_SUCCESS;
}

// Destroys everything in the device's primary context - memory, modules,
// streams, events and graphs, which is what the server's inventory forgets for
// it - and nothing in any other context.
//
// It does not touch the retain count. The driver is explicit: "Resetting the
// primary context does not release it, an application that has retained the
// primary context should explicitly release its usage", and "it is safe for
// other modules to call cuDevicePrimaryCtxRelease() even after resetting the
// device". So every retain taken before a reset is still owed a release after
// it, the context stays active while any are held, and it is usable again
// straight away, empty.
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  Contents gone;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    gone = take_contents_locked(primary_token(dev));
  }
  settle(gone);
  return CUDA_SUCCESS;
}

// Active while anybody holds a retain, which is the state the server reads to
// learn whether a release it made was the last one in the process.
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags,
                                    int* active) {
  if (!valid_device(dev)) return CUDA_ERROR_INVALID_DEVICE;
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_primary_retains.find(dev);
  if (flags) *flags = 0;
  if (active) *active = it != g_primary_retains.end() && it->second > 0;
  return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
  if (!free || !total) return CUDA_ERROR_INVALID_VALUE;
  CUresult r = need_context();
  if (r != CUDA_SUCCESS) return r;
  *total = size_t(24) << 30;
  *free = size_t(20) << 30;
  return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize > (size_t(1) << 40)) return CUDA_ERROR_OUT_OF_MEMORY;
  void* host = std::malloc(bytesize);
  if (!host) return CUDA_ERROR_OUT_OF_MEMORY;
  CUdeviceptr d = 0;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    CUcontext ctx = nullptr;
    CUresult r = enter_locked(&ctx);
    if (r == CUDA_SUCCESS) r = capture_permits_locked();
    if (r != CUDA_SUCCESS) {
      std::free(host);
      return r;
    }
    d = g_next_ptr;
    // A page of nothing after each one, so a copy running one byte off the
    // end lands in no allocation rather than the next.
    g_next_ptr += (bytesize + kPage - 1) / kPage * kPage + kPage;
    g_allocs[d] = Alloc{bytesize, ctx, host};
  }
  rgpu_fake::count(rgpu_fake::kAlloc, 1);
  *dptr = d;
  return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  bool known;
  void* host = nullptr;
  CUresult r;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_allocs.find(dptr);
    known = it != g_allocs.end();
    CUcontext ctx = nullptr;
    r = enter_locked(&ctx);
    if (r == CUDA_SUCCESS && known) r = capture_permits_locked();
    if (r == CUDA_SUCCESS) {
      if (known) {  // in its own context, whichever is current
        host = it->second.host;
        g_allocs.erase(it);
      } else {
        r = CUDA_ERROR_INVALID_VALUE;
      }
    }
  }
  if (!known) {
    // A stale free is the worst thing the server's cleanup could do - on a
    // real driver the address may belong to somebody else by now - so it is
    // counted where a test can see it rather than just refused.
    rgpu_fake::count(rgpu_fake::kStale, 1);
  }
  if (r != CUDA_SUCCESS) return r;
  std::free(host);
  rgpu_fake::count(rgpu_fake::kAlloc, -1);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t n) {
  if (!src) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_mu);
  void* to = nullptr;
  CUresult r = range_locked(dst, n, &to);
  if (r != CUDA_SUCCESS) return r;
  std::memcpy(to, src, n);
  return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t n) {
  if (!dst) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_mu);
  void* from = nullptr;
  CUresult r = range_locked(src, n, &from);
  if (r != CUDA_SUCCESS) return r;
  std::memcpy(dst, from, n);
  return CUDA_SUCCESS;
}

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t n) {
  rgpu_fake::count(rgpu_fake::kMemset, 1);
  std::lock_guard<std::mutex> lk(g_mu);
  void* to = nullptr;
  CUresult r = range_locked(dst, n, &to);
  if (r != CUDA_SUCCESS) return r;
  std::memset(to, value, n);
  return CUDA_SUCCESS;
}

// The one fire-and-forget call the fake actually completes. Every other
// asynchronous entry point falls through to the generated stub and fails,
// which is fine until a test needs asynchronous work that works: without it
// there is no way to put successful no-reply traffic either side of a failure.
CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t n,
                         CUstream stream) {
  CUresult r = stream_ok(stream);
  if (r != CUDA_SUCCESS) return r;
  return cuMemsetD8_v2(dst, value, n);
}

// --- placement ---------------------------------------------------------------
//
// Which context and device own an allocation. This, not whether a use under
// another context fails, is what a test can rely on to see where memory went:
// the driver infers placement from a pointer, so a copy through a pointer that
// landed on the wrong device still succeeds.

namespace {

// The device a live context is on, or -1. Called with g_mu held.
int device_of_locked(CUcontext ctx) {
  const int dev = primary_device(ctx);
  if (dev >= 0) return dev;
  auto it = g_contexts.find(reinterpret_cast<unsigned long long>(ctx));
  return it == g_contexts.end() ? -1 : it->second;
}

// One attribute of the allocation `a`. Only the attributes the fake models;
// the rest are CUDA_ERROR_NOT_SUPPORTED rather than a made-up answer. Called
// with g_mu held.
CUresult attribute_locked(const Alloc& a, CUpointer_attribute attribute,
                          void* data) {
  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT:
      *static_cast<CUcontext*>(data) = a.ctx;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
      *static_cast<unsigned int*>(data) = CU_MEMORYTYPE_DEVICE;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
      *static_cast<int*>(data) = device_of_locked(a.ctx);
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}

// The "default NULL value" cuPointerGetAttributes gives for a pointer that is
// not a CUDA pointer.
CUresult attribute_default(CUpointer_attribute attribute, void* data) {
  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT:
      *static_cast<CUcontext*>(data) = nullptr;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
      *static_cast<unsigned int*>(data) = 0;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
      *static_cast<int*>(data) = 0;
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}

}  // namespace

// No context needs to be current: the attributes describe the pointer, not
// the caller. A pointer that is not in any live allocation is
// CUDA_ERROR_INVALID_VALUE, as the header documents for one "not allocated
// by, mapped by, or registered with a CUcontext".
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute,
                               CUdeviceptr ptr) {
  if (!data) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = containing_locked(ptr);
  if (it == g_allocs.end()) return CUDA_ERROR_INVALID_VALUE;
  return attribute_locked(it->second, attribute, data);
}

// "Unlike cuPointerGetAttribute, this function will not return an error when
// the ptr encountered is not a valid CUDA pointer. Instead, the attributes are
// assigned default NULL values and CUDA_SUCCESS is returned."
CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                CUpointer_attribute* attributes, void** data,
                                CUdeviceptr ptr) {
  if (numAttributes && (!attributes || !data)) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = containing_locked(ptr);
  for (unsigned int i = 0; i < numAttributes; i++) {
    if (!data[i]) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = it == g_allocs.end()
                     ? attribute_default(attributes[i], data[i])
                     : attribute_locked(it->second, attributes[i], data[i]);
    if (r != CUDA_SUCCESS) return r;
  }
  return CUDA_SUCCESS;
}

// A context's API version, reported about the context it is handed, not the
// current one: a handle that names no context is CUDA_ERROR_INVALID_CONTEXT
// whatever is current. The version itself is any fixed number.
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
  if (!version) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ctx || !bindable_locked(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
  *version = 3011;
  return CUDA_SUCCESS;
}

// Records an event in the context the caller names, which need not be the
// current one. A handle that names no context is refused with
// CUDA_ERROR_INVALID_CONTEXT, which the call documents, whatever is current.
CUresult cuCtxRecordEvent(CUcontext ctx, CUevent event) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ctx || !bindable_locked(ctx)) return CUDA_ERROR_INVALID_CONTEXT;
  if (!g_events.count(reinterpret_cast<unsigned long long>(event))) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}

// Peer access from the current context to another. Both calls report
// CUDA_ERROR_INVALID_CONTEXT "if there is no current context" as well as for a
// bad peer, so their answer can be about either. Which peers have been enabled
// is not kept: nothing here copies between contexts.
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
  std::lock_guard<std::mutex> lk(g_mu);
  CUcontext ctx = nullptr;
  const CUresult r = enter_locked(&ctx);
  if (r != CUDA_SUCCESS) return r;
  if (!peerContext || !bindable_locked(peerContext) || peerContext == ctx) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  if (Flags != 0) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
  std::lock_guard<std::mutex> lk(g_mu);
  CUcontext ctx = nullptr;
  const CUresult r = enter_locked(&ctx);
  if (r != CUDA_SUCCESS) return r;
  if (!peerContext || !bindable_locked(peerContext)) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  return CUDA_SUCCESS;
}

// "Returns in *device the handle of the current context's device." A context
// destroyed under this thread is reported the way every other call here
// reports it; a primary context is on its device whether or not it is
// initialised.
CUresult cuCtxGetDevice(CUdevice* device) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_mu);
  const CUcontext ctx = current();
  if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
  const int dev = device_of_locked(ctx);
  if (dev < 0) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
  *device = dev;
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

// A function is its module's handle with a marker in the top byte, so it lives
// and dies with the module and belongs to the module's context, without a
// table of its own.
namespace {
constexpr unsigned long long kFunctionMark = 0xF0ull << 56;

CUresult function_ok(CUfunction f, CUcontext* owner = nullptr) {
  const auto v = reinterpret_cast<unsigned long long>(f);
  if ((v >> 56) != 0xF0) {
    CUresult r = need_context();
    return r != CUDA_SUCCESS ? r : CUDA_ERROR_INVALID_HANDLE;
  }
  return owned(g_modules, v & ~kFunctionMark, CUDA_ERROR_INVALID_HANDLE,
               owner);
}
}  // namespace

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
  // The client sized this from the image header; a wrong size would have
  // truncated it before it got here.
  unsigned int magic = 0;
  std::memcpy(&magic, image, sizeof(magic));
  CUresult r = need_context();
  if (r != CUDA_SUCCESS) return r;
  if (magic != 0xBA55ED50u) return CUDA_ERROR_INVALID_IMAGE;
  unsigned long long h = 0;
  r = mint(&g_modules, 0x0D0100u, &h);
  if (r != CUDA_SUCCESS) return r;
  *module = reinterpret_cast<CUmodule>(h);
  rgpu_fake::count(rgpu_fake::kModule, 1);
  return CUDA_SUCCESS;
}

// The header says "Unloads a module hmod from the current context", but on
// hardware a module loaded under one context unloads while another is current
// (real-GPU probe, check 4): like a free, it is identified by its handle, not
// the current context. So the module is unloaded wherever it lives, as long as
// some context is current. Its return list names CUDA_ERROR_INVALID_VALUE and
// not CUDA_ERROR_INVALID_HANDLE for a handle that names no module.
CUresult cuModuleUnload(CUmodule m) {
  CUresult r = retire(&g_modules, reinterpret_cast<unsigned long long>(m),
                      CUDA_ERROR_INVALID_VALUE);
  if (r != CUDA_SUCCESS) return r;
  rgpu_fake::count(rgpu_fake::kModule, -1);
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                             const char* name) {
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  const auto h = reinterpret_cast<unsigned long long>(hmod);
  CUresult r = owned(g_modules, h, CUDA_ERROR_INVALID_HANDLE);
  if (r != CUDA_SUCCESS) return r;
  if (std::strcmp(name, kCheckedKernel) != 0) return CUDA_ERROR_NOT_FOUND;
  *hfunc = reinterpret_cast<CUfunction>(h | kFunctionMark);
  return CUDA_SUCCESS;
}

// Three pointers then an int, which is the layout of the checked kernel. Past
// the last parameter is CUDA_ERROR_INVALID_VALUE, which the server reads as
// the end of the list, so a function that is not valid here must not say
// that.
CUresult cuFuncGetParamInfo(CUfunction f, size_t index, size_t* offset,
                            size_t* size) {
  if (!f || !offset || !size) return CUDA_ERROR_INVALID_VALUE;
  CUresult r = function_ok(f);
  if (r != CUDA_SUCCESS) return r;
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
  (void)gy; (void)gz; (void)by; (void)bz; (void)shmem;
  CUresult r = need_context();
  if (r != CUDA_SUCCESS) return r;
  if (!f) return CUDA_ERROR_INVALID_HANDLE;
  CUcontext fctx = nullptr, sctx = nullptr;
  r = function_ok(f, &fctx);
  if (r != CUDA_SUCCESS) return r;
  r = stream_ok(stream, &sctx);
  if (r != CUDA_SUCCESS) return r;
  // The launch runs in the stream's context - the current one for a null
  // stream, which is inferred - and the function has to be from that context
  // (see the note on what lives in a context, above).
  if (fctx != sctx) return CUDA_ERROR_INVALID_HANDLE;
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
  unsigned long long h = 0;
  CUresult r = mint(&g_streams, 0x57EAu, &h);
  if (r != CUDA_SUCCESS) return r;
  *stream = reinterpret_cast<CUstream>(h);
  rgpu_fake::count(rgpu_fake::kStream, 1);
  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream stream) {
  CUresult r = retire(&g_streams, reinterpret_cast<unsigned long long>(stream),
                      CUDA_ERROR_INVALID_HANDLE);
  if (r != CUDA_SUCCESS) return r;
  rgpu_fake::count(rgpu_fake::kStream, -1);
  // A capture open on it goes with it. What a real driver does here is not
  // documented; ending it quietly is the reading that loses nothing.
  bool had_capture;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    had_capture = g_captures.erase(capture_key(stream, nullptr)) > 0;
  }
  if (had_capture) rgpu_fake::count(rgpu_fake::kCapture, -1);
  return CUDA_SUCCESS;
}

CUresult cuEventCreate(CUevent* event, unsigned int) {
  if (!event) return CUDA_ERROR_INVALID_VALUE;
  unsigned long long h = 0;
  CUresult r = mint(&g_events, 0xE7E17u, &h);
  if (r != CUDA_SUCCESS) return r;
  *event = reinterpret_cast<CUevent>(h);
  rgpu_fake::count(rgpu_fake::kEvent, 1);
  return CUDA_SUCCESS;
}

CUresult cuEventDestroy_v2(CUevent event) {
  CUresult r = retire(&g_events, reinterpret_cast<unsigned long long>(event),
                      CUDA_ERROR_INVALID_HANDLE);
  if (r != CUDA_SUCCESS) return r;
  rgpu_fake::count(rgpu_fake::kEvent, -1);
  return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream stream) { return stream_ok(stream); }

// --- stream capture --------------------------------------------------------
//
// Enough to tell whether the calls arrive intact, and what a capture left open
// restricts (see g_captures). The dependency array is the part worth checking:
// the driver owns it, so it cannot cross the wire as a pointer, and the client
// has to be handed a copy of the contents instead.

namespace {
CUgraphNode g_nodes[2] = {reinterpret_cast<CUgraphNode>(0xDEB1),
                          reinterpret_cast<CUgraphNode>(0xDEB2)};

// Whether `stream` has a capture open. Called with g_mu held; `r` is the
// stream's own check.
bool capturing_locked(CUstream stream, CUresult* r) {
  CUcontext ctx = nullptr;
  *r = stream_ok_locked(stream, &ctx);
  return *r == CUDA_SUCCESS && g_captures.count(capture_key(stream, ctx)) > 0;
}
}  // namespace

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) {
  rgpu_fake::count(rgpu_fake::kCaptureModeExchange, 1);
  if (!mode) return CUDA_ERROR_INVALID_VALUE;
  if (*mode != CU_STREAM_CAPTURE_MODE_GLOBAL &&
      *mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
      *mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const CUstreamCaptureMode previous = t_capture_mode;
  t_capture_mode = *mode;
  *mode = previous;
  return CUDA_SUCCESS;
}

// "it may only be initiated if the stream is not already in capture mode"; the
// code for a second one is the only one the return list offers.
CUresult cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode) {
  if (mode != CU_STREAM_CAPTURE_MODE_GLOBAL &&
      mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
      mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  {
    std::lock_guard<std::mutex> lk(g_mu);
    CUcontext ctx = nullptr;
    CUresult r = stream_ok_locked(stream, &ctx);
    if (r != CUDA_SUCCESS) return r;
    const CaptureKey key = capture_key(stream, ctx);
    if (g_captures.count(key)) return CUDA_ERROR_INVALID_VALUE;
    g_captures[key] = Capture{mode, std::this_thread::get_id()};
  }
  rgpu_fake::count(rgpu_fake::kCapture, 1);
  return CUDA_SUCCESS;
}

// The graph belongs to no context (see the note on what lives in a context):
// it outlives the capture context and every other.
CUresult cuStreamEndCapture(CUstream stream, CUgraph* graph) {
  unsigned long long h = 0;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    CUcontext ctx = nullptr;
    CUresult r = stream_ok_locked(stream, &ctx);
    if (r != CUDA_SUCCESS) return r;
    auto it = g_captures.find(capture_key(stream, ctx));
    if (it == g_captures.end()) return CUDA_ERROR_ILLEGAL_STATE;
    if (it->second.mode != CU_STREAM_CAPTURE_MODE_RELAXED &&
        it->second.owner != std::this_thread::get_id()) {
      return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    }
    g_captures.erase(it);
    if (graph) h = mint_in_locked(&g_graphs, 0xC0FFEEu, /*ctx=*/nullptr);
  }
  rgpu_fake::count(rgpu_fake::kCapture, -1);
  if (!graph) return CUDA_ERROR_INVALID_VALUE;
  *graph = reinterpret_cast<CUgraph>(h);
  rgpu_fake::count(rgpu_fake::kGraph, 1);
  return CUDA_SUCCESS;
}

CUresult cuStreamIsCapturing(CUstream stream, CUstreamCaptureStatus* status) {
  std::lock_guard<std::mutex> lk(g_mu);
  CUresult r = CUDA_SUCCESS;
  const bool capturing = capturing_locked(stream, &r);
  if (r != CUDA_SUCCESS) return r;
  if (status) {
    *status = capturing ? CU_STREAM_CAPTURE_STATUS_ACTIVE
                        : CU_STREAM_CAPTURE_STATUS_NONE;
  }
  return CUDA_SUCCESS;
}

CUresult cuStreamGetCaptureInfo_v2(CUstream stream,
                                   CUstreamCaptureStatus* status,
                                   cuuint64_t* id, CUgraph* graph,
                                   const CUgraphNode** deps,
                                   size_t* ndeps) {
  std::lock_guard<std::mutex> lk(g_mu);
  CUresult r = CUDA_SUCCESS;
  const bool capturing = capturing_locked(stream, &r);
  if (r != CUDA_SUCCESS) return r;
  if (status) {
    *status = capturing ? CU_STREAM_CAPTURE_STATUS_ACTIVE
                        : CU_STREAM_CAPTURE_STATUS_NONE;
  }
  if (id) *id = capturing ? 0x1D : 0;
  if (graph) *graph = capturing ? reinterpret_cast<CUgraph>(0xC0FFEEull)
                                : nullptr;
  if (deps) *deps = g_nodes;
  if (ndeps) *ndeps = capturing ? 2 : 0;
  return CUDA_SUCCESS;
}

// An executable, and a clone, belong to no context (see the note on what
// lives in a context): each outlives the graph's context and every other.
CUresult cuGraphInstantiateWithFlags(CUgraphExec* exec, CUgraph graph,
                                     unsigned long long) {
  unsigned long long h = 0;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    CUresult r = owned_locked(g_graphs, reinterpret_cast<unsigned long long>(graph),
                              CUDA_ERROR_INVALID_VALUE);
    if (r != CUDA_SUCCESS) return r;
    if (!exec) return CUDA_ERROR_INVALID_VALUE;
    h = mint_in_locked(&g_graph_execs, 0xE7E0u, /*ctx=*/nullptr);
  }
  *exec = reinterpret_cast<CUgraphExec>(h);
  rgpu_fake::count(rgpu_fake::kGraphExec, 1);
  return CUDA_SUCCESS;
}

CUresult cuGraphLaunch(CUgraphExec exec, CUstream stream) {
  CUresult r = owned(g_graph_execs, reinterpret_cast<unsigned long long>(exec),
                     CUDA_ERROR_INVALID_VALUE);
  if (r != CUDA_SUCCESS) return r;
  return stream_ok(stream);
}

CUresult cuGraphClone(CUgraph* clone, CUgraph original) {
  if (!clone) return CUDA_ERROR_INVALID_VALUE;
  unsigned long long h = 0;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    CUresult r = owned_locked(g_graphs,
                              reinterpret_cast<unsigned long long>(original),
                              CUDA_ERROR_INVALID_VALUE);
    if (r != CUDA_SUCCESS) return r;
    h = mint_in_locked(&g_graphs, 0xC0FFEEu, /*ctx=*/nullptr);
  }
  *clone = reinterpret_cast<CUgraph>(h);
  rgpu_fake::count(rgpu_fake::kGraph, 1);
  return CUDA_SUCCESS;
}

CUresult cuGraphDestroy(CUgraph graph) {
  CUresult r = retire(&g_graphs, reinterpret_cast<unsigned long long>(graph),
                      CUDA_ERROR_INVALID_VALUE);
  if (r != CUDA_SUCCESS) return r;
  rgpu_fake::count(rgpu_fake::kGraph, -1);
  return CUDA_SUCCESS;
}

CUresult cuGraphExecDestroy(CUgraphExec exec) {
  CUresult r = retire(&g_graph_execs, reinterpret_cast<unsigned long long>(exec),
                      CUDA_ERROR_INVALID_VALUE);
  if (r != CUDA_SUCCESS) return r;
  rgpu_fake::count(rgpu_fake::kGraphExec, -1);
  return CUDA_SUCCESS;
}

}  // extern "C"
