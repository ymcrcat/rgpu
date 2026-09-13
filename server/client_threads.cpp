#include "server/client_threads.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/cublas_ids.h"
#include "common/generated/api_ids.h"
#include "common/internal_ids.h"

namespace rgpu {
namespace {

thread_local ClientThreads* t_threads = nullptr;

bool verbose() {
  static const bool on = std::getenv("RGPU_VERBOSE") != nullptr;
  return on;
}

void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[rgpu-server] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

SavedContext saved(CUcontext ctx) {
  SavedContext s;
  s.ctx = ctx;
  return s;
}

// What the serving thread has current, as the driver says. After anything
// that did not go as asked, so `applied` is never a belief the driver
// contradicts.
void read_back(ClientThreads& threads) {
  CUcontext now = nullptr;
  cuCtxGetCurrent(&now);
  if (now != threads.applied.ctx) threads.applied = now ? saved(now) : SavedContext{};
}

// Leaves the serving thread with no context. Its driver stack holds one entry
// at most, so this is one pop. More than one is tolerated but not relied on -
// a call the interception does not know may have pushed - and every context on
// it is kept in some client thread's stack here, so popping it takes nothing
// from anybody. Bounded: a driver that will not let go gets the request
// refused, not an endless loop on every request.
CUresult clear(ClientThreads& threads) {
  constexpr int kMaxDepth = 8;
  CUresult r = CUDA_SUCCESS;
  for (int i = 0; threads.applied.ctx && i < kMaxDepth; i++) {
    r = cuCtxSetCurrent(nullptr);
    CUcontext now = nullptr;
    cuCtxGetCurrent(&now);
    threads.applied = now ? saved(now) : SavedContext{};
    if (r != CUDA_SUCCESS) break;
  }
  if (!threads.applied.ctx) return CUDA_SUCCESS;
  if (!threads.said_stuck) {
    threads.said_stuck = true;
    logf("could not leave the serving thread without a context (%d): "
         "context %p is still current, so a request from a client thread "
         "with no context is refused rather than run under it",
         r, (void*)threads.applied.ctx);
  }
  return r != CUDA_SUCCESS ? r : CUDA_ERROR_INVALID_CONTEXT;
}

// How deep a client thread's context stack may grow. It is server memory, the
// driver's own stack on the serving thread stays one deep and so never limits
// it, and every destroy sweeps every entry, so a client pushing in a loop
// must be stopped here. CUDA programs push a handful deep at most. The default
// is kept small because the cost multiplies: a destroy sweeps every entry of
// every live and retired thread, so at the thread cap (4096 live, 64 retired)
// a deep default would make each destroy a walk of millions of entries, and
// the stacks alone megabytes.
size_t max_stack_depth() {
  static const size_t n = [] {
    const char* v = std::getenv("RGPU_MAX_CONTEXT_STACK");
    const long parsed = v ? std::atol(v) : 0;
    return parsed > 0 ? static_cast<size_t>(parsed) : size_t{64};
  }();
  return n;
}

// Whether `t` may have one more context pushed. Said once per session when it
// may not.
bool may_push(ClientThreads& threads, const ClientThread& t) {
  if (t.stack.size() < max_stack_depth()) return true;
  if (!threads.said_deep) {
    threads.said_deep = true;
    logf("refusing to push a context onto a client thread's stack %zu deep, "
         "the most allowed (RGPU_MAX_CONTEXT_STACK)",
         t.stack.size());
  }
  return false;
}

// Every entry naming a context that is gone, in every thread this session
// knows, live or recently retired.
template <typename Match>
void sweep(ClientThreads& threads, Match match) {
  for (auto& entry : threads.live) {
    for (SavedContext& s : entry.second.stack) {
      if (!s.gone && match(s)) s.gone = true;
    }
  }
  for (auto& entry : threads.retired) {
    for (SavedContext& s : entry.second.stack) {
      if (!s.gone && match(s)) s.gone = true;
    }
  }
}

// --- the intercepted calls ----------------------------------------------------
//
// Each acts on the issuing thread's stack and then shows the driver its top,
// keeping the serving thread's own stack one deep. Without a bound session and
// a request running - which never happens in the server - they are the
// driver's own calls.

// Replaces the top of the thread's stack, or pushes onto an empty one; null
// pops it. As documented.
CUresult w_cuCtxSetCurrent(CUcontext ctx) {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  if (!t) return cuCtxSetCurrent(ctx);
  if (!ctx) {
    if (t->stack.empty()) return CUDA_SUCCESS;  // documented as a no-op
    t->stack.pop_back();
    return client_thread_show(*threads, *t);
  }
  const SavedContext s = saved(ctx);
  // The driver's top is this thread's, so replacing it is right whether the
  // thread's stack is deeper or empty.
  CUresult r = cuCtxSetCurrent(ctx);
  if (r != CUDA_SUCCESS) {
    read_back(*threads);
    return r;
  }
  if (t->stack.empty()) {
    t->stack.push_back(s);
  } else {
    t->stack.back() = s;
  }
  threads->applied = s;
  return CUDA_SUCCESS;
}

// Pushed onto the thread's stack here; the driver has its top replaced, so its
// own stack stays one deep.
CUresult w_cuCtxPushCurrent_v2(CUcontext ctx) {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  if (!t) return cuCtxPushCurrent_v2(ctx);
  // Not a context. Setting null would pop instead, so it is refused here with
  // the code the call documents for a context it cannot push.
  if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
  // Past the cap: one of the codes the call documents, and nothing changes.
  if (!may_push(*threads, *t)) return CUDA_ERROR_INVALID_VALUE;
  const SavedContext s = saved(ctx);
  CUresult r = cuCtxSetCurrent(ctx);
  if (r != CUDA_SUCCESS) {
    read_back(*threads);
    return r;
  }
  t->stack.push_back(s);
  threads->applied = s;
  return CUDA_SUCCESS;
}

// Pops the thread's stack and shows the driver what is underneath. The popped
// handle is returned even if that context has been destroyed, as CUDA returns
// it.
CUresult w_cuCtxPopCurrent_v2(CUcontext* pctx) {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  if (!t) return cuCtxPopCurrent_v2(pctx);
  // An empty stack here means an empty one in the driver, so this is the
  // driver's own answer for popping nothing.
  if (t->stack.empty()) {
    CUresult r = cuCtxPopCurrent_v2(pctx);
    read_back(*threads);
    return r;
  }
  const CUcontext popped = t->stack.back().ctx;
  t->stack.pop_back();
  CUresult r = client_thread_show(*threads, *t);
  if (pctx) *pctx = popped;
  return r;
}

// The thread's current context is the top of its stack, so that is what is
// answered - including a destroyed one, which CUDA leaves current and names,
// though the serving thread has nothing current for it. The driver is still
// asked, for its errors.
CUresult w_cuCtxGetCurrent(CUcontext* pctx) {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  if (!t || !pctx) return cuCtxGetCurrent(pctx);
  CUcontext driver = nullptr;
  CUresult r = cuCtxGetCurrent(&driver);
  if (r != CUDA_SUCCESS) return r;
  *pctx = t->stack.empty() ? nullptr : t->stack.back().ctx;
  return CUDA_SUCCESS;
}

struct Wrapper {
  const char* name;
  void* fn;
};

const Wrapper kWrappers[] = {
    {"cuCtxSetCurrent", reinterpret_cast<void*>(&w_cuCtxSetCurrent)},
    {"cuCtxPushCurrent_v2", reinterpret_cast<void*>(&w_cuCtxPushCurrent_v2)},
    {"cuCtxPopCurrent_v2", reinterpret_cast<void*>(&w_cuCtxPopCurrent_v2)},
    {"cuCtxGetCurrent", reinterpret_cast<void*>(&w_cuCtxGetCurrent)},
};

}  // namespace

void client_threads_bind(ClientThreads* threads) { t_threads = threads; }

namespace {

// Makes `t`'s current context current on the serving thread, or leaves it
// with none.
CUresult show_context(ClientThreads& threads, ClientThread& t) {
  SavedContext* top = t.stack.empty() ? nullptr : &t.stack.back();
  if (top && !top->gone) {
    if (top->ctx == threads.applied.ctx) {
      return CUDA_SUCCESS;
    }
    if (cuCtxSetCurrent(top->ctx) == CUDA_SUCCESS) {
      threads.applied = *top;
      return CUDA_SUCCESS;
    }
    // Destroyed where no sweep saw it. The same as any destroyed context.
    top->gone = true;
    if (verbose()) {
      logf("context %p could not be made current again; it was destroyed "
           "since its client thread selected it",
           (void*)top->ctx);
    }
    read_back(threads);
  }
  // Nothing current, or a destroyed context: the thread has no context, and a
  // call that needs one fails on its own with nothing current to misuse.
  return clear(threads);
}

// Gives the serving thread `t`'s capture mode. One exchange, and only when it
// differs, so threads that keep to one mode pay nothing.
CUresult show_capture_mode(ClientThreads& threads, const ClientThread& t) {
  if (t.capture_mode == threads.applied_mode) return CUDA_SUCCESS;
  CUstreamCaptureMode mode = t.capture_mode;
  const CUresult r = cuThreadExchangeStreamCaptureMode(&mode);
  if (r == CUDA_SUCCESS) {
    threads.applied_mode = t.capture_mode;
    return CUDA_SUCCESS;
  }
  // Refused rather than run under another thread's mode, which would not fail
  // anything obvious: it would invalidate a capture, or let a call through
  // that the thread's own mode forbids. An exchange that fails leaves the
  // mode as it was, so the next request tries again.
  if (!threads.said_mode) {
    threads.said_mode = true;
    logf("could not give the serving thread a client thread's stream capture "
         "mode (%d), so its request is refused rather than run under another "
         "thread's mode",
         r);
  }
  return r;
}

}  // namespace

CUresult client_thread_show(ClientThreads& threads, ClientThread& t) {
  CUresult r = show_context(threads, t);
  if (r != CUDA_SUCCESS) return r;
  return show_capture_mode(threads, t);
}

void client_threads_restore_default_mode(ClientThreads& threads) {
  if (threads.applied_mode == CU_STREAM_CAPTURE_MODE_GLOBAL) return;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  const CUresult r = cuThreadExchangeStreamCaptureMode(&mode);
  if (r == CUDA_SUCCESS) {
    threads.applied_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
    return;
  }
  logf("session cleanup: could not put the serving thread back in the default "
       "stream capture mode (%d); releasing in the mode a client left it in",
       r);
}

CUresult client_threads_exchange_capture_mode(CUstreamCaptureMode* mode) {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  if (!t || !mode) return cuThreadExchangeStreamCaptureMode(mode);
  const CUstreamCaptureMode wanted = *mode;
  const CUresult r = cuThreadExchangeStreamCaptureMode(mode);
  if (r == CUDA_SUCCESS) {
    t->capture_mode = wanted;
    threads->applied_mode = wanted;
  }
  return r;
}

void client_thread_after(ClientThreads& threads, ClientThread& t,
                         uint32_t api_id, CUresult* result) {
  // What the call left current, as the driver says. The interception keeps
  // this equal to `applied`, so normally nothing more happens; a call it does
  // not cover that changed the context anyway is taken at its word, as a
  // replaced top, or a popped one if nothing is current now.
  CUcontext now = nullptr;
  cuCtxGetCurrent(&now);
  if (now != threads.applied.ctx) {
    if (!now) {
      if (!t.stack.empty()) t.stack.pop_back();
      threads.applied = SavedContext{};
    } else {
      const SavedContext s = saved(now);
      if (t.stack.empty()) {
        t.stack.push_back(s);
      } else {
        t.stack.back() = s;
      }
      threads.applied = s;
    }
  }

  // CUDA fails a call on a thread whose current context was destroyed with
  // CUDA_ERROR_CONTEXT_IS_DESTROYED, and the context stays current, so it
  // goes on doing so until the thread selects another. Here nothing is current
  // instead, so the driver says CUDA_ERROR_INVALID_CONTEXT; while the thread's
  // top is still the destroyed context, that is corrected.
  //
  // Not corrected: the calls listed below, which report
  // CUDA_ERROR_INVALID_CONTEXT about a context they were handed as an
  // argument rather than the current one, and keep their own answer; and the
  // maths libraries, whose calls carry no driver result here - only a driver
  // call's result is a CUresult to correct.
  //
  // Not listed, because their answer may be about the current context:
  // cuCtxEnablePeerAccess and cuCtxDisablePeerAccess, which the header says
  // return it "if there is no current context" as well as for a bad peer; and
  // the calls that name a context inside a structure - cuMemcpy3DPeer, the
  // kernel and generic node parameters - where a null context may mean the
  // current one, as the header says of CUDA_KERNEL_NODE_PARAMS.
  if (*result != CUDA_ERROR_INVALID_CONTEXT) return;
  if (t.stack.empty() || !t.stack.back().gone) return;
  const bool driver_call =
      api_id < API__count || (api_id >= kInternalBase && api_id < kCublasBase);
  if (!driver_call) return;
  switch (api_id) {
    case API_cuCtxSetCurrent:
    case API_cuCtxPushCurrent_v2:
    case API_cuCtxPopCurrent_v2:
    case API_cuCtxDestroy_v2:
    case API_cuCtxDetach:
    case API_cuCtxGetApiVersion:
    case API_cuCtxGetId:
    case API_cuCtxRecordEvent:
    case API_cuCtxWaitEvent:
    case API_cuCtxGetDevResource:
    case API_cuMemcpyPeer:
    case API_cuMemcpyPeerAsync:
    case API_cuGraphAddMemcpyNode:
    case API_cuGraphAddMemsetNode:
    case API_cuGraphConditionalHandleCreate:
    case API_cuGraphExecMemcpyNodeSetParams:
    case API_cuGraphExecMemsetNodeSetParams:
    case API_cuDevicePrimaryCtxRetain:
    case API_cuDevicePrimaryCtxRelease_v2:
      return;
    default:
      *result = CUDA_ERROR_CONTEXT_IS_DESTROYED;
  }
}

void* client_threads_wrapper(const char* name) {
  for (const Wrapper& w : kWrappers) {
    if (std::strcmp(w.name, name) == 0) return w.fn;
  }
  return nullptr;
}

bool client_threads_may_create() {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  return !t || may_push(*threads, *t);
}

// The driver pushed the new context onto the serving thread's stack; it goes
// onto the issuing thread's here, and the driver's is brought back to one
// entry.
void client_threads_created(CUcontext ctx) {
  ClientThreads* threads = t_threads;
  ClientThread* t = threads ? threads->caller : nullptr;
  if (!t || !ctx) return;
  if (threads->applied.ctx) {
    // Pops the new context, then replaces what was underneath with it.
    cuCtxSetCurrent(nullptr);
    cuCtxSetCurrent(ctx);
  }
  const SavedContext s = saved(ctx);
  t->stack.push_back(s);
  threads->applied = s;
  read_back(*threads);
}

// "If ctx is current to the calling thread then ctx will also be popped from
// the current thread's context stack (as though cuCtxPopCurrent() were
// called). If ctx is current to other threads, then ctx will remain current to
// those threads". So the issuing thread pops it, and every other entry naming
// it - in this thread's stack below the top, and in every other thread's - is
// marked destroyed, so that no later context given the same address is ever
// made current in its place.
void client_threads_destroyed(CUcontext ctx) {
  ClientThreads* threads = t_threads;
  if (!threads || !ctx) return;
  ClientThread* t = threads->caller;
  if (t && !t->stack.empty() && !t->stack.back().gone &&
      t->stack.back().ctx == ctx) {
    t->stack.pop_back();
  }
  sweep(*threads, [&](const SavedContext& s) { return s.ctx == ctx; });
  if (threads->applied.ctx == ctx) {
    read_back(*threads);
    // The header says it was popped; if the driver kept it current anyway,
    // it is popped here, so nothing destroyed stays current.
    if (threads->applied.ctx == ctx) clear(*threads);
  }
  if (t) client_thread_show(*threads, *t);
}

}  // namespace rgpu
