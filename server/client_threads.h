// What each client thread was looking at: its CUDA context stack.
//
// CUDA's current context, and the stack of contexts it is the top of, belong
// to the calling thread, and one server thread serves every thread of a client
// process. So each client thread's stack is kept here, and the server thread
// is made to show that thread's top before its request runs. See
// docs/superpowers/specs/2026-09-12-per-thread-cuda-contexts.md.
//
// The stack kept here is the whole truth about a client thread's contexts. The
// serving thread's own driver stack is only ever one deep - just the top of
// whichever client thread was served last, or nothing - so moving from one
// client thread to another is one cuCtxSetCurrent, and making "no context"
// current is one pop that cannot uncover anybody's context underneath. The
// calls that would make the driver's stack deeper or shallower - selecting,
// pushing, popping, creating and destroying a context - are intercepted and
// carried out on the issuing thread's stack here instead, with the driver then
// shown the new top.
//
// Nothing here is owned. Every context named is accounted for by the session's
// inventory, which is what releases it; forgetting a slot is forgetting a
// value and nothing more.
#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>

namespace rgpu {

// One entry of a client thread's context stack.
//
// A CUcontext is an address in driver memory, and a created context's address
// may be handed out again once it is destroyed - to a cuCtxCreate anywhere in
// the process, another tenant's included. The defence against binding a thread
// to a reused address is the sweep, not a refusal from the driver: every
// destroy this session sees is swept through every stack
// (client_threads_destroyed), which marks the entry gone before its address can
// be reused, and a gone entry is never made current. It has to be the sweep,
// because on hardware cuCtxSetCurrent accepts a destroyed handle and leaves it
// current (real-GPU probe, check 9) - a destroy could not be noticed at restore
// time. The one destroy the sweep cannot see is another session's destroy of a
// context shared across sessions: the documented one-server-per-tenant limit,
// which nothing here defends against.
//
// A primary context is different. Its last release "automatically reset[s]"
// it and a reset "does not release it": either way it is emptied, not
// replaced, and its handle survives. So a thread that had it current keeps it
// current through another session's last release or a reset, as in CUDA, and
// nothing is swept for them.
struct SavedContext {
  CUcontext ctx = nullptr;
  // Destroyed since it was saved. Never made current again. It stays in the
  // stack, as a destroyed context stays current to the threads it was current
  // to in CUDA, so that popping what was pushed on top of it finds it again.
  // While it is the top, the thread has no context on the serving thread, its
  // calls that need one fail with CUDA_ERROR_CONTEXT_IS_DESTROYED, and
  // cuCtxGetCurrent still names it.
  bool gone = false;
};

struct ClientThread {
  std::vector<SavedContext> stack;  // bottom first; back() is current
  // A call sent without expecting a reply has nowhere to report a failure, so
  // the first one is held and handed to the next call from this same thread
  // that does reply. In CUDA the failing call would have returned its error to
  // the thread that made it; handing it to whichever thread replied next told
  // a thread of a failure it did not cause, and the thread that caused it
  // never learned of it.
  //
  // It lives with the session, not the connection. A dropped connection is not
  // an acknowledgement: the client was told the call completed, so it will not
  // send it again, and an error left behind on the old connection would turn a
  // failed launch into an apparent success.
  //
  // Written and taken only in the same critical section, under the session's
  // mutex, that records the request as completed (server/main.cpp), so that
  // completing a request and holding or reporting its failure are one step.
  // The only other reader is the log line for a failure its thread never
  // learned of (say_unreported), on the serving thread, when the slot is
  // forgotten.
  CUresult pending_async = CUDA_SUCCESS;
  // The stream capture mode, which CUDA keeps per thread too ("A thread's mode
  // is one of the following"), starting at what the header calls the default.
  // Only cuThreadExchangeStreamCaptureMode changes it, and it is intercepted
  // (client_threads_exchange_capture_mode), so unlike the context nothing has
  // to be read back.
  CUstreamCaptureMode capture_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
};

// A session's client threads, and what the serving thread has current.
//
// Touched only by the thread serving the session, which is also the thread
// that clears it at expiry, so it takes no lock: nothing else - not the
// handshake, not another session - reads or writes it. (A slot's
// pending_async is written under the session's mutex, but for the step it
// shares with completing a request, not to protect it from anybody.)
struct ClientThreads {
  std::unordered_map<uint32_t, ClientThread> live;
  // Oldest first, at most kRetiredSlots (server/main.cpp).
  std::deque<std::pair<uint32_t, ClientThread>> retired;
  // What the serving thread has current - its driver stack is never deeper
  // than this one entry - or nothing. It lives with the session rather than
  // with a connection, because the thread, and so the driver's idea of its
  // current context, outlasts every connection the session has.
  SavedContext applied;
  // The serving thread's own capture mode. A new thread starts at the
  // default, as a new client thread does, and so does this one.
  CUstreamCaptureMode applied_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  // The thread whose request is running, for the intercepted calls. Null
  // outside a request.
  ClientThread* caller = nullptr;
  uint32_t first_id = 0;  // the first thread seen, for the log
  bool said_many = false;
  bool said_full = false;
  bool said_zero = false;
  bool said_stuck = false;
  bool said_deep = false;
  bool said_mode = false;
  // Failures of calls sent without a reply, for the log: how many were held
  // for their thread, and how many of those never reached it because the
  // thread's slot was forgotten first. A client can make both as fast as it
  // can send frames, so the server says the first of each and gives the counts
  // when the session expires (server/main.cpp), unless RGPU_VERBOSE is set.
  uint64_t held_failures = 0;
  uint64_t unreported_failures = 0;

  // Forgets; releases nothing, and changes nothing in the driver. The
  // serving thread keeps whatever capture mode it last had, which is still
  // recorded in applied_mode, until expiry puts it into RELAXED
  // (client_threads_relax_capture_mode).
  void clear() {
    live.clear();
    retired.clear();
    applied = SavedContext{};
    caller = nullptr;
  }
};

// Binds the serving thread to its session's table, the way inventory_bind
// binds the inventory, so the intercepted calls find the issuing thread's
// stack. nullptr unbinds.
void client_threads_bind(ClientThreads* threads);

// Whether the serving thread already shows `t`'s thread state - its current
// context and its capture mode - so nothing need happen before its request.
// Inline, because it is every request. The two are checked apart: threads
// sharing a context still each have a capture mode of their own.
inline bool client_thread_shown(const ClientThread& t,
                                const ClientThreads& threads) {
  if (t.capture_mode != threads.applied_mode) return false;
  if (t.stack.empty() || t.stack.back().gone) {
    return threads.applied.ctx == nullptr;
  }
  return t.stack.back().ctx == threads.applied.ctx;
}

// Makes `t`'s current context current on the serving thread, or none if its
// top is destroyed, and gives the serving thread `t`'s capture mode.
// CUDA_SUCCESS, or the error to refuse the request with if the serving thread
// could not be left without a context or given the mode.
CUresult client_thread_show(ClientThreads& threads, ClientThread& t);

// Puts the serving thread into RELAXED stream capture mode, whatever the last
// client thread served left it in, and records that in applied_mode. For
// expiry, before the session's inventory is released: RELAXED is immune to the
// capture restrictions that another session's live GLOBAL capture would
// otherwise impose on this thread's frees, and just as deterministic. Logged if
// the driver refuses.
void client_threads_relax_capture_mode(ClientThreads& threads);

// cuThreadExchangeStreamCaptureMode for the thread whose request is running:
// the driver's exchange on the serving thread, which already has that
// thread's mode, recorded as the thread's new mode.
CUresult client_threads_exchange_capture_mode(CUstreamCaptureMode* mode);

// After a request: what the driver says is current, recorded as `t`'s top if
// something the interception does not cover changed it, and the request's
// result corrected to CUDA_ERROR_CONTEXT_IS_DESTROYED if it failed for want of
// a context while `t`'s current context is a destroyed one. `api_id` is the
// request's call.
void client_thread_after(ClientThreads& threads, ClientThread& t,
                         uint32_t api_id, CUresult* result);

// The interception of cuCtxSetCurrent, cuCtxPushCurrent_v2,
// cuCtxPopCurrent_v2 and cuCtxGetCurrent, or nullptr for any other name. See
// driver_wrapper().
void* client_threads_wrapper(const char* name);

// Told by the inventory's wrappers, after the driver call succeeded: a context
// was created, and pushed on the calling thread as CUDA does; a context was
// destroyed, by cuCtxDestroy or cuCtxDetach.
void client_threads_created(CUcontext ctx);
// Asked by the cuCtxCreate wrapper before the driver call: whether the calling
// thread's stack has room for the context a create pushes. A thread's stack is
// capped at RGPU_MAX_CONTEXT_STACK entries (64 by default); a push or create
// past it is refused with CUDA_ERROR_INVALID_VALUE.
bool client_threads_may_create();
void client_threads_destroyed(CUcontext ctx);

}  // namespace rgpu
