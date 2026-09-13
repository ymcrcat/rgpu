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
// the process, another tenant's included. So a saved created context is only
// ever made current again if nothing has destroyed it since it was saved:
// every destroy is swept through every stack (client_threads_destroyed).
//
// A primary context is different. Its last release "automatically reset[s]"
// it and a reset "does not release it": either way it is emptied, not
// replaced, and its handle survives. So a thread that had it current keeps it
// current through another session's last release or a reset, as in CUDA, and
// nothing is swept for them. If a real driver ever handed out a different
// handle for a device's primary context after that, making the old one current
// again would fail, and the entry would take the same fail-clean path as a
// created context destroyed where no sweep saw it.
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
  // Nothing else reads it.
  CUresult pending_async = CUDA_SUCCESS;
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
  // The thread whose request is running, for the intercepted calls. Null
  // outside a request.
  ClientThread* caller = nullptr;
  uint32_t first_id = 0;  // the first thread seen, for the log
  bool said_many = false;
  bool said_full = false;
  bool said_zero = false;
  bool said_stuck = false;
  bool said_deep = false;

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

// Whether the serving thread already shows `t`'s current context, so nothing
// need happen before its request. Inline, because it is every request.
inline bool client_thread_shown(const ClientThread& t,
                                const SavedContext& applied) {
  if (t.stack.empty() || t.stack.back().gone) return applied.ctx == nullptr;
  const SavedContext& top = t.stack.back();
  return top.ctx == applied.ctx;
}

// Makes `t`'s current context current on the serving thread, or none if its
// top is destroyed. CUDA_SUCCESS, or the error to refuse the request with if
// the serving thread could not be left without a context.
CUresult client_thread_show(ClientThreads& threads, ClientThread& t);

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
// capped at RGPU_MAX_CONTEXT_STACK entries (1024 by default); a push or create
// past it is refused with CUDA_ERROR_INVALID_VALUE.
bool client_threads_may_create();
void client_threads_destroyed(CUcontext ctx);

}  // namespace rgpu
