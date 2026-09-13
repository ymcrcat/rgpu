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

// One entry of a client thread's context stack: a context, and - if it is a
// primary context - the device and generation it was saved under (see
// inventory.h). A CUcontext is an address in driver memory that may be handed
// out again once the context is destroyed, so a saved handle is only ever
// made current again if nothing has destroyed the context since it was saved.
struct SavedContext {
  CUcontext ctx = nullptr;
  int dev = -1;
  uint64_t gen = 0;
  // Destroyed since it was saved, by this session or - for a primary context
  // - by any. Never made current again. It stays in the stack, as a destroyed
  // context stays current to the threads it was current to in CUDA, so that
  // popping what was pushed on top of it finds it again. While it is the top,
  // the thread has no context and its calls that need one fail with
  // CUDA_ERROR_CONTEXT_IS_DESTROYED.
  bool gone = false;
};

struct ClientThread {
  std::vector<SavedContext> stack;  // bottom first; back() is current
  // Also per-thread in CUDA, and not kept yet: the stream capture mode, and
  // the deferred error of a call sent without a reply.
};

// A session's client threads, and what the serving thread has current.
//
// Touched only by the thread serving the session, which is also the thread
// that clears it at expiry, so it takes no lock: nothing else - not the
// handshake, not another session - reads or writes it.
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
  return top.ctx == applied.ctx && top.gen == applied.gen;
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

// The interception of cuCtxSetCurrent, cuCtxPushCurrent_v2 and
// cuCtxPopCurrent_v2, or nullptr for any other name. See driver_wrapper().
void* client_threads_wrapper(const char* name);

// Told by the inventory's wrappers, after the driver call succeeded: a context
// was created, and pushed on the calling thread as CUDA does; a context was
// destroyed; a device's primary context was reset.
void client_threads_created(CUcontext ctx);
void client_threads_destroyed(CUcontext ctx);
void client_threads_reset(int dev);

}  // namespace rgpu
