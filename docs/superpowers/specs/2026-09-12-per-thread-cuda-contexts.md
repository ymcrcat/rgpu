# Per-thread CUDA contexts across the wire

This describes what was built for issue #2, in slices A, B, C1, C2a and C2b.
The first draft planned some of it differently; where the build departed from
the plan, the text below says what was built, and "What changed from the
first draft, and why" at the end says what moved and what forced it.

## Why this exists

CUDA's context contract is per host thread. Each thread has its own current
context and its own context stack, and a call takes its meaning from whichever
context the calling thread last made current. That is the promise every CUDA
program is written against, including PyTorch's.

rgpu broke it. Every call from a client process is served by one server
thread, so there was exactly one current context for the whole process, and
the last client thread to change it won. Worse, the runtime shim cached the
fact that a thread had selected its context in a thread-local (`t_ctx_set`,
`client/cudart_impl.cpp`), so once a thread had selected once it never
selected again - even after another thread had moved the ground under it.

The failure is issue #2's scenario, and it is silent:

1. Client thread A calls `cudaSetDevice(0)`; the shim retains device 0's
   primary context and makes it current on the server thread.
2. Client thread B calls `cudaSetDevice(1)`; the same server thread's current
   context is now device 1's.
3. Client thread A calls `cudaMalloc`. Its `t_ctx_set` is still true, so it
   sends no context call, and the allocation lands on device 1.

No error is returned. A pointer comes back and it belongs to the wrong device.

The global RPC mutex is not a defence. It gives every request a total order,
which is what makes the frames well-formed; it says nothing about which
context each of them runs under. Batching makes it one degree worse: a run of
`kFlagNoReply` frames queued by thread A can be flushed by thread B's next
synchronous call, so A's launches go out and execute after B has already
changed the context.

The fix carries client thread identity in the protocol, so the server can put
each request back under the context the client thread that issued it
selected. Nothing else recovers the contract: the information simply was not
on the wire.

## Goals

**Primary:** every operation runs under the context the issuing client thread
selected, including when calls from other client threads intervene, and
including when those calls were batched.

Also in scope, because they are the same contract and the thread id is what
makes them possible:

- the context stack (`cuCtxPushCurrent` / `cuCtxPopCurrent`) is per client
  thread, not per session
- the stream capture mode is per client thread
- a deferred failure of a call sent without a reply surfaces on the client
  thread that caused it, not on whichever thread happens to call next
- a context destroyed by one client thread makes other threads' saved currency
  fail cleanly instead of pointing at freed driver state - or, worse, at a new
  context that was given the same address

**Not in scope:** concurrency. A multithreaded client is correct and still
serialized - one socket, one server thread, one call at a time. Making two
client threads drive two GPUs at once is a different piece of work, and this
design is its prerequisite rather than its delivery.

## What was verified before this design

Confirmed by reading the code as it stood before slice A. The line numbers are
from that tree and have since moved.

| Claim | Where |
|---|---|
| `t_ctx_set` short-circuits context selection per client thread | `client/cudart_impl.cpp:96`, cleared only at `:375` (`cudaSetDevice`) and `:480` (`cudaDeviceReset`) |
| One server thread serves a whole session across reconnects | `server/main.cpp:411` `serve_session`, and the comment at `:225` saying it *has* to be the same thread |
| `ReqHeader` is five `uint32_t` and goes on the wire as a memcpy of the struct | `common/wire.h:48`, `common/net.h:63` |
| `flags` has exactly one bit defined | `kFlagNoReply`, `common/wire.h:44` |
| Both ends refuse a version mismatch and close | `client/rpc.cpp:211`, `server/main.cpp:467` |
| `cuCtxSetCurrent`, `cuCtxPushCurrent_v2`, `cuCtxPopCurrent_v2`, `cuCtxGetCurrent` and `cuCtxCreate_*` are all generated and fully remoted | `codegen/api.json`, `codegen/emit.py:242` `emit_server_case` |
| The replay buffer holds whole frames as raw bytes | `client/rpc.cpp:122` `SentFrame`, replayed verbatim at `:237` |
| Acknowledgement is "everything up to this id", which assumes total order | `forget_acked_locked`, `client/rpc.cpp:135`; `session->last_req`, `server/main.cpp:387` |
| The cached last reply is a single slot per session | `server/main.cpp:252`, resent at `:513` |
| `pending_async` is one slot per session, handed to the next replying call whichever thread sent it | `server/main.cpp:247`, `:383` |
| The inventory records the context that was current at creation, read from the driver | `server/inventory.cpp:37` `current_ctx()`, used by `note()` |
| The inventory's release runs on the session thread and makes contexts current itself | `release_inventory`, `server/inventory.cpp`; the `Current` helper |
| The fake driver has one device and one primary-context token for all of them | `tests/fake_cuda.cpp:97` (`cuDeviceGetCount` returns a hard 1), `:71` (`primary_token()` is the constant `0xC0FFEE01`, despite a comment saying one per device) |
| The fake keeps a `t_current` and tags allocations with it, but no operation checks the tag | `tests/fake_cuda.cpp:67`, `:27` (`Alloc::ctx`), `:73` (`range_ok` ignores it) |

Assumed, not measured:

- `cuCtxSetCurrent` on an already-retained context is cheap in the real driver
  (thread-local bookkeeping, no ioctl). The design leans on this for the
  alternating-threads case.
- `cuCtxGetCurrent` is a thread-local read and costs nothing next to a socket
  read. The server calls it once per request.
- No client we care about migrates CUDA work between OS threads (green
  threads, fibers, work-stealing over a thread pool where a logical task moves
  mid-context). PyTorch does not.

Still not measured on hardware, and worth one cheap single-GPU check: that a
copy or free through another context's pointer succeeds (the fake assumes the
header's placement inference), that `cuCtxSetCurrent` refuses a destroyed
handle, that a reset keeps the retain count, and what `cuCtxGetCurrent` costs.

## Thread identity on the wire

`ReqHeader` grew one field:

```
struct ReqHeader {
  uint32_t magic;
  uint32_t api_id;
  uint32_t req_id;
  uint32_t flags;
  uint32_t thread_id;   // new
  uint32_t payload_len;
};
```

Six `uint32_t`, 24 bytes, no padding on x86_64 or aarch64 - which matters,
because the struct is written and read by memcpy. `common/wire.h` asserts the
size, and `tests/test_wire.cpp` pins the offsets. `kProtocolVersion` went from
2 to 3.

The alternative of stealing bits from `flags` was considered and rejected: a
thread id needs range, and `flags` is a bitfield with one bit spoken for. A
separate field costs four bytes on a frame that is already at least 24 and
usually far more.

### A client-assigned id, not an OS tid

The id is minted by the client: a plain `thread_local uint32_t`, set on the
thread's first RPC from a process-wide atomic counter starting at 1. Zero is
reserved and never valid, so a zero-filled or truncated header is caught
rather than silently attributed to some thread. The counter wraps after about
4.3 billion threads that ever called, and after that it hands out ids live
threads still hold; the minting site says so.

Not the OS thread id, for four reasons:

- **Portability.** `gettid()` is Linux; macOS has `pthread_threadid_np` and it
  is 64-bit. The client builds on both. An opaque id is the same everywhere.
- **Recycling.** The kernel reuses tids. A thread that exits and a new one
  that inherits its number would inherit the dead thread's saved context on
  the server - the same class of bug we are fixing, in a rarer and much harder
  form. An id the client mints is never reused.
- **Density.** Tids are sparse and can be large. A counter makes "this client
  has too many live threads" a condition the server can state and check.
- **Lifetime control.** Because the client owns the id, it can say when one
  is finished. It cannot say that about a tid it does not own.

`call()` and `call_async()` read the id before taking `g_mu` and pass it into
`queue_frame_locked`. It is thread-local so it cannot differ either way, but
passing it documents that the id belongs to the calling thread rather than to
whoever eventually flushes the queue - which is precisely the distinction that
makes cross-thread batching correct.

### Handshake and negotiation

Nothing new in the handshake. `Handshake.version` already existed, both sides
compare it strictly, and the version bump is the whole negotiation. There is
deliberately no capability bit and no dual-format parsing: the client shim and
the server are built from the same tree by the same codegen, and supporting
two header sizes would mean `recv_frame` branching on a per-connection version
for the rest of time.

The server writes its `HandshakeReply`, carrying its version, *before* closing
on a mismatch, so the client can say "server speaks protocol 3, this client
speaks 2" instead of "handshake failed".

### Retiring an id

A `thread_local` guard object's destructor pushes the finished id onto a small
`g_retired` vector under its own `g_retired_mu`, and sets an atomic flag. No
I/O in the destructor - PyTorch threads call into the shim while the process
is exiting. It is not `g_mu`, because a thread exiting while another thread's
call waits on the network would otherwise wait inside its thread-exit path.

When the flag is set, the next frame queued by any thread is preceded by an
`rgpu_thread_gone` internal call carrying the retired ids, with
`kFlagNoReply`. Two rules keep it honest:

- **A thread never announces itself.** Thread-local destructors run in reverse
  order of construction, so anything a thread built before its first RPC -
  PyTorch's per-thread caches among them - destructs after the retirer and may
  still call.
- **A late call is announced again.** A retired thread that queues a frame
  lists its id again, so a notice always follows the thread's last frame.

So the server must expect frames for an id after its notice. It keeps the 64
most recently retired slots (`kRetiredSlots`, `server/main.cpp`); a late call
for one of them gets its slot back, context, stack and all. A late call from
further back gets an empty slot, which the next notice drops again.

Best-effort by construction: a lost notice leaks a slot until the session
expires, which is a bound the session already imposes on everything else.

## Per-client-thread state on the server

The session gains a table, alongside - and deliberately separate from - the
inventory (`server/client_threads.h`):

```
struct SavedContext {
  CUcontext ctx;
  bool gone;                           // destroyed since it was saved
};
struct ClientThread {
  std::vector<SavedContext> stack;     // bottom first; back() is current
  CUresult pending_async;              // a failure with no reply to report it
  CUstreamCaptureMode capture_mode;    // GLOBAL, the default, to start
};
struct ClientThreads {
  std::unordered_map<uint32_t, ClientThread> live;
  std::deque<std::pair<uint32_t, ClientThread>> retired;   // at most 64
  SavedContext applied;                // what the serving thread has current
  CUstreamCaptureMode applied_mode;    // and its capture mode
  ClientThread* caller;                // the thread whose request is running
  ...                                  // said-once flags for the log
};
```

**The table is not under `session->mu`.** Only the serving thread reads or
writes it, and it is also the thread that clears it at expiry; nothing else -
not the handshake, not another session - touches it. The one exception to
"no lock" is `pending_async`, which is written under `session->mu`, not to
protect it but because holding a failure is part of the step that records the
request as completed (see "Deferred errors are per thread").

**`applied` lives on the session, not in `serve()`.** `serve()` runs once per
connection, but the serving thread and its driver state outlast every
connection. A local reset to null on reconnect would let a thread with no
context skip its restore and run under whatever the previous thread left
current - and then have the readback record that context as its own.

The two structures answer different questions and must not be merged. The
inventory answers *what does this session own and owe back*; the thread table
answers *what was thread T looking at*. Nothing in the thread table is owned
by it. Every `CUcontext` in there is a borrowed handle to something the
inventory accounts for, and the single most important rule of this design is
that **session expiry must not release anything from the thread table**.
Clearing it is a `.clear()` and nothing more.

The two do interact, in a way that is entirely to our benefit. The inventory
records each resource against the context current at the moment of creation.
Before, that could be the wrong context, because the wrong context was
current. With the issuing thread's context restored before dispatch, the
inventory records the right one with no change of its own, and
`release_inventory`'s per-context ordering is correct rather than
accidentally correct. That is the strongest argument for restoring *before*
dispatch rather than doing anything cleverer.

At expiry, `serve_session` unbinds the table, logs any failure a slot still
holds, clears the table, and only then calls `release_inventory`: first
because the release moves the current context around freely, so any slot
surviving into that window would be stale; and because clearing first makes it
structurally impossible for someone to later "helpfully" release contexts
named in the table and double-free what the inventory owns.

### The slot's stack is the truth, and the driver's stack is one deep

The serving thread's own driver stack only ever holds the top of whichever
client thread was served last, or nothing; `applied` records which. So:

- **Switching** threads is one `cuCtxSetCurrent`, which replaces the top.
- **Applying "no context"** is one pop, and it cannot uncover anybody's
  context, because nothing is underneath.
- No thread's stack is ever popped out of the driver to be saved, or pushed
  back to be restored.

That only holds if every call that changes the depth of the driver's stack is
carried out on the issuing thread's slot instead. They are intercepted through
`driver_wrapper()`, bound to the session with `client_threads_bind` the way
`inventory_bind` binds the inventory, so the generated dispatch learns nothing
about any of it and `codegen/emit.py` stays out of the change:

| Call | On the thread's stack | Driver calls |
|---|---|---|
| `cuCtxSetCurrent(x)` | replaces the top, or pushes onto an empty stack | 1 |
| `cuCtxSetCurrent(NULL)` | pops, then shows the new top | 0 or 1 |
| `cuCtxPushCurrent_v2(x)` | pushes | 1 (`cuCtxSetCurrent(x)`) |
| `cuCtxPushCurrent_v2(NULL)` | refused, `CUDA_ERROR_INVALID_CONTEXT` | 0 |
| `cuCtxPopCurrent_v2` | pops, shows the new top; returns the popped handle | 0 or 1 |
| `cuCtxGetCurrent` | answers the slot's top, even a destroyed one | 1 (for its errors) |
| `cuCtxCreate_v2` | pushes (inventory wrapper hook) | 2, only if a context was current: back to one entry |
| `cuCtxDestroy_v2`, `cuCtxDetach` | pops the caller's top if it names the context; sweeps | 0 or 1 |
| `cuThreadExchangeStreamCaptureMode` | records the thread's new mode | 1 |

**Readback is kept, as the safety net.** After every request the server asks
the driver what is current. The interception keeps that equal to `applied`,
so normally nothing happens; a call nobody listed that changed the context
anyway is taken at its word, as a replaced top, or a popped one if nothing is
current. Readback alone was never sufficient - `cuCtxGetCurrent` reports only
the top, not the depth - which is why the stack calls are intercepted.

**Leaving the serving thread with no context** pops until nothing is current,
bounded to 8 pops. More than one entry is tolerated but not relied on - a call
the interception does not know may have pushed - and every context on it is
held in some slot, so popping takes nothing from anybody. A driver that will
not let go gets the request refused with its own code, logged once per
session, rather than an endless loop on every request.

### Applying a thread's state per call

The serving loop, for each frame, in this order:

1. **At most once.** If the request id is at or before the last request the
   session completed (`req_at_or_before`, `common/wire.h`, which orders ids
   across the 32-bit wrap), it is not run: the cached reply is sent again if
   it is that request's, and otherwise it is skipped. This sits before
   everything else - the refusals, the thread table, the restore, the
   readback - so a copy of a request changes no slot and no held failure.
2. **Thread 0 is refused** with `CUDA_ERROR_INVALID_VALUE`, logged once per
   session. The connection stays open: closing it would make a real client
   reconnect and replay the same frame indefinitely.
3. **`rgpu_thread_gone` is handled without a slot.** It changes the table
   rather than running under it, and the thread sending it may be new - at the
   cap it is the only thing that frees room.
4. **The slot.** One `find`; a thread not live goes through `admit_thread`,
   which gives it its retired slot back if it has one. Past
   `RGPU_MAX_CLIENT_THREADS` live threads (4096 by default) the call is
   refused with `CUDA_ERROR_INVALID_VALUE`, logged once per session.
5. **Restore, only if needed.** `client_thread_shown` compares the thread's
   top with `applied` and the thread's capture mode with `applied_mode`,
   separately: threads that share a context still each have a mode. Only if
   either differs does `client_thread_show` run one `cuCtxSetCurrent` (or a
   clear) and one mode exchange.
6. **Dispatch**, with `threads.caller` set so the intercepted calls find the
   issuing thread's slot.
7. **Readback** and the destroyed-context correction (below).
8. **Complete** the request - and hold its failure, or hand a held one to its
   reply - in one critical section under `session->mu`.

**Null is applied like any other value.** A brand-new client thread that has
selected nothing gets no current context, and a context-requiring call
correctly fails with `CUDA_ERROR_INVALID_CONTEXT` instead of quietly
inheriting whatever the previous thread left behind. Calls that genuinely need
no context - `cuInit`, `cuDeviceGet`, `cuDeviceGetCount`,
`cuDevicePrimaryCtxRetain` - work fine without one. So there is no exception
list to maintain.

**When the restore is skipped.** A single-threaded client hits the "already
shown" branch on every call after its first selection, so it pays no
`cuCtxSetCurrent` and no mode exchange at all. So does a run of consecutive
calls from the same thread, which is what a multithreaded client's traffic
mostly looks like, and threads that share a context and a mode. Only genuine
ping-pong between threads on different contexts pays one switch per call.

### Contexts that die under another thread

A created context's handle is an address in driver memory, and once the
context is destroyed that address may be handed to the next `cuCtxCreate`
anywhere in the process, another tenant's included. A slot that still named
it would bind a thread to somebody else's context.

- **Sweep on destroy.** When `cuCtxDestroy_v2` or `cuCtxDetach` succeeds, the
  issuing thread pops its top if that is the context - the header: "If ctx is
  current to the calling thread then ctx will also be popped" - and every
  other entry naming it, in every live and retired slot, is **marked** `gone`.
  Not struck: CUDA leaves a destroyed context current to the other threads
  that had it ("ctx will remain current to those threads"), so popping what
  was pushed on top of it has to find it again. A gone entry is never made
  current.
- **A gone top.** The serving thread is left with no context and the call is
  dispatched: a selection or a push succeeds, and a call that needs a context
  fails on its own with nothing current to misuse. That failure comes back
  from the driver as `CUDA_ERROR_INVALID_CONTEXT`, and while the thread's top
  is still the destroyed entry the server corrects it to
  `CUDA_ERROR_CONTEXT_IS_DESTROYED`, which is what CUDA returns to such a
  thread. `cuCtxGetCurrent` still names the destroyed handle, as CUDA does.
  The error lasts exactly until the thread selects, pushes over the entry or
  pops it.
- **Calls that report on a context they were handed** keep the driver's own
  `CUDA_ERROR_INVALID_CONTEXT`, since it is about their argument, not the
  current context: `cuCtxSetCurrent`, `cuCtxPushCurrent_v2`,
  `cuCtxPopCurrent_v2`, `cuCtxDestroy_v2`, `cuCtxDetach`,
  `cuCtxGetApiVersion`, `cuCtxGetId`, `cuCtxEnablePeerAccess`,
  `cuCtxDisablePeerAccess`, `cuCtxRecordEvent`, `cuCtxWaitEvent`,
  `cuCtxGetDevResource`, `cuMemcpyPeer`, `cuMemcpyPeerAsync`,
  `cuGraphAddMemcpyNode`, `cuGraphAddMemsetNode`,
  `cuGraphConditionalHandleCreate`, `cuGraphExecMemcpyNodeSetParams`,
  `cuGraphExecMemsetNodeSetParams`, `cuDevicePrimaryCtxRetain` and
  `cuDevicePrimaryCtxRelease_v2`. Each takes a `CUcontext` argument. Calls
  that carry a context inside a structure (`cuMemcpy3DPeer`, kernel and generic
  graph node parameters) are not listed, because a null context there means
  the current one. The maths libraries' results are never corrected: only a
  driver call's result is a `CUresult`.
- **Fail clean on restore.** If `cuCtxSetCurrent` refuses a saved context
  anyway - destroyed where no sweep saw it - the entry is marked gone, the
  current context is read back into `applied`, and the thread is treated as
  above.

**Primary contexts are not swept, and carry no generation.** A primary
context's last release "automatically reset[s]" it, and a reset "does not
release it": either way it is emptied, not replaced, and its handle survives.
So a thread that had it current keeps it current through another session's
last release or a device reset, as in CUDA, and goes on allocating without
selecting again. If a real driver ever handed out a different handle for a
device's primary context afterwards, restoring the old one would fail and take
the fail-clean path above.

**What is refused instead of tracked.** `cuCtxAttach` would make `cuCtxDetach`
something other than a destroy, and the green-context family
(`cuGreenCtx*`, `cuCtxFromGreenCtx`) hands out contexts that carry no record
and whose destroy releases a primary context unseen. All of them return
`CUDA_ERROR_NOT_SUPPORTED`, logged once per process.

**The stack is capped.** A thread's stack lives in server memory, the
driver's own stack stays one deep and so never limits it, and every destroy
sweeps every entry. A push or create past `RGPU_MAX_CONTEXT_STACK` entries (64
by default) is refused with `CUDA_ERROR_INVALID_VALUE`, which both calls
document; a create is refused before the driver is called, so no context is
made. The default is small because it multiplies with the thread cap.

### Stream capture mode

`cuThreadExchangeStreamCaptureMode` sets and returns "a thread's mode", GLOBAL
being "the default mode", and `cuCtxGetCurrent` says nothing about it. It is
the only call that changes the mode, and `handle_capture_mode` was already
hand-written, so the slot records the mode the thread asked for and the
serving thread is given it before the thread's request, one exchange and only
when it differs from `applied_mode`. The mode the driver hands back is the
serving thread's, which by then is the issuing thread's. An exchange that
fails refuses the request, logged once per session, rather than running it
under another thread's mode - which would not fail anything obvious: it would
invalidate a capture, or let through a call the thread's own mode forbids.

What is *not* per client thread is the capture itself. To the driver, every
client thread's `cuStreamBeginCapture` is the one serving thread's, so a
capture not begun RELAXED may be ended from another client thread, and it
restricts every client thread not in RELAXED mode as its own capture would,
THREAD_LOCAL included. The server says so in its log.

Beyond that: we handle what we can enumerate. See Risks.

## One server thread per session, or one per client thread?

**One per session. Unchanged.** The per-thread state moves into the session;
the execution does not.

A thread per client thread is genuinely attractive - it would give real
concurrency, and CUDA's thread state would live where the driver naturally
keeps it, with no save, no restore and no readback. It also breaks three
things that work.

**Ordering.** `req_id` comes from a single counter minted under the client's
global mutex, and both sides read "up to id N" as "everything before N is
done". `forget_acked_locked` drops every unacked frame at or before the id it
was told about, and the serving loop refuses to run any request at or before
the last one completed. With N serving threads, request 7 can finish before
request 5, and the client would drop frame 5 from its replay buffer while
frame 5 was still in flight. The acknowledgement scheme is not adaptable to
this without becoming per-thread, which means N counters, N windows, and a
handshake that reconciles all of them.

**The cached reply.** `session->last_reply` is one slot - the reply that may
have been lost when the connection died. With N serving threads there are N
last replies and no way for the client to say which one it missed with a
single `last_req_id` in its handshake.

**It buys nothing yet.** The client is serialized: `call()` takes `g_mu`,
writes, and blocks reading the reply. One socket, one outstanding request.
Fanning out on the server without also fanning out the client just moves the
queue. Fanning out the client means per-thread connections, per-thread request
id spaces and a per-thread replay window - a much larger change than issue #2,
and one that should be justified by a measurement rather than by this bug.

So, said plainly, what the choice preserves and what it costs:

- Total order across a session: **preserved**.
- `session->last_req` meaning "everything up to here completed":
  **preserved**, and completing a request is still one critical section with
  holding its failure and caching its reply.
- Replay after a dropped connection: **preserved, for free.** `SentFrame`
  holds whole frames as bytes, so a replayed request carries the same
  `thread_id` it was sent with and lands under the same context. Not one line
  of the replay path changed.
- The single cached reply: **preserved**.
- The slot table, `applied` and every held failure surviving a reconnect:
  **yes**, because they hang off the `Session`, which is what survives.
- Concurrency: **still none.** Two client threads on two GPUs are correct and
  serialized. The server says so once, the first time a session admits a
  second distinct thread id, so nobody diagnoses this as a mystery. "A second
  distinct id ever admitted", not "more than one live at once": notices are
  processed ahead of a new thread's frames, so a client whose threads run one
  after another never has two live slots, and the per-thread state that is not
  modelled matters to it too.

### Deferred errors are per thread

A call sent without a reply has nowhere to report a failure, so the server
holds the first one and hands it to a later call that replies. That slot is
`ClientThread::pending_async`, not a field of the session: thread A's failed
launch is handed to A's next call that replies, and B, whose call may have
flushed A's launch, is told nothing. Only a call that succeeded on its own
carries a held failure; one that failed reports its own.

It covers the server's own refusals of no-reply frames as well as the
driver's failures: past the stack cap, after the thread's context was
destroyed, and from a thread since announced gone, whose late call gets the
failure back with its slot. Two refusals have no slot to hold it in:

- **At the thread cap**, the refused thread's failure is put in a fresh slot in
  the retired set, which `admit_thread` hands back once the thread is let in.
  Without that, a launch refused at the cap - followed by a notice that makes
  room, which is exactly how the client sends frames - would look like a
  success. The same goes for a failed `rgpu_thread_gone`, held for its sender.
- **A frame naming thread 0** has no thread that could ever be told, since every
  call it sends is refused. Its failure is reported to no thread, and the
  refusal's once-per-session log says so.

**At most once is preserved.** The failure is stored in the same critical
section, under `session->mu`, that records the request as completed, and a
held one is taken in the same section that caches the reply. A copy of a
request that already ran is skipped before any of this, so it touches no
slot's failure.

**A failure left on a thread that never calls again.** If a slot is forgotten
while it still holds one - evicted from the retired set, or at session expiry
- the server logs it: that thread will never make the call that would have
carried it. This is the right end for it. In CUDA a sticky error belongs to the
context, not the thread, and survives the thread's exit: every later call in
that context fails with it, from any thread - and a real driver still does
that here, on its own, so nothing about per-thread holding hides one. A
thread's last error dies with the thread. What the server holds is neither: it
is the ordinary result of one call, which CUDA would have returned to the
thread at the call site, so it goes with its thread.

## The client side

`t_ctx_set` is replaced by a generation, `t_ctx_gen` compared against a global
`g_ctx_gen`, and for the first time it tells the truth: it is a claim about
*this thread's slot*, which nothing else can touch. Caching it is then sound,
and it is worth keeping: it is what stops every runtime call from carrying a
redundant `cuCtxSetCurrent` round trip.

What invalidates it:

- `cudaSetDevice` to a different device sets `t_ctx_gen = 0`.
- `cudaDeviceReset` bumps `g_ctx_gen` under `g_ctx_mu`, which releases every
  primary context process-wide and so invalidates every thread's selection,
  for one integer compare. `ensure_context` reads the generation under the
  same lock it reads the primary-context record with, so a reset landing in
  between leaves the new selection already stale, never falsely fresh.
- A reconnect: **nothing**. A resumed session keeps its slots, so the cached
  fact is still true. A session that could not be resumed already gives up.

`client/shim.cpp` needed no change for currency: the `cuCtx*` calls stay fully
remoted and simply land on the calling thread's slot. The client change is in
`client/rpc.cpp` (mint, stamp, retire) and the generation in
`client/cudart_impl.cpp`. Thread A's queued no-reply frames each carry A's id,
so when B's `call()` flushes them the server puts each one back under A's
context before running it. The batching instance of the bug closes because
identity is per frame rather than per connection or per flush.

## Compatibility

A clean refusal at the handshake, in both directions. No fallback, no
downgrade path. Both were checked with real binaries of both versions.

- **Old client (v2) to new server (v3):** the server replies with its version
  and closes. The client says `server speaks protocol 3, this client speaks 2`
  and every call returns `CUDA_ERROR_NOT_INITIALIZED`.
- **New client (v3) to old server (v2):** the old server closes without
  replying, so the client reports "handshake failed" rather than a version
  mismatch. Correct behaviour, poor message, and not fixable retroactively.
  Every mismatch from v3 onward reports both versions.

Both sides then refuse to proceed, which is the right outcome: a v2 server
cannot honour the contract, and running against it anyway is exactly the
silent wrong-device bug we are here to remove.

## Testing without a GPU

The fake driver was why this bug was invisible: one device, one primary
context token for all of them, and a context tag on allocations that no
operation looked at. It now models what the header documents
(`tests/fake_cuda.cpp`):

1. **More than one device.** `RGPU_FAKE_DEVICES` sets the count, defaulting to
   1 so no existing test changed behaviour.
2. **A distinct primary context per device**, `0xC0FFEE01 + dev`, with its own
   retain count; the last release resets it, and a reset keeps the count.
3. **A per-thread context stack**, and a per-thread capture mode.
4. **Objects carry their context, and are used there.** A copy, memset or free
   through another context's pointer succeeds, because the header says the
   driver infers placement from the pointer's value. Placement is what is
   observable: `CU_POINTER_ATTRIBUTE_CONTEXT`,
   `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` and `cuCtxGetDevice`.
5. **Destroyed context handles handed out again**, with
   `RGPU_FAKE_REUSE_CONTEXTS=1`, so a test can show a stale handle never binds
   the context that took its address.
6. **Counters** (`tests/fake_stats.h`), published to `RGPU_FAKE_STATS`:
   resources and mistakes (`stale`, `overreleases`) that must come back to
   zero, and observations that need not - `ctxsets` (every
   `cuCtxSetCurrent`), `crossctx` (device-memory reads and writes that ran
   under a context other than the memory's own), `modeswaps` (capture mode
   exchanges) and `memsets`.

No test depends on a behaviour the fake infers rather than the header
documents. The tests assert placement, `cuCtxGetCurrent`, the fake's counters
and the server's own rules - never that a use under the wrong context fails.
`tests/test_fake_cuda.cpp` checks the fake itself from several OS threads.

### The tests that would have caught the original bug

`tests/threadctx_smoke.cpp`, run from `tests/run_smoke.sh` on its own server
with `RGPU_FAKE_DEVICES=2`, `RGPU_BATCH=1`, low thread and stack caps, and
handle reuse on:

1. **The issue's scenario, by placement.** Thread A: `cudaSetDevice(0)`,
   `cudaMalloc`. Thread B: `cudaSetDevice(1)`, `cudaMalloc`. Thread A:
   `cudaMalloc` again. `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` must say device 0
   for both of A's allocations and device 1 for B's.
2. **Currency readback.** Thread A `cuCtxSetCurrent(P0)`, thread B
   `cuCtxSetCurrent(P1)`, thread A `cuCtxGetCurrent` must return `P0`. The
   smallest possible regression test for the whole design; written first and
   watched to fail.
3. **Stack depth.** Thread A pushes over its context; thread B pushes and pops
   over another; each pops back to its own. Also: a stack survives another
   thread's context-free call, and a dozen created contexts survive a thread
   with none.
4. **Batching across threads.** Thread A queues memsets without replies under
   device 0; thread B's synchronous call under device 1 flushes them. The
   fake's `memsets` must not move while they are queued and must move by all
   of them at B's call - so the test fails if batching stopped - and
   `crossctx` must not move at all.

Beyond those four: one thread's calls cost no `cuCtxSetCurrent` and no mode
exchange; a context destroyed or detached by another thread, whose address a
new context then took, is never rebound and gives
`CUDA_ERROR_CONTEXT_IS_DESTROYED`; a thread recovers from it; a primary context
stays current through another session's last release and through a reset
(`threadctx_smoke reset`, alone on a server, which also checks the default
stack cap); `cuCtxAttach` and green contexts are refused; a failure with no
reply surfaces on the thread that caused it, including the server's own
refusals; two threads sharing a context keep their own capture modes; and, on
raw connections of the test's own, a late call after its thread's notice keeps
its context, the retired set and the live threads are bounded, and a thread
with no context still has none after a reconnect.

Extensions to tests that already existed:

- **`tests/reconnect_smoke.cpp threads`**: two threads on two devices queue
  memsets, and the server breaks the connection in the middle of the batch.
  After the replay each thread's context is still its own and no memset ran
  under the other's. `run_smoke.sh` pins the number of calls sent again, so
  the break is known to land inside the batch.
- **`tests/expiry_smoke.cpp threads`**: two threads on two devices, one with a
  created context on top, exit holding everything. After expiry every counter
  is zero, both devices' retains included, and nothing is stale.
- **`tests/test_wire.cpp`**: `sizeof(ReqHeader) == 24`, the field offsets,
  the handshake layouts and `kProtocolVersion == 3`.
- **`tests/hostile_smoke.cpp`**: a v2 handshake gets a reply naming v3 and is
  closed, and a frame with `thread_id == 0` is refused while the connection
  keeps working.
- **`tests/thread_id_smoke.cpp`**: against a scripted server, the exact frames
  the client writes - ids per frame through a cross-thread flush and a replay,
  retirement, late calls - and at runtime, a reset on one thread invalidating
  another's selection.

## Order of work, as it landed

1. **Slice A.** The fake driver: devices, primary contexts per device,
   per-thread stacks, objects carrying their context. Nothing existing went
   red.
2. **Slice B.** The wire field and version 3, reply-then-close on a mismatch,
   minting, stamping and retiring ids, and `t_ctx_gen`.
3. **Slice C1.** The slot table, restore before dispatch, readback after,
   `applied` on the session, the thread and retired bounds, thread 0, tests 1,
   2 and 4.
4. **Slice C2a.** The per-thread stack with the driver's one deep, the destroy
   sweep and marked entries, `CUDA_ERROR_CONTEXT_IS_DESTROYED`,
   `cuCtxGetCurrent` from the slot, detach and the refusals, primary contexts
   keeping currency, the stack cap, test 3.
5. **Slice C2b.** Per-thread deferred errors and capture mode, the reconnect
   and expiry extensions, proof that test 4 batches, the longer pass-through
   list and a stack cap of 64.

Between C2a and its review follow-ups a separate fix made the serving loop
at-most-once and ordered request ids across their wrap; C2b kept it so.

## Risks

**What it costs the single-threaded case.** Per request: one hash lookup, a
context and a mode compare, one `cuCtxGetCurrent`, and four bytes per frame. No
`cuCtxSetCurrent` and no mode exchange, because the skip covers them, and a
test holds that to zero. `tests/bench.cpp` against `rgpu-server-fake` showed
the protocol change inside run-to-run noise (round trips 28.5-29.7 µs before,
28.7-29.3 µs after) and the slot table likewise (17.7-18.3 µs before,
17.6-17.7 µs after). On the fake, `cuCtxGetCurrent` is a thread-local read;
its cost on hardware is unmeasured. If it does show up, the fallback is to
read back only after calls from a name list, trading the robustness argued
for above for a few nanoseconds. Take the measurement before taking the trade.

**Thread state we have not enumerated.** Currency is covered by interception
plus readback. The stack, the capture mode and deferred failures are covered
by interception. Which thread began a stream capture is known not to be
covered. Anything else CUDA keeps per thread is not covered, and we do not
have a complete list of what that is. This is the design's honest weak point,
which is why the server logs once, loudly, when a session first has more than
one client thread, naming what it keeps and what it does not. A client that
migrates CUDA work between OS threads is outside CUDA's contract, we key on the
OS thread, and we would silently give it the wrong slot; we cannot detect it.

**The slot table is client-controlled memory.** The protocol has no
authentication, so a client can mint ids and push contexts without bound. Live
slots are capped at `RGPU_MAX_CLIENT_THREADS` (4096), retired ones at 64, and
each stack at `RGPU_MAX_CONTEXT_STACK` (64); past a cap the call is refused.
At the defaults a session holds a few megabytes of stack entries at most, and a
destroy sweeps at most a few hundred thousand of them.

**Contexts from another session.** A created context's handle used from
another session's connection is not covered: another session's destroy is
neither swept here nor recorded. Reaching it needs a handle from another
tenant.

**Per-thread deferred errors change observable behaviour.** An error caused by
thread A no longer surfaces on thread B. That is more correct and it is what
CUDA does, but it is a behaviour change, and anything that happened to depend
on the old leak will notice. Nothing single-threaded can.

**Concurrency expectations.** The most likely way this design disappoints is
that someone reads "per-thread contexts" as "per-thread parallelism". It is
not. The doc says so in Goals, the server says so in its log, and the
follow-up issue for per-thread connections should be filed so the gap is on
the record rather than in someone's head.

## What changed from the first draft, and why

- **The table is not under `session->mu`, and `applied` lives on the
  session.** The draft put the table under the session lock and `applied` in
  `serve()` as a plain local. Only the serving thread touches the table, so a
  lock protects nothing; and `serve()` is per connection while the serving
  thread's driver state is not, so a local `applied` reset to null on
  reconnect would run a thread with no context under the last thread's
  context. A raw-frame test now pins it.
- **Test 1 asserts placement, not a refused memcpy.** The draft had the fake
  refuse a memcpy into another context's pointer. The header says the driver
  infers placement from a pointer, so on hardware that copy most likely
  succeeds; a test resting on its failure would have asserted fake-only
  behaviour. The fake infers placement too, and the tests read where memory
  went.
- **The driver's stack is kept one deep, and the stack calls act on the
  slot.** The draft intercepted push and pop on top of readback and restored
  only the top. That flattened stacks across threads: applying a new thread's
  "no context" with one `cuCtxSetCurrent(NULL)` uncovered whatever was below,
  and replacing the top exposed other threads' entries to a pop. Keeping the
  driver one deep makes a switch one call and "no context" one pop that
  uncovers nothing. It needed create and destroy intercepted too, which the
  draft's "closed set of two" did not foresee.
- **Destroyed entries are marked, not struck.** The draft said "strike it from
  every stack". CUDA leaves a destroyed context current to other threads and
  a pop returns it, so striking would make a pop skip past it. A marked entry
  is never made current, fails the thread's calls with
  `CUDA_ERROR_CONTEXT_IS_DESTROYED`, and is still named by `cuCtxGetCurrent`.
- **Primary contexts keep currency and carry no generation in slots.** The
  draft swept on `cuDevicePrimaryCtxReset`, and C2a first stamped each saved
  primary context with a generation. The header says a reset "does not
  release" the context and a last release resets it: the handle survives, and
  a CUDA thread keeps it current. Sweeping made other threads fail where CUDA
  would not.
- **`cuCtxDetach` is swept, and `cuCtxAttach` and green contexts are
  refused.** The draft did not mention them. Detach destroys a created
  context as destroy does, so it has to sweep; attach would make detach
  something else, and green contexts are contexts nothing records.
- **Caps on the stack, and the thread cap refuses rather than closes.** The
  draft capped live slots only. A per-thread stack in server memory, swept on
  every destroy, needs its own cap, and its default came down from 1024 to 64
  because the two caps multiply. Refusals keep the connection open, because a
  real client would otherwise reconnect and replay the same frame forever.
- **The at-most-once check sits before everything in the loop.** The draft
  predates it. A replayed copy of a request that already ran must not run,
  and must not touch the thread table either - admitting a thread, holding a
  refusal or restoring a context are all side effects - so the check comes
  first, and holding a failure stays in the critical section that completes
  the request.
- **Deferred failures without a slot.** The draft moved `pending_async` to the
  slot and stopped there. Refusals at the thread cap and of thread 0 have no
  slot; the first now waits in a retired slot for its thread, the second is
  reported to nobody, and a failure forgotten with its slot is logged.
- **A thread never announces its own retirement, and late calls are announced
  again.** The draft's retirement assumed a thread's last call came before its
  notice. Thread-local destructors run in reverse order of construction, so
  they do not, and the server keeps recently retired slots for them.
- **`cuCtxGetCurrent` is answered from the slot.** The draft read it back from
  the driver. A thread whose top is destroyed has nothing current on the
  serving thread, but CUDA names the destroyed context, so the slot answers.
