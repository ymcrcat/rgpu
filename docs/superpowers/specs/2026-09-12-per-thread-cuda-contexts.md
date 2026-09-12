# Per-thread CUDA contexts across the wire

## Why this exists

CUDA's context contract is per host thread. Each thread has its own current
context and its own context stack, and a call takes its meaning from whichever
context the calling thread last made current. That is the promise every CUDA
program is written against, including PyTorch's.

rgpu breaks it. Every call from a client process is served by one server
thread, so there is exactly one current context for the whole process, and the
last client thread to change it wins. Worse, the runtime shim caches the fact
that a thread has selected its context in a thread-local
(`t_ctx_set`, `client/cudart_impl.cpp:82`), so once a thread has selected once
it never selects again - even after another thread has moved the ground under
it.

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
synchronous call (`client/rpc.cpp:374`), so A's launches go out and execute
after B has already changed the context.

The fix is to carry client thread identity in the protocol, so the server can
put each request back under the context the client thread that issued it
selected. Nothing else recovers the contract: the information simply is not on
the wire today.

## Goals

**Primary:** every operation runs under the context the issuing client thread
selected, including when calls from other client threads intervene, and
including when those calls were batched.

Also in scope, because they are the same contract and the thread id is what
makes them possible:

- the context stack (`cuCtxPushCurrent` / `cuCtxPopCurrent`) is per client
  thread, not per session
- a deferred asynchronous error surfaces on the client thread that caused it,
  not on whichever thread happens to call next
- a context destroyed by one client thread makes other threads' saved currency
  fail cleanly instead of pointing at freed driver state

**Not in scope:** concurrency. After this change a multithreaded client is
correct and still serialized - one socket, one server thread, one call at a
time. Making two client threads drive two GPUs at once is a different piece of
work, and this design is its prerequisite rather than its delivery.

## What was verified before this design

Confirmed by reading the code:

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
| The fake keeps a `t_current` and tags allocations with it, but no operation checks the tag | `tests/fake_cuda.cpp:67`, `:27` (`Alloc::ctx`, added by the inventory work for device reset), `:73` (`range_ok` ignores it) |

Assumed, not measured:

- `cuCtxSetCurrent` on an already-retained context is cheap in the real driver
  (thread-local bookkeeping, no ioctl). The design leans on this for the
  alternating-threads case.
- `cuCtxGetCurrent` is a thread-local read and costs nothing next to a socket
  read. The design calls it once per request; if that turns out to be false,
  see Risks for the fallback.
- No client we care about migrates CUDA work between OS threads (green
  threads, fibers, work-stealing over a thread pool where a logical task moves
  mid-context). PyTorch does not.

## Thread identity on the wire

`ReqHeader` grows one field:

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
because the struct is written and read by memcpy. `kProtocolVersion` goes from
2 to 3.

The alternative of stealing bits from `flags` was considered and rejected: a
thread id needs range, and `flags` is a bitfield with one bit spoken for. A
separate field costs four bytes on a frame that is already at least 24 and
usually far more.

### A client-assigned id, not an OS tid

The id is minted by the client: a `thread_local` initialized on the thread's
first RPC from a process-wide counter starting at 1. Zero is reserved and
never valid, so a zero-filled or truncated header is caught rather than
silently attributed to some thread.

Not the OS thread id, for four reasons:

- **Portability.** `gettid()` is Linux; macOS has `pthread_threadid_np` and it
  is 64-bit. The client builds on both (`build-macos/`). An opaque id is the
  same everywhere.
- **Recycling.** The kernel reuses tids. A thread that exits and a new one
  that inherits its number would inherit the dead thread's saved context on
  the server - the same class of bug we are fixing, in a rarer and much harder
  form. An id the client mints is never reused.
- **Density.** Tids are sparse and can be large. A counter from 1 makes the
  server's slot table small and makes "this client has too many live threads"
  a condition the server can state and check.
- **Lifetime control.** Because the client owns the id, it can say when one
  is finished. It cannot say that about a tid it does not own.

### Handshake and negotiation

Nothing new in the handshake. `Handshake.version` already exists, both sides
already compare it strictly, and the version bump is the whole negotiation.
There is deliberately no capability bit and no dual-format parsing: the client
shim and the server are built from the same tree by the same codegen, and
supporting two header sizes would mean `recv_frame` branching on a per-connection
version for the rest of time.

One improvement to make while here: the new server should write its
`HandshakeReply` (carrying its version) *before* closing on a mismatch, rather
than closing silently as `server/main.cpp:469` does today. The client can then
say "server speaks 3, this client speaks 4" instead of "handshake failed".

### Retiring an id

A `thread_local` guard object's destructor pushes the finished id onto a small
`g_retired` vector under `g_mu`. No I/O in the destructor - PyTorch threads
call into the shim while the process is exiting, and the stats code
(`client/rpc.cpp:44`) already carries the scar tissue from that. The next
frame queued by any thread emits a `rgpu_thread_gone` internal call carrying
the retired ids, with `kFlagNoReply`, ahead of its own frame. The server drops
those slots.

Best-effort by construction: a lost notice leaks a ~64-byte slot until the
session expires, which is a bound the session already imposes on everything
else. If this lands in slices, shipping without the notice at all is
acceptable for the first one; the slot table just grows with the number of
threads the process ever created.

## Per-client-thread state on the server

`Session` gains a table, alongside - and deliberately separate from - the
inventory:

```
struct ClientThread {
  CUcontext current = nullptr;        // what this client thread last made current
  std::vector<CUcontext> stack;       // cuCtxPushCurrent depth
  CUstreamCaptureMode capture_mode;   // also thread state; see below
  CUresult pending_async = CUDA_SUCCESS;
};
std::unordered_map<uint32_t, ClientThread> threads;   // under session->mu
```

The two structures answer different questions and must not be merged. The
inventory answers *what does this session own and owe back*; the thread table
answers *what was thread T looking at*. Nothing in the thread table is owned
by it. Every `CUcontext` in there is a borrowed handle to something the
inventory already accounts for, and the single most important rule of this
design is that **session expiry must not release anything from the thread
table**. Clearing it is a `.clear()` and nothing more.

The two do interact, in a way that is entirely to our benefit. `note()` in
`server/inventory.cpp` records each resource against `current_ctx()` - what
the driver says is current at the moment of creation. Today that can be the
wrong context, because the wrong context is current. Once the server restores
the issuing thread's context before dispatch, `note()` records the right one
with no change to `server/inventory.cpp` at all, and `release_inventory`'s
per-context sync-then-destroy ordering becomes correct rather than
accidentally correct. That is the strongest argument for restoring *before*
dispatch rather than doing anything cleverer.

At expiry, in `serve_session` (`server/main.cpp:452`): clear
`session->threads` first, then call `release_inventory`. First because
`release_inventory` moves the current context around freely through its
`Current` helper, so any slot surviving into that window would be stale; and
because clearing first makes it structurally impossible for someone to later
"helpfully" destroy contexts named in the table and double-free what the
inventory owns.

### Contexts that die under another thread

A context destroyed by client thread B leaves thread A's slot naming freed
driver state. Two defences, both needed:

- **Sweep on destroy.** When `cuCtxDestroy_v2` succeeds, and when
  `cuDevicePrimaryCtxReset` succeeds, null every slot naming that context and
  strike it from every stack. The inventory already knows the primary context
  handle per device (`Inventory::primary_ctx`), which is what makes the reset
  case findable.
- **Fail cleanly on restore.** If `cuCtxSetCurrent` rejects the saved context
  anyway, null the slot and return `CUDA_ERROR_INVALID_CONTEXT` for that call.
  That is what a real driver does to a thread whose current context was
  destroyed out from under it, so it is the right answer as well as the safe
  one.

## Applying the context per call

At the top of the request loop in `serve()`, before dispatch:

```
ClientThread& t = slot_for(h.thread_id);
if (t.current != applied) { cuCtxSetCurrent(t.current); applied = t.current; }
```

`applied` is a plain local in `serve()` - it is the serving thread's own state
and nothing else touches it.

**Null is applied like any other value.** A brand-new client thread that has
selected nothing gets a null current context, and a context-requiring call
correctly fails with `CUDA_ERROR_INVALID_CONTEXT` instead of quietly
inheriting whatever the previous thread left behind. Calls that genuinely need
no context - `cuInit`, `cuDeviceGet`, `cuDeviceGetCount`,
`cuDevicePrimaryCtxRetain` - work fine with a null current context. So there
is no exception list to maintain, which is worth more than the handful of
`cuCtxSetCurrent(nullptr)` calls it costs.

**When the restore is skipped.** A single-threaded client hits the `t.current
== applied` branch on every call after the first, so it pays no
`cuCtxSetCurrent` at all. So does a run of consecutive calls from the same
thread, which is what a multithreaded client's traffic mostly looks like even
when several threads are active - threads issue in bursts, not in strict
alternation. Only genuine ping-pong between threads on different contexts pays
one `cuCtxSetCurrent` per call.

### Learning what the current context became

After the call returns, read it back:

```
t.current = cuCtxGetCurrent();
```

This is the part worth defending. The alternative is to intercept the calls
that change currency and update the slot from what they were asked to do. That
requires an exhaustive list, and it is wrong the moment CUDA adds an entry
point that changes currency as a side effect - `cuCtxCreate_v4` and the green
context conversions are recent reminders that the list is not closed. Reading
it back asks the driver what actually happened. It is right by construction
and it costs one thread-local read.

Readback is not sufficient on its own, because `cuCtxGetCurrent` reports only
the top of the stack, not the depth. So the two stack calls are intercepted as
well - `cuCtxPushCurrent_v2` pushes the pre-call `current` onto `t.stack`,
`cuCtxPopCurrent_v2` pops it - and the readback then confirms the top. That is
a closed set of two, not an open-ended list.

The interception goes where the inventory's already does: a wrapper returned
by `driver_wrapper()` (`server/inventory.h`), bound to the serving thread's
current slot the same way `inventory_bind` binds the inventory. Nothing in the
generated dispatch learns about any of this, which is the property that keeps
`codegen/emit.py` out of the change entirely.

### Thread state the readback does not cover

Stream capture mode is per-thread driver state and `cuCtxGetCurrent` says
nothing about it. `handle_capture_mode` (`server/main.cpp:188`) is already
hand-written, so recording the mode in the slot and re-applying it alongside
the context is a few lines in a function we already own. It is the same bug as
issue #2 wearing different clothes, and it should be fixed in the same change
rather than filed for later.

Beyond that: we handle what we can enumerate. See Risks.

## One server thread per session, or one per client thread?

**One per session. Unchanged.** The per-thread state moves into the session;
the execution does not.

A thread per client thread is genuinely attractive - it would give real
concurrency, and CUDA's thread state would live where the driver naturally
keeps it, with no save, no restore and no readback. It also breaks three
things that currently work.

**Ordering.** `req_id` comes from a single counter minted under the client's
global mutex, and both sides read "up to id N" as "everything before N is
done". `forget_acked_locked` (`client/rpc.cpp:135`) drops every unacked frame
at or below the id it was told about. With N serving threads, request 7 can
finish before request 5, and the client would drop frame 5 from its replay
buffer while frame 5 was still in flight. The acknowledgement scheme is not
adaptable to this without becoming per-thread, which means N counters, N
windows, and a handshake that reconciles all of them.

**The cached reply.** `session->last_reply` is one slot - the reply that may
have been lost when the connection died (`server/main.cpp:252`). With N
serving threads there are N last replies and no way for the client to say
which one it missed with a single `last_req_id` in its handshake.

**It buys nothing yet.** The client is already serialized: `call()` takes
`g_mu`, writes, and blocks reading the reply (`client/rpc.cpp:369`). One
socket, one outstanding request. Fanning out on the server without also
fanning out the client just moves the queue. Fanning out the client means
per-thread connections, per-thread request id spaces and a per-thread replay
window - a much larger change than issue #2, and one that should be justified
by a measurement rather than by this bug.

So, said plainly, what the choice preserves and what it costs:

- Total order across a session: **preserved**, exactly as today.
- `session->last_req` still meaning "everything up to here completed":
  **preserved**.
- Replay after a dropped connection: **preserved, for free.** `SentFrame`
  holds whole frames as bytes, so a replayed request carries the same
  `thread_id` it was sent with and lands under the same context. Not one line
  of the replay path changes.
- The single cached reply: **preserved**.
- The slot table surviving a reconnect: **yes**, because it hangs off the
  `Session`, which is what already survives.
- Concurrency: **still none.** Two client threads on two GPUs will be correct
  and serialized. The server should say so once, the first time a session's
  slot count exceeds one, so nobody diagnoses this as a mystery.

### Deferred errors become per-thread

Making `pending_async` a field of `ClientThread` rather than of `Session` is a
one-line move that the thread id makes possible, and it belongs in this
change. Today thread A's failed no-reply launch is handed to whichever thread
replies next (`server/main.cpp:383`), so B reports a failure it did not cause
and A never learns. CUDA's own model is a sticky error per context and a last
error per thread - which the shim already reproduces on the client side with
`t_last_error` (`client/cudart_impl.cpp:36`). Per-slot is what makes those two
halves agree.

Existing single-threaded tests are unaffected: with one slot the behaviour is
identical, which `reconnect_smoke` - single-threaded, and asserting a deferred
`CUDA_ERROR_INVALID_HANDLE` at `tests/reconnect_smoke.cpp:80` - will confirm
without modification.

## The client side

`t_ctx_set` **stays**, and for the first time it tells the truth.

Its meaning today is "this client thread has made its context current", which
is a claim about a piece of state that another thread can change. Once the
server keeps currency per client thread, it becomes a claim about *this
thread's slot*, which nothing else can touch. Caching it is then sound, and it
is worth keeping: it is what stops every runtime call from carrying a redundant
`cuCtxSetCurrent` round trip.

What must invalidate it:

- `cudaSetDevice` to a different device - already handled
  (`client/cudart_impl.cpp:375`).
- `cudaDeviceReset` - handled *badly* today. It clears `t_ctx_set` on the
  calling thread only, while releasing every primary context process-wide
  (`g_primary.clear()`, `:480`), leaving every other thread believing in a
  context that has been released. Replace the boolean with a generation
  counter: `t_ctx_gen` compared against a global `g_ctx_gen` that
  `cudaDeviceReset` bumps. Then the invalidation is process-wide and costs one
  integer compare. The server-side sweep turns any race here into a clean
  `CUDA_ERROR_INVALID_CONTEXT` rather than a wrong answer.
- A reconnect: **nothing**. A resumed session keeps its slots, so the cached
  fact is still true. A session that could not be resumed already gives up
  entirely (`client/rpc.cpp:219`).

`client/shim.cpp` needs no change at all. The `cuCtx*` calls stay fully
remoted; they now simply land on the calling thread's slot. The whole client
change is in `rpc.cpp`: mint the id, stamp it into `queue_frame_locked`, and
retire it.

Read the id before taking `g_mu` and pass it into `queue_frame_locked` as a
parameter. It is thread-local so it cannot differ either way, but passing it
documents that the id belongs to the calling thread rather than to whoever
eventually flushes the queue - which is precisely the distinction that makes
cross-thread batching correct. Thread A's queued no-reply frames each carry
A's id, so when B's `call()` flushes them the server puts each one back under
A's context before running it. The batching instance of the bug closes for
free, and it closes because identity is per frame rather than per connection
or per flush.

## Compatibility

A clean refusal at the handshake, in both directions. No fallback, no
downgrade path.

- **Old client (v2) to new server (v3):** the server sees the version
  mismatch, logs it and closes (`server/main.cpp:467`). Already correct.
- **New client (v3) to old server (v2):** the old server closes without
  replying, so the client's `read_exact` fails and it reports "handshake
  failed" rather than a version mismatch. Correct behaviour, poor message, and
  not fixable retroactively. Making the *new* server reply-then-close means
  every mismatch from v3 onward reports precisely.

Both sides then set `g_connect_failed` and the client refuses to proceed,
which is the right outcome: a v2 server cannot honour the contract, and
running against it anyway is exactly the silent wrong-device bug we are here
to remove.

## Testing without a GPU

The fake driver is why this bug was invisible. `tests/fake_cuda.cpp` keeps a
`t_current` (`:67`) and, since the inventory work, tags each allocation with
the context it was made in (`Alloc::ctx`, `:27`) - but that tag exists only so
a device reset knows what to throw away. **No operation checks it.** `range_ok`
(`:73`) looks at the size and ignores the context, so a memcpy into a pointer
allocated under a different context succeeds happily, which is exactly why
issue #2 is a source-level finding rather than a red test.

Three things the fake has to learn, in increasing order of value:

1. **More than one device.** `cuDeviceGetCount` returns a hard 1 (`:97`) and
   `cuDeviceGet` rejects any other ordinal. Make the count come from
   `RGPU_FAKE_DEVICES`, defaulting to 1 so no existing test changes behaviour,
   and set it to 2 in the new test's environment.
2. **A distinct primary context per device.** `primary_token()` (`:71`) is a
   single constant, and its own comment already claims one per device. Make it
   `0xC0FFEE01 + dev` and make good on the comment, so a test can tell which
   context a call ran under at all. `g_primary_retains` is already keyed by
   device, so the bookkeeping is in place.
3. **Enforce the tag.** This is the substance, and it is a small change on top
   of what already exists: `range_ok` also compares `t_current` against
   `Alloc::ctx` and returns `CUDA_ERROR_INVALID_CONTEXT` on a mismatch, and
   modules, streams and events get the same treatment. That is the isolation a
   real driver enforces, and it is what converts issue #2 from an invisible
   wrong answer into a loud failure.

Optionally, a `fake_stats` counter for `cuCtxSetCurrent` calls, so the
single-threaded no-extra-switches claim is guarded by a test rather than by
this paragraph.

### The tests that would have caught the original bug

A new `tests/threadctx_smoke.cpp`, run from `tests/run_smoke.sh` with
`RGPU_FAKE_DEVICES=2` and `RGPU_BATCH=1` pinned the way `reconnect_smoke`
pins it:

1. **The issue's scenario, exactly.** Thread A: `cudaSetDevice(0)`,
   `cudaMalloc`. Barrier. Thread B: `cudaSetDevice(1)`, `cudaMalloc`. Barrier.
   Thread A: `cudaMemcpy` into its own pointer. With the fake tagging
   resources this fails today and passes after the fix. **This is the test.**
2. **Currency readback.** Thread A `cuCtxSetCurrent(cA)`, barrier, thread B
   `cuCtxSetCurrent(cB)`, barrier, thread A `cuCtxGetCurrent` must return
   `cA`. This one needs nothing at all from the fake and is the smallest
   possible regression test for the whole design - it should be written first
   and watched to fail.
3. **Stack depth.** Thread A pushes `cA`; thread B pushes `cB` and pops;
   thread A pops and must get back what it started with. Guards the half of
   the slot that readback alone cannot.
4. **Batching across threads.** Thread A issues a run of no-reply calls under
   context 0 without flushing; thread B makes a synchronous call under context
   1, which flushes A's batch. A's work must have run under context 0. Cases
   1-3 all pass with a flush-order mistake that this one catches.

Extensions to tests that already exist:

- **`tests/reconnect_smoke.cpp`**: two threads with different contexts, the
  connection dropped mid-batch via `RGPU_DROP_AFTER`, and after the replay
  each thread's context is still its own. Proves the thread id survives replay
  and that the slot lives on the session rather than the connection.
- **`tests/expiry_smoke.cpp`**: the same assertions with two client threads on
  two devices - both devices' primary retains must come back to zero. Proves
  the slot table does not obstruct `release_inventory`, which is the
  interaction most likely to be got wrong.
- **`tests/test_wire.cpp`**: `sizeof(ReqHeader) == 24`, the field offsets, and
  `kProtocolVersion == 3`. That file already owns the wire-layout facts.
- **`tests/hostile_smoke.cpp`**: a v2 handshake against the v3 server is
  refused rather than half-parsed; and a frame with `thread_id == 0` is
  rejected.

## Order of work

Each step leaves the tree green.

1. Fake driver: multi-device, per-device primary contexts, context-tagged
   resources. Land alone. Any red it produces in existing tests is real and
   gets triaged before anything else moves.
2. Test 2 (currency readback), written to fail.
3. Wire: the header field, version 3, `test_wire` assertions, reply-then-close
   on the server's version mismatch.
4. Client: mint, stamp, retire; `t_ctx_gen` in place of `t_ctx_set`'s boolean.
5. Server: the slot table, restore-before-dispatch, readback-after,
   push/pop interception, the destroy sweep, per-slot `pending_async` and
   capture mode.
6. Tests 1, 3, 4, and the reconnect and expiry extensions.

## Risks

**What it costs the single-threaded case.** One `cuCtxGetCurrent` and one hash
lookup per request, and four bytes per frame. No `cuCtxSetCurrent` at all,
because the skip covers it. Against a request that has already paid a socket
read - and on a real deployment, a network round trip - this should be
unmeasurable, but "should be" is not a measurement. Run `tests/bench.cpp`
against `rgpu-server-fake` before and after; that harness exists precisely to
price protocol changes without renting a GPU. If the readback does show up,
the fallback is to do it only after calls from a name list, trading the
robustness argued for above for a few nanoseconds. Take the measurement before
taking the trade.

**Thread state we have not enumerated.** Currency is covered exhaustively by
the readback. The context stack and the capture mode are covered by
interception. Anything else CUDA keeps per thread is not covered, and we do
not have a complete list of what that is. This is the design's honest weak
point, and it is where a narrow refusal earns its keep: the server should log
once, loudly, when a session first has more than one client thread, so that
any residual weirdness in a multithreaded client is attributed to a known
limitation rather than hunted for weeks. Refusal is also right for a client
that migrates CUDA work between OS threads - that is not CUDA's contract, we
key on the OS thread, and we would silently give it the wrong slot. We cannot
detect it, which is worth writing down.

**The slot table is client-controlled memory.** The protocol has no
authentication (`server/main.cpp:586`), so a client can mint ids without
bound. Cap the live slots per session - a few thousand, configurable - and
refuse past it. This is the same discipline `handle_graph_nodes` already
applies to a client-supplied capacity (`server/main.cpp:160`).

**A stricter fake will surface unrelated sloppiness.** Enforcing the context
tag is the point, and it will also find any place the existing tests are
careless about currency - `expiry_smoke` and the inventory tests being the
most likely, since they are the ones that move contexts around. That is why
step 1 lands alone, and why it lands after the inventory work rather than
alongside it.

**Per-thread deferred errors change observable behaviour.** An error caused by
thread A no longer surfaces on thread B. That is more correct and it is what
CUDA does, but it is a behaviour change, and anything that happened to depend
on the old leak will notice. Nothing single-threaded can.

**Concurrency expectations.** The most likely way this design disappoints is
that someone reads "per-thread contexts" as "per-thread parallelism". It is
not. The doc says so in Goals, the server should say so in its log, and the
follow-up issue for per-thread connections should be filed at the same time as
this lands, so the gap is on the record rather than in someone's head.
