# rgpu: what works, what is missing, what to build next

Written 2026-09-11, after the torch.compile work. This is the honest state of
the thing: what has been measured, what has never been tried, and what would
have to be true before anyone but us relied on it.

The design and the reasoning behind the architecture live in
`docs/superpowers/specs/2026-09-07-cuda-api-remoting-design.md`. This file is
the product view.

## What works today

Stock PyTorch wheels, unmodified, on a machine with no GPU and no NVIDIA
driver, against a remote GPU.

| | state |
|---|---|
| Inference, eager | ResNet-18 logits match CPU to 1.4e-03 |
| cuBLAS, cuBLASLt | forwarded, matmul matches CPU |
| cuDNN | forwarded, 9.10 verified against 9.5 headers |
| torch.compile | all five tested modes match eager |
| CUDA graphs | capture and replay, `reduce-overhead` works |

Performance, ResNet-18 at batch 1 on an RTX A4000, client and server on the
same machine:

| | native | remoted | round trips per inference |
|---|---|---|---|
| eager | 1.89 ms | 3.97 ms | 49 |
| torch.compile | 1.50 ms | 4.51 ms | 86 |
| torch.compile, reduce-overhead | 1.18 ms | 1.18 ms | 2 |

With graphs the remoting overhead is gone rather than reduced, because an
iteration is one replay rather than several hundred calls.

Training works too: eight rungs, each checking a gradient rather than a loss,
all matching the same work on CPU. Linear and convolution backward, batch norm
in training mode, SGD and Adam, a ResNet-18 training step, a falling loss over
ten steps, and mixed precision with a gradient scaler.

### Over a real network

From a laptop in one city to a rented 3090 in another, through an ssh tunnel,
a round trip costs **34.6 ms**. ResNet-18 at batch 1:

| | per inference | round trips |
|---|---|---|
| eager | 1632 ms | 49 |
| captured as a CUDA graph | 70 ms | 2 |

Twenty-three times faster, and within a millisecond of two round trips, which
is what the round trip count predicted. The client was an arm64 container on
the laptop and the server an x86_64 host, so that pairing is confirmed on a
real link as well.

The lesson for anyone using this: **over a network, the only number that
matters is round trips per iteration.** Everything else is detail.

## The macOS path

`python/` is a second way to reach this that needs none of the above: a
PyTorch device named `rgpu`, registered from pure Python, that runs on the
stock CPU-only `torch` wheel - no CUDA libraries, no Docker, no driver on the
client at all. What used to need a C++ shim in front of the driver is now a
`pip install -e python` and an import.

Verified so far, against a CPU reference on a local server: inference,
training, mixed precision, `torch.compile`, and surviving a dropped
connection (the same reconnect story as gap 2 below, ported to this path).
156 tests pass this way.

What is still unmeasured is everything on real hardware: no run yet on an
actual remote GPU, over an actual network, from this Mac. No performance
numbers exist for this path yet, and none are recorded here until they are
measured.

## Gaps, in the order they would stop someone

### 1. It has no authentication at all

Anyone who can open a TCP connection to port 9713 can run arbitrary kernels on
that GPU, read any memory allocated on it, and load arbitrary code. There is no
authentication, no authorisation and no transport encryption.

Today this is managed by keeping the port on an SSH tunnel, which is fine for a
test rig with one user and not fine for anything shared. Before this is used by
more than one person, or over anything but a tunnel, it needs at minimum a
shared secret at connection setup and TLS on the wire. Multi-tenancy needs more
than that: one connection can already see another's device memory through the
GPU itself, so isolation means one server process per tenant, not one server
with a table of them.

### 2. A dropped connection is survivable; a dead server is not

Solved for the common case. A session is the client process and a connection
is one attempt by it to reach the server; the server keeps a session alive for
a grace period after its connection goes, and hands the next connection with
the same id back to the thread that was serving it. Nothing is torn down, so
every pointer and handle the application holds stays valid.

Demonstrated on a real link: a training run from the laptop to a rented GPU,
the ssh tunnel killed after step 4 and restored ten seconds later. The client
retried, reconnected and resumed, and every loss afterwards was bit-identical
to an uninterrupted run. The tunnel happened to die after the server had run a
request but before its reply arrived, so the server sent its cached reply
rather than running the call twice - the harder of the two cases, exercised by
accident.

Not solved, and not solvable by replaying calls: a server that actually dies
takes the device memory with it. The `record` tag can rebuild contexts and
modules, but the contents of memory that no longer exists need either
continuous shadowing or application-level checkpoints.

### 3. Training works, but nothing has trained for long

The gradients are right and the loss falls, on a model small enough to check
against CPU. What has not happened is a real run: hours rather than seconds, a
dataset rather than random tensors, and therefore a data path across the link
that nobody has measured. Gap 5 below is the part of training that is still
unknown.

### 4. Multi-GPU and NCCL

Not started, and not simply more of the same. NCCL brings its own transport and
wants direct GPU-to-GPU paths, so a collective across remoted GPUs is a
different problem from forwarding an API.

### 5. Smaller known limits

- **Threading.** One connection under one lock, and the current-context state
  is per connection rather than per thread. Correct for PyTorch, wrong for a
  client that drives several contexts from several threads.
- **`cuLaunchKernelEx`** is refused: it carries an attribute array we do not
  marshal. That is thread-block clusters, so Hopper and later.
- **Managed and zero-copy memory** are refused by design. They cannot work
  across a machine boundary, and failing clearly beats corrupting silently.
- **Thirteen array-shaped calls are refused** rather than wrong, as of this
  week's audit; see below.

## What the audit found, and what it implies

`cuGraphGetNodes` takes a count that is the caller's capacity going in and the
number written coming out. The generator could not see that. It treated the
array as a single handle, never sent the capacity, and would have reported one
node, or none, with no error at all.

Thirteen more functions had the same shape. They are refused now, loudly, by
the pair of names that gives it away (`nodes` and `numNodes`). None of them was
in a working path, so nothing regressed.

The implication is worth stating plainly: **the generator's failure mode is
silence.** It produces code that compiles and runs for shapes it does not
understand. The defence is the unimplemented-call log and tests that check
values rather than status codes, and both need to stay ahead of the generator.
Every hand-written call in this repository exists because the generator was
wrong about it, and there will be more.

## What it would take to be usable by someone else

Usable means a person who did not write this can run their own work on it
without reading the source. In rough order of what blocks that.

### Table stakes

1. **Authentication and encryption.** Today the port is the credential. A
   shared secret at connection setup and TLS on the wire, or it cannot leave a
   tunnel.
2. ~~Surviving a dropped connection.~~ Done; see gap 2. What remains is the
   server dying, which needs checkpointing rather than reconnection.
3. **An install that is one step.** Right now: build shims in a container, copy
   a server to the GPU host, set five LD_PRELOAD entries and three environment
   variables. It needs to be a client package and a server image, with one
   command that runs your script against a remote GPU.
4. **Making the fast path the default.** The gap between 1632 ms and 70 ms is
   entirely whether the work was captured into a graph. A user who does not
   know that will conclude the whole idea is too slow. At minimum this means
   documenting `torch.compile(mode="reduce-overhead")` as the supported path;
   better would be detecting an uncaptured hot loop and saying so.

### Needed before real workloads

5. **The data path.** Training against a remote GPU ships every batch across
   the link. Nothing here has measured that, and at 34 ms and whatever
   bandwidth the link has, it may well dominate everything this document
   measures. The fixes are known - stage the dataset on the GPU host, prefetch
   deeper, compress - but the measurement comes first.
6. **Multi-GPU and NCCL**, for anything distributed.
7. **Honest failure.** A missing entry point currently logs a name and returns
   "not supported", which is right for us and useless for a user. It should say
   what was unsupported, and what to do about it.
8. **Memory behaviour.** Out-of-memory on the far side should look like
   out-of-memory here, and `torch.cuda.memory_allocated` should mean something.

### Operational

9. **Observability.** The round trip counter exists behind an environment
   variable. It should be a supported report: how many round trips, where they
   went, what that costs on this link.
10. **Version negotiation.** The server resolves driver entry points lazily, so
    it tolerates a different CUDA version, but nothing checks compatibility
    explicitly or explains a mismatch.
11. **Lifecycle and cost.** A GPU left running is the expensive failure mode,
    and we have made it twice ourselves. Idle detection and shutdown belong in
    the product, not in a skill file.

### What is not on this list

Shared memory and RDMA transport. With graphs the round trips are down to two
per iteration, so a faster round trip is no longer what stands between this and
being useful.

## What to build next

In the order I would do them.

1. **Measure over a real network.** Every performance claim carries a "on
   loopback" asterisk. `tc netem` on the GPU host, or running the client from
   the laptop against a rented GPU, turns the predicted rows into measured
   ones. This also exercises the arm64-client to x86_64-server path, which has
   not been run since the first week.
2. **A workload that is not ResNet-18.** A small model at batch 1 is the
   unfavourable end for remoting, and an LLM decode loop is the pathological
   one: a few microseconds of work per launch, thousands of launches. If graphs
   hold up there, they hold up anywhere.
3. **Authentication.** A shared secret and TLS. Small, and it is what stands
   between this and anyone else using it.
4. **Training.** The cuDNN training group, then find out what else breaks.
5. **Reconnect.** Make the `record` tag mean something: replay the recorded
   calls to rebuild a session after a drop.

Lower priority than they used to be: shared-memory transport when co-located,
and RDMA. Graphs took the pressure off both. At two round trips per iteration
the latency of each one has stopped being the thing that matters.

## How to know it still works

- `./scripts/build_client.sh` builds and runs the whole GPU-free suite: wire
  format, kernel arguments, cuBLAS, cuBLASLt, cuDNN, CUDA graphs and the
  runtime API, all against a fake driver. It needs no GPU and takes a minute.
- `tests/torch/ladder.py` works up from "is there a GPU" to ResNet-18 against
  a real one, comparing every rung to a CPU reference.
- `tests/torch/compile_test.py` does the same for the five torch.compile modes.
- `.claude/skills/rcuda-runpod` has the rules for renting a GPU to run those
  two without wasting money, and the traps that have already cost us some.
