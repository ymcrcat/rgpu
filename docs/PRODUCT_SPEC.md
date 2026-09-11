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

**Every one of those numbers is loopback.** Client and server were the same
box. Nothing here has been measured over a network, which is the single
largest gap in what we claim.

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

### 2. Nothing survives a dropped connection

Contexts, modules, allocations and descriptors all live in the server process
and die with the connection. There is no reconnect, no checkpoint and no
migration. A network blip in the middle of a long job loses the job.

The protocol was designed with this in mind - calls that establish durable
server state are tagged `record` in the annotations, and replaying the recorded
set reconstructs a session - but nothing uses that tag yet. It is the hook for
mrCUDA-style migration, not an implementation of it.

### 3. Training has never been run

Everything above is inference. The backward pass has not been attempted once.
Known missing pieces: the cuDNN batch-norm training group
(`BatchNormalizationForwardTrainingEx`, `BackwardEx`, the two workspace size
queries, activation descriptors). Unknown pieces: whatever else autograd
touches. This is the largest untested area, and the plan always had it after
milestone 1.

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
