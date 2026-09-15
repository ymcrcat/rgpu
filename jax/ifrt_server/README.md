# The missing IFRT proxy server

`jax/pjrt-precondition.md` establishes that jaxlib already ships the IFRT proxy
**client** (`jaxlib/_ifrt_proxy.so`, exposing `get_client`), and that
`jax.extend.backend` re-exports it as a supported extension point. If the
matching server runs on a GPU host, a Mac running stock JAX gets a remote GPU
through ordinary `jax.Array` placement — no hand-written PJRT plugin.

OpenXLA ships that server as **libraries only**. `xla/python/ifrt_proxy/server`
has `grpc_server.{h,cc}` and `ifrt_backend.{h,cc}`, its `BUILD` declares eight
`cc_library` targets and no `cc_binary`, and a code search for
`grpc_server_main` across the whole repository returns nothing. So the server
exists and is maintained, but nobody ships a runnable one.

`rgpu_ifrt_server.cc` is that missing `main()`. It is ~40 lines around
`GrpcServer::CreateFromIfrtClientFactory`, modelled on
`xla/python/ifrt_proxy/integration_tests/scoped_pjrt_cpu_via_proxy.cc`, the only
in-tree example of standing the server up over a PJRT client.

## Status

**Not yet built or run.** Our `main()` has never reached the compiler: three
build attempts all failed earlier, inside gRPC, on XLA's own target. Treat the
include paths and the GPU client entry point as unverified.

### Build attempt, 2026-09-15

Rented A40 box, 96 cores, 200 GB disk, XLA HEAD `1bfc689`, bazel 8.7.0 via
bazelisk, `configure.py --backend=CPU`.

**A full XLA build takes about 15 minutes here, not hours.** The first attempt
ran 905 s across 6,419 actions before failing. Retries with a warm cache were
105 s and 38 s. That makes iteration cost roughly two minutes and a few cents,
so an earlier "budget hours" warning was wrong by an order of magnitude — on a
many-core box. Note RunPod's API reported `vcpuCount: 9` for a machine where
`nproc` says 96; trust `nproc`.

All three failures were the same upstream target, and none involved our code:

```
external/grpc+/src/core/BUILD:11471:16
  Compiling src/core/channelz/v2tov1/property_list.cc failed:
  undeclared inclusion(s) in rule '@@grpc+//src/core:channelz_v2tov1_property_list'
```

`property_list.cc` includes `google/protobuf/{any,duration,timestamp}.upb.h`
without the target declaring them — a dependency-hygiene bug in the gRPC
revision XLA currently pins.

What was tried:

| flag | effect |
|---|---|
| none | fails the modules `layering_check` |
| `--features=-layering_check` | gets past that, then fails Bazel's undeclared-inclusion check on the same file |
| `--features=-strict_header_check` | **no effect** — not a feature this toolchain (`rules_ml_toolchain`) defines, so it is silently ignored |

### The revision is now pinned

`build_on_host.sh` pins XLA to **`dcf304bc5dca1932b99f740b911dbd73631a1a69`**,
which is what jaxlib 0.11.1 was built from — read out of
`third_party/xla/revision.bzl` at the `jax-v0.11.1` tag.

That pin does two jobs:

- **Protocol.** The IFRT proxy performs a version handshake, so a server built
  from a different XLA than the client's jaxlib may refuse the connection even
  when it compiles. Matching the revision removes that whole class of problem.
- **The build break.** `dcf304bc` is dated 2026-08-17, a month before the HEAD
  that failed (`1bfc689`, 2026-09-15), so it predates the gRPC breakage above.
  Not yet confirmed by a build.

To follow a different jaxlib, read the commit out of the matching jax tag
rather than guessing:

```sh
gh api repos/jax-ml/jax/contents/third_party/xla/revision.bzl?ref=jax-vX.Y.Z \
  -q .content | base64 -d | grep XLA_COMMIT
```

The alternative — patching gRPC to declare the upb headers it already includes
— means carrying a patch against an external repository, which is worse than
pinning and does nothing for the handshake.

## Building it

It has to be built inside an OpenXLA checkout, because it depends on XLA
targets that are not exposed by any released package.

Copy `build_on_host.sh` to the build machine and run it there. It installs
bazelisk, fetches the pinned XLA revision, drops our source into the tree,
configures, and builds upstream's target before ours:

```sh
bash build_on_host.sh              # CPU backend, the cheap proof
BACKEND=CUDA bash build_on_host.sh
```

About fifteen minutes cold on a 96-core box, under two minutes warm, and tens
of gigabytes of disk.

**Do the CPU backend first.** `--backend=cpu` needs no CUDA in the build at
all, and proves the part that is actually in question: that a Mac client can
drive a remote IFRT proxy server transparently. Adding CUDA afterwards is a
build concern, not a design one.

## Running it

```sh
# on the GPU host
./rgpu_ifrt_server --port=12345 --backend=gpu
# from the client
ssh -N -L 12345:127.0.0.1:12345 user@gpuhost
IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true python your_script.py
```

The client needs `grpc://127.0.0.1:12345` and that environment variable set to
exactly lowercase `true`. `1`, `TRUE`, `True` and `yes` are silently ignored and
leave ALTS credentials in force, which fails every connection with
`INVALID_ARGUMENT: Invalid credentials` and no hint that a setting was dropped.

No authentication, as everywhere else in rgpu: bind localhost, tunnel in.
Anyone who can reach the port can run arbitrary computations on the GPU and
read its memory.

## Why this is still the cheaper path

Writing a PJRT C-API plugin means implementing device discovery, compilation,
buffers, execution, transfers, completion events and deletion, then keeping up
with the C API. This is a `main()` over an implementation Google maintains and
tests. The cost here is a build, not a design.
