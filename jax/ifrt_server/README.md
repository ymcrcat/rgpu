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

### What to try next

**Pin XLA to the revision jaxlib 0.11.1 was built from, rather than HEAD.**
This is worth doing for its own sake regardless of the build break: the IFRT
proxy performs a version handshake between client and server, so a server built
from HEAD may well refuse to talk to a 0.11.1 client even if it compiles. A
matching revision fixes both problems at once, and is very likely to predate
this gRPC breakage.

Failing that, the surgical fix is patching the gRPC target to declare the upb
deps it already includes, which means carrying a patch against an external
repository — worse than pinning.

## Building it

It has to be built inside an OpenXLA checkout, because it depends on XLA
targets that are not exposed by any released package.

```sh
git clone https://github.com/openxla/xla.git
mkdir -p xla/xla/python/ifrt_proxy/rgpu
cp rgpu_ifrt_server.cc BUILD xla/xla/python/ifrt_proxy/rgpu/
cd xla
./configure.py --backend=CUDA      # or --backend=CPU for the cheap first test
bazel build -c opt //xla/python/ifrt_proxy/rgpu:rgpu_ifrt_server
```

Budget hours, not minutes, and tens of gigabytes. A full XLA build with CUDA on
a 9-vCPU box is the dominant cost of this whole approach.

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
