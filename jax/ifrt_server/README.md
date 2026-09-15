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

**Not yet built or run.** The source is written from the upstream headers and
the reference example; it has not been compiled, so treat the exact include
paths and the GPU client entry point as unverified against a real checkout.
They move between XLA releases.

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
