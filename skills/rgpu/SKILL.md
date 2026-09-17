---
name: rgpu
description: Use, configure, test, or troubleshoot rGPU remote GPU execution through its PyTorch device or CUDA compatibility shim. Apply when running workloads with rGPU, choosing between its two backends, operating its server and client, or validating an rGPU setup.
---

# Use rGPU

rGPU has two independent paths. Choose one before changing setup or code:

| Path | Choose it when | Server | Client setting |
| --- | --- | --- | --- |
| Python operation backend | Python code can use `device="rgpu"`; this is the direct macOS path | `rgpu-opserver`, port 9720 | `RGPU_OPSERVER` |
| CUDA compatibility shim | An existing Linux CUDA program must keep using `device="cuda"` | `rgpu-server`, port 9713 | `RGPU_SERVER` |

Do not mix their ports, environment variables, or servers. JAX work in this
repository is experimental and is not a supported third path.

## Work from the current checkout

Use the repository's maintained documentation rather than copying commands
from old plans:

- `website/content/docs/quickstart.mdx` for the Python path.
- `website/content/docs/cuda-shim.mdx` for the CUDA path.
- `website/content/docs/configuration.mdx` for environment variables.
- `website/content/docs/operations.mdx` for deployment and isolation.
- `scripts/README.md` for the available helpers.

If these files are unavailable, ask where rGPU is checked out instead of
guessing version-specific installation commands.

## Python operation backend

Install the `python/` package on client and server. Their PyTorch major.minor
versions must match. Start the remote side with `rgpu-opserver`, then use
`rgpu-run --host user@gpu-host python program.py` or an existing SSH tunnel.
The program must import `rgpu` and explicitly place tensors on `device="rgpu"`.

For a GPU-free protocol smoke test, run `rgpu-opserver --device cpu` locally.
Treat that as transport validation only; it does not prove CUDA behavior or
performance.

## CUDA compatibility shim

Run `./scripts/build_client.sh`; it fetches missing headers, regenerates stale
sources, builds the client shims, and runs the GPU-free C++ tests. Do not tell
users to run its internal fetch and code-generation steps separately unless
they are debugging those stages.

The CUDA client must be Linux. Build and run `rgpu-server` on the GPU host,
then launch the client with the installed `rgpu` wrapper or the preload setup
documented in the CUDA shim guide.

## Safety and verification

Neither server protocol authenticates or encrypts connections. Keep the
server on localhost and use SSH or a private transport. Never expose ports
9713 or 9720 directly to the public network.

Distinguish these checks when reporting results:

- CPU opserver or fake-driver tests validate protocol behavior without a GPU.
- A CUDA-enabled server plus a real workload validates remote GPU execution.
- Benchmark claims require the measured host, network, workload, and batching
  settings; use `RGPU_STATS` for CUDA shim traffic when useful.

Do not start billable cloud resources, publish changes, or delete remote
resources unless the user has authorized that action.
