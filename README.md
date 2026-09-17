# rGPU

rGPU runs GPU work on a remote NVIDIA machine while the application stays on
the client. It currently offers two paths:

| Path | Use it for | Interface |
| --- | --- | --- |
| Python operation backend | PyTorch programs that can opt into an `rgpu` device | `torch` operations over TCP |
| CUDA compatibility shim | Existing Linux CUDA programs, including stock CUDA PyTorch | `libcuda`, CUDA Runtime, cuBLAS, cuBLASLt, and cuDNN shims |

The Python backend is the simpler integration. The CUDA shim covers existing
binaries but has a larger compatibility surface.

## Documentation

The Fumadocs site in [`website/`](website/) is the product documentation:

- [Quickstart](website/content/docs/quickstart.mdx)
- [Python backend](website/content/docs/training.mdx)
- [CUDA compatibility shim](website/content/docs/cuda-shim.mdx)
- [Operations](website/content/docs/operations.mdx)
- [Configuration reference](website/content/docs/configuration.mdx)
- [Performance](website/content/docs/performance.mdx)
- [Troubleshooting](website/content/docs/troubleshooting.mdx)

Engineering records and experiments are indexed in [`docs/README.md`](docs/README.md).

## Quick start: Python backend

Activate the workload environment and install the local package:

```bash
cd /path/to/workload
source venv/bin/activate
python -m pip install -e /path/to/rgpu/python
```

Deploy and start the server through SSH:

```bash
/path/to/rgpu/scripts/deploy_opserver.sh \
  user@gpu-host -p 2222 -i ~/.ssh/gpu_key
```

Run the local program through the managed tunnel:

```bash
rgpu-run --host user@gpu-host --ssh-port 2222 -i ~/.ssh/gpu_key \
  python train.py
```

Or select the device directly:

```python
import torch
import rgpu

x = torch.arange(8, device="rgpu")
print((x * 2).cpu())
```

## Quick start: CUDA shim

Build the generated client and its GPU-free tests:

```bash
./scripts/build_client.sh
```

Build and start `rgpu-server` on a GPU host, then expose it through an SSH
tunnel. The full setup and loader requirements are in the
[CUDA shim guide](website/content/docs/cuda-shim.mdx).

The protocol has no authentication or encryption. Bind servers to localhost
and reach them through SSH or another private transport.

## Development

```bash
# C++ client and fake-driver tests
./scripts/build_client.sh

# Python tests
python -m pip install -e './python[test]'
python -m pytest python/tests

# Static documentation
npm --prefix website ci
npm --prefix website run build
```

See [`scripts/README.md`](scripts/README.md) for the remaining build, cloud,
and hardware commands. Generated C++ is committed; its policy and regeneration
steps are in [`codegen/README.md`](codegen/README.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `client/` | CUDA client shims and transport |
| `server/` | CUDA server and dispatch |
| `common/` | Shared protocol and generated API metadata |
| `python/` | PyTorch operation backend and launcher |
| `tests/` | C++, Python, CUDA, and hardware checks |
| `codegen/` | CUDA header parser and source generators |
| `website/` | Fumadocs product documentation |
| `docs/` | Design records, measurements, and experiment reports |
| `jax/` | Experimental JAX work; not a supported product path |
| `scripts/` | Build, deployment, cloud, and test helpers |
| `skills/` | Installable agent guidance for using rGPU |

Current implementation status is recorded in
[`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md); measured performance is in
[`docs/performance-notes.md`](docs/performance-notes.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
