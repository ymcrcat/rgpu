# rGPU

[![CI](https://github.com/ymcrcat/rgpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ymcrcat/rgpu/actions/workflows/ci.yml)

rGPU runs GPU work on a remote NVIDIA machine while the application stays on
the client. It currently offers two paths:

| Path | Use it for | Interface |
| --- | --- | --- |
| PyTorch device | PyTorch programs that can opt into an `rgpu` device | `torch` operations over TCP |
| CUDA shim | Existing Linux CUDA programs, including stock CUDA PyTorch | `libcuda`, CUDA Runtime, cuBLAS, cuBLASLt, and cuDNN shims |

The PyTorch device is the simpler integration. The CUDA shim covers existing
binaries but has a larger compatibility surface.

## Documentation

The Fumadocs site in [`website/`](website/) is the product documentation:

- [Quickstart](website/content/docs/quickstart.mdx)
- [Training](website/content/docs/training.mdx)
- [nanoGPT example](website/content/docs/nanogpt.mdx)
- [CUDA shim](website/content/docs/cuda-shim.mdx)
- [Operations](website/content/docs/operations.mdx)
- [Configuration reference](website/content/docs/configuration.mdx)
- [Performance](website/content/docs/performance.mdx)
- [Troubleshooting](website/content/docs/troubleshooting.mdx)

Engineering records and experiments are indexed in [`docs/README.md`](docs/README.md).

## Quick start: PyTorch device

Follow the [quickstart](website/content/docs/quickstart.mdx) to install rGPU
and deploy the server. Save this as `smoke.py` in your workload directory:

```python
import torch
import rgpu

x = torch.ones(4, device="rgpu")
print((x * 2).sum().item())  # 8.0
```

Run it in the environment where rGPU is installed, using your server's SSH
destination and options:

```sh
rgpu-run --host user@gpu-host --ssh-port 2222 -i ~/.ssh/gpu_key \
  python smoke.py
```

The program selects the device; `rgpu-run` opens the tunnel and configures the
connection. The expected output is `8.0`.

For existing Linux CUDA programs, follow the
[CUDA shim guide](website/content/docs/cuda-shim.mdx), starting with
`./scripts/build_client.sh`.

Neither protocol authenticates or encrypts connections. Keep `rgpu-opserver`
on its default localhost bind and use SSH. The CUDA server listens on all IPv4
interfaces: restrict port 9713 with host/cloud firewall rules before starting
it, even when using an SSH tunnel. See [deployment](website/content/docs/operations.mdx).

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
| `python/` | PyTorch device and launcher |
| `tests/` | C++, Python, CUDA, and hardware checks |
| `codegen/` | CUDA header parser and source generators |
| `website/` | Fumadocs product documentation |
| `docs/` | Design records, measurements, and experiment reports |
| `jax/` | Experimental JAX work; not a supported product path |
| `scripts/` | Build, deployment, cloud, and test helpers |
| `skills/` | Installable agent guidance for using rGPU |

Historical implementation notes and experimental results are indexed in
[`docs/README.md`](docs/README.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
