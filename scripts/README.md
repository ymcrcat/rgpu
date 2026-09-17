# Script catalog

Run these commands from the repository root unless a script says otherwise.

## Build and install

| Script | Purpose |
| --- | --- |
| `build_client.sh` | Fetch missing CUDA headers, regenerate stale sources, build the Linux client shims, and run C++ tests |
| `build_macos.sh` | Build the driver API shim on macOS; add `--test` for GPU-free tests |
| `fetch_headers.sh` | Fetch the CUDA headers used by code generation and local builds |
| `fetch_cudart.sh` | Fetch a stock CUDA Runtime library for compatibility testing |
| `install.sh` | Install built client libraries and the `rgpu` launcher |
| `install_agent_skill.sh` | Install the rGPU guidance skill for Claude Code and Codex |

## GPU server and hardware checks

| Script | Purpose |
| --- | --- |
| `deploy_server.sh` | Copy the checkout to an SSH host and build the CUDA server there |
| `deploy_opserver.sh` | Copy the Python backend to an SSH host and start `rgpu-opserver` there |
| `build_fatbin.sh` | Build the test CUDA kernel on a machine with `nvcc` |
| `hw_check.sh` | Run driver probes and end-to-end checks on a GPU host |
| `remote_torch.sh` | Run the PyTorch compatibility ladder on the remote host |
| `opserver_pod.sh` | Install and start the Python operation server on a GPU pod |

## Client workload

| Script | Purpose |
| --- | --- |
| `run_torch.sh` | Run stock CUDA PyTorch in the Linux client container against an rGPU server |

## Cloud hosts

| Script | Purpose |
| --- | --- |
| `runpod.sh` | Create, inspect, start, stop, or delete the test RunPod host |
| `gcp_up.sh` | Create or start the configured Google Cloud GPU VM |
| `gcp_down.sh` | Stop the Google Cloud VM, or delete it with `--delete` |

`runpod.sh create` and `gcp_up.sh` can start billable resources. The matching
stop/delete commands release them when testing is complete.
