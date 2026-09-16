# Code generation

rGPU generates the repetitive CUDA driver dispatch and weak compatibility
stubs. Generated C++ is committed so a checkout can be reviewed and built
without regenerating it first.

## Regenerate

```bash
PLATFORM=linux/amd64 ./codegen/run.sh   # use linux/arm64 on Apple Silicon
```

The script fetches missing headers, builds the `rgpu-codegen` container, parses
the driver and runtime headers, and updates the generated sources.

## Inputs and outputs

| Path | Policy |
| --- | --- |
| `annotations.py`, `parse.py`, `emit.py`, `emit_runtime.py` | Authoritative generator source; committed |
| `third_party/cuda_include/` | Downloaded NVIDIA headers; ignored |
| `api.json`, `report.txt` | Driver API scratch output and coverage report; ignored |
| `runtime_api.json` | Parsed runtime API snapshot; committed |
| `cublas_api.json`, `cublaslt_api.json`, `cudnn_api.json` | Parsed library API snapshots; committed |
| `client/generated/`, `common/generated/`, `server/generated/`, `tests/generated/` | Generated C++ outputs; committed |

`run.sh` currently regenerates the driver and runtime surfaces. The cuBLAS,
cuBLASLt, and cuDNN JSON snapshots and their weak stubs are maintained as
paired inputs and outputs; change both together when those library surfaces are
refreshed.

After generation, review and commit the resulting source diff. CI runs the
same generator and fails if committed driver/runtime output is stale.
