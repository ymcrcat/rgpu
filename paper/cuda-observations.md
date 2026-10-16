# CUDA observations supplied for the October 15, 2026 paper revision

This is a preserved maintainer summary supplied with the revision request, not a raw benchmark log or a new run. The source review covered main at `ced62f546e7c3d6c63a3b7b5ead7caaedc05b2c9`, after the paper branch base `58a0fa5`. Exact per-run timestamps, commands, build hashes, trial counts, seeds (except the separately documented launcher check), native graph settings, and raw timings were not supplied. Ranges below are reported ranges, not confidence intervals.

## Supplied observations

- llama.cpp and Ollama run unchanged through the CUDA shim with output identical to native in the tested cases. Stock Ollama 0.40.3 uses `rgpu ollama serve`; `GGML_CUDA_DISABLE_GRAPHS=1` is required.
- GPU detection on a 123 ms link fell from 117 round trips / 28.3 s to 20 round trips / 3.3 s. Device properties are cached and fetched with one bulk attribute call; occupancy is cached per kernel.
- Round trips per generated token fell from 45 to 21 with the occupancy cache, then to one with upload-only queued stream synchronization. `RGPU_LAZY_SYNC` is tri-state.
- Sending frame header and payload together removed an SSH-tunnel behavior that made each reply cost two round trips.
- Server frame reads now use a 64 KiB buffer. Previously there were 319,028 receive syscalls for 320 tokens, about 1,000 per token. Client queue flushes now occur every 1 KiB, controlled by `RGPU_FLUSH_BYTES`.

A40 pod, `qwen2.5:0.5b`, 320 generated tokens, client and server on the same machine:

| Application/configuration | Tokens/s |
| --- | --- |
| llama.cpp server, rGPU, one CPU thread, before final read/flush changes | 105-121 |
| llama.cpp server, rGPU, one CPU thread, after final read/flush changes | 210-217 |
| llama.cpp server, native, one CPU thread | 379-390 |
| Ollama native, default threads | about 160 |
| Ollama native, `num_thread` 1 | 437-480 |
| Ollama rGPU, default threads | about 49 |
| Ollama rGPU, `num_thread` 1 | 174-226 |

- The container showed 96 cores; the default runner used 48 busy-waiting threads. Its allowance was 8.5 cores (`cpu.max`: `850000 100000`), and `cpu.stat` showed throttling. `-t 1` / `--poll 0` address the slowdown; `OMP_WAIT_POLICY=passive` did nothing. No separate throughput for `--poll 0` was supplied.
- WAN, Mac arm64 container to x86 pod, 123 ms RTT, 1.5 MiB/s upload: about 3 tokens/s, link-bound, with 0.58 MiB logits downloaded per token. Loading 374 MiB of weights took about seven minutes. No download-throughput measurement was supplied.
- Fake-driver microbenchmark `tests/token_bench.cpp`: launch issue about 0.21 microseconds per call. The later 1 KiB flush adds about 0.25 microseconds per launch where there is no GPU time to overlap. Neither figure is completed GPU work.
- Real-GPU `tests/hw/cublas_check.cu` and `cudnn_check.cu` pass through the shims via `tests/hw/math_check.sh`.

## Flush-threshold comparison

Measured with the 64 KiB server read buffer in place, llama.cpp server, one CPU thread, same A40 pod, `RGPU_FLUSH_BYTES` set per run. One session alternated 1 KiB and 256 KiB: 216.8 and 210.2 tok/s at 1 KiB, 123.9 and 131.1 at 256 KiB. An earlier sweep in the same pod (two rounds) gave 120.7-124.0 at 256 KiB, 134.1-138.3 at 64 KiB, 154.3-159.9 at 16 KiB, 154.8-160.3 at 4 KiB, 170.8-194.0 at 1 KiB and 186.2-198.3 with a send per call. Output was identical in every run.

## Repository corroboration and scope differences

- `8453dbd` records detection counts/times and the header/payload fix.
- `7054c10` records the receive count, buffer and flush sizes, llama.cpp before/after rates, unchanged output, and fake-driver flush penalty. It explicitly says the new policy has not been run on a real network link or with PyTorch. The WAN result therefore must not be attributed to this final transport revision.
- `website/content/docs/ollama.mdx` at `ced62f5` records native Ollama 161 tokens/s by default and 437-450 with one thread, versus 49 and 174-226 through rGPU. The supplied native range is broader (437-480). Preserve the distinction; do not invent additional samples or merge these into a trial distribution.
- That document verifies the stock launcher on Linux x86_64 / RTX 4090 with Ollama 0.40.3, temperature 0, seed 1, and 34 identical tokens. It separately records a direct llama.cpp check with 320 identical tokens at temperature 0. These correctness checks are not the A40 timing dataset.
- The WAN observations used an earlier copied Ollama installation, not an end-to-end validation of the stock launcher on arm64. `OLLAMA_LOAD_TIMEOUT=30m` is the documented slow-link setting.
- `website/content/docs/cuda-shim.mdx` records the real-GPU math checks; their script compares both native and shimmed execution with CPU references. No fresh hardware check was run for this revision.

## TODO

Archive raw CUDA timing samples, run commands, exact client/server builds and library versions, native graph settings, and hardware/load records. Validate the final read/flush policy over WAN and with PyTorch and the stock launcher on arm64; measure larger models and separate buffer/flush ablations. The paper leaves these limitations explicit rather than extrapolating new results.
