# rGPU paper artifact

- `rgpu-paper.pdf`: rendered systems-paper draft.
- `paper.md`: editable manuscript, references, and source provenance.
- `structure.md`: rationale for the structure, prior-art comparisons, and submission priorities.
- `benchmark.py`: standalone CPU loopback experiment with numerical and counter assertions.
- `results.json`: environment, summary, and all 35 raw timing samples.
- `checks.txt`: output of 40 selected existing correctness tests.
- `stretched_lines.py`: lists justified lines in the PDF whose gaps exceed three normal spaces, measured from the finished PDF. `render.py` prints the same report exactly, from the space it added to each line, and writes every justified line to `figures/justified_lines.json`; `stretched_lines.py rgpu-paper.pdf --check figures/justified_lines.json` compares the two.
- `render.py`: regenerate the three figures and PDF from the manuscript and data.
- `gpu_benchmark.py`: the Section 6.8 instrument. Runs native, colocated, and WAN configurations on one GPU for the Section 6.4 chain and a transformer training step.
- `native.json`, `colocated.json`, `wan.json`: its three legs on one L40S (September 26, 2026): 14, 35, and 35 samples (84 in total), with environment and forward-loss check records. `colocated.json` has an empty `rgpu_commit`; the server package hash, the host's `env.txt`, and the raw RTT samples are not included.
- `gpu_experiments.py`: the Section 6.9 instrument: matched optimizer baseline, pinned and delayed rGPU training with client and server CPU time, graph-shipped training, and a deterministic comparison of training state.
- `delay_proxy.py`: loopback TCP proxy that adds a fixed delay in each direction, for the RTT sweep.
- `run_followup_on_gpu_host.sh`, `run_followup_part2.sh`: run Section 6.9 on the GPU host.
- `followup/`: its outputs, one JSON per condition, plus `host.txt` (hashes, CPU quota, core assignment, load) and `rtt_wan.txt` (raw round-trip samples).
- `review-astra.md`: external review of the September 26 draft.
- `run_on_gpu_host.sh`: sets up torch on the GPU host, runs the native and colocated legs, and leaves an rgpu server bound to loopback for the WAN leg.
- `dispatch_bench.py`, `dispatch.txt`: single-thread dispatch cost of ordinary CPU-tensor `torch.add` on the two client CPUs. It does not measure rGPU dispatch and does not explain the WAN-versus-colocated result; see Section 6.8.

Source inspected: `/Users/yan/Documents/LlmGames/rgpu`, commit `7a8ba43ffd46edb7efa77c93593221f813a8b712`.

This directory contains the paper and its supporting artifacts. Historical GPU observations (H) are distinguished from fresh CPU measurements (F), analytical estimates (A), and the same-GPU L40S run (G) of Section 6.8.

To rerun the benchmark, use the rGPU Python environment:

```sh
PYTHONPATH=/path/to/rgpu/python python benchmark.py > results.json
```

To regenerate the document, use Python with `reportlab`, `matplotlib`, `markdown`, and `pyphen` installed:

```sh
python render.py
```

`render.py` checks key reported values against `results.json` before rendering. If new measurements change those values, update the manuscript's evaluation and abstract before rendering again.

## Agent handoff

The user requested a detailed systems-conference-style paper, including a Background reviewing prior work, performance evaluation, and a structure informed by related papers. The current PDF is thirteen pages; its layout was rendered and visually checked.

Start with `paper.md` and `structure.md`. Preserve the distinction between historical GPU results (H), fresh CPU measurements (F), analytical estimates (A), and the same-GPU run (G). The historical A40-versus-MPS result does not measure remoting overhead. The rGPU implementation was not modified.

Status as of September 26, 2026 (second pass):

- `benchmark.py` was re-run unchanged. Message and wait counts reproduced exactly; timings moved by roughly 20% on a loaded host. `results.json` was deliberately left as the original samples because `render.py` asserts on them; the re-run is recorded in Appendix A of the manuscript.
- Ratios in Section 6.4 are now quoted to two significant figures, with the reason stated in the text.
- Djinn (OSDI 2026): the full text was not publicly retrievable on September 26, 2026 after four independent routes (USENIX page refuses automated retrieval; a coauthor group's publications page; a third-party OSDI 2026 index; searches restricted to arXiv, USENIX, ACM, and Semantic Scholar). The manuscript says so in Section 2.4 and reference [13]. The comparison remains outstanding and cannot be done until the paper is public.
- Tessera (arXiv 2604.10180) was added to Section 2.1, Table 1, and the references as adjacent kernel-granularity work; it is positioned as a different problem, not a competitor.
- `gpu_benchmark.py` was run on one rented L40S (host: Linux, Python 3.10.12, torch 2.14.0+cu130; WAN client: macOS, Python 3.14.7, torch 2.14.0; TF32 off, driver 580.159.03). Placement order was fixed (native, colocated, WAN); each leg's seven trials share one process and session. Native chain 2.02 ms / train 698.98 ms; colocated (client on the pod's EPYC 9554) chain eager 55.74, sync 92.72, graph 7.78, train eager 6399.59, loss-read-each-step 6400.36; WAN (Apple M3 client through an ssh tunnel, 48.8 ms median RTT) chain eager 150.47, sync 1194.06, graph 56.39, train eager 3214.55, loss-read-each-step 3931.33. Counts identical across placements (601/621/41/64,941/64,951 messages; 1/21/1/1/11 waits). The forward-loss check agreed with the CPU reference within 1e-4 on all three legs (not bitwise). Native used the default AdamW path (multi-tensor `foreach`, per the PyTorch source) while rGPU parameters take the single-tensor path, so the 9.2x and 4.6x are user-visible slowdowns, not isolated transport cost. WAN training was 2.0x faster than colocated; this is consistent with client CPU effects (the M3 runs ordinary `torch.add` 2.3x faster, `dispatch.txt`), but client speed, server contention on the shared host, runtime differences, and network cost have not been separated.
- Not measured: LAN condition, RTT sweep on one client CPU, larger batches, graph shipping for training, CUDA path on GPU, a matched `foreach=False, fused=False` native baseline, gradient/update correctness, a matched two-boundary comparison, and a coarse-RPC baseline. Sections 7.2 and 7.3 list them.
- `render.py` draws `figures/same_gpu.png` from the three JSONs; it does not assert on them (only the CPU experiment's values are hard-checked).
- Rendering needs `reportlab`, `matplotlib`, and `markdown`; they are now installed in the repository's Python venv, which is the interpreter `render.py` was last run with.

Status as of September 27, 2026 (third pass, after external review):

- The draft was revised against `review-astra.md`: causal claims narrowed, optimizer mismatch and numerical-check scope stated, implementation descriptions qualified, provenance gaps disclosed, related work extended.
- A second L40S session (Section 6.9) ran the experiments the review asked for. Client and server pinned to disjoint cores; added round trip 0/10/50/116 ms through `delay_proxy.py`; eager 6463 ms per ten steps with 6439 ms of client CPU time and 3450 ms of server CPU time; no dependence of eager time on round trip; blocked time about one round trip per wait; same-cores and unpinned placements within drift of the pinned one. Native default AdamW 702.72 ms, single-tensor 724.72 ms. Graph-shipped training 2190 ms, 20,931 messages, slower start (5950 against 1298 ms for two warmup steps). Laptop M3 through the tunnel 3159 ms eager, also CPU-bound. Five-step training state matches native to the spread between native's own optimizer paths.
- Found while running it: with the rgpu device registered, native CUDA `backward()` fails an assertion in PyTorch's autograd engine in the same process. Unexplained; the instruments use separate processes.
- The compiled-output conjugate/negative bit gap the review found was reproduced end to end and fixed in rGPU commit `c0318a3`; the second session ran at that commit.
- Section 6.6 now carries the compression conclusions from the performance record and issue #11, tied to the finding that the client is the constraint.
- Not measured, and listed in Section 7: a profile of the client's per-operation work, bandwidth and loss, larger batches, the CUDA path on a GPU, a coarse-RPC baseline, any prior system. Djinn's full text was still not retrieved.
