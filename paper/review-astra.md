# rGPU draft review

Review date: September 26, 2026. Manuscript references below are to `paper/paper.md`; other paths are relative to the repository unless stated otherwise. H = historical observations; F = fresh CPU loopback measurements; A = analytical calculations; G = same-GPU L40S measurements. Source inspection and local probes are identified separately: they are not a fifth performance dataset.

I read the manuscript, positioning document, artifact notes, both benchmark instruments, all four JSON datasets, host script, dispatch benchmark and output, selected-test transcript, rendering script, and relevant implementation. I recomputed summaries from raw samples and ran only local, socket-free probes; I did not rerun benchmarks or tests that start servers, use a GPU, or require credentials. The current checkout is `cdc336aab778052c31c65f1a000740202699b645`; `git diff` against the paper's `7a8ba43ffd46edb7efa77c93593221f813a8b712` shows no differences under `python/rgpu`, `client`, `server`, or `common`. This establishes the relevance of the inspected implementation, not the provenance of the unversioned cloud package.

## 1. Summary

rGPU implements two independent ways to retain an application process on a client while executing accelerator work elsewhere: CUDA/library interception and a PyTorch device that predicts output metadata locally, allocates remote identities before execution, and streams operations to a stateful server. It adds optional forward/backward graph shipping and transport reconnection with bounded replay, distinguishing receipt and submission from GPU completion. The paper offers these mechanisms as an implementation and characterization contribution rather than inventing remoting or asynchronous execution. Its evidence combines explicitly historical deployment comparisons, a CPU loopback mechanism experiment, analytical wait-cost calculations, and a new L40S comparison reporting substantial eager-remoting slowdowns relative to native execution. The strongest new proposed explanation is that client dispatch dominates those slowdowns and can hide WAN latency; that explanation goes beyond what the measurements establish.

## 2. Verdict

**Reject for OSDI/NSDI/ATC as it stands.** The manuscript does not establish a sufficiently distinct research contribution, and its principal new causal conclusion confounds client CPU, software runtime, placement, and server contention. Correct arithmetic does not repair the missing controlled decomposition, optimizer-baseline mismatch, limited correctness validation, or lack of measurements comparing the two advertised boundaries.

**Current fit:** a systems/ML-systems workshop experience paper, work-in-progress presentation, or implementation demo, after removing the unsupported causal assertions. The present artifact is useful as an engineering report; it is not yet a convincing full research submission.

## 3. Major issues

### 1. The central research contribution remains an implementation combination, without the comparative evidence needed to turn it into a systems result

**Location:** Introduction, lines 17–19: “a concrete implementation and characterization of their combination”; Section 2.4, lines 59–76; Sections 7.2–7.3, lines 330–342.

The paper fairly concedes that its ingredients are established. However, the promised question—how the interception boundary determines practicality—is not answered by measuring primarily one boundary. F and G exercise the Python path; the CUDA evidence is an unrelated historical launch microbenchmark. There is no workload run through both interfaces on the same hardware, no measured alternative based on coarse RPC, and no demonstration of a client-resident integration that cannot be served adequately by executing the program remotely. The counter and graph results are predictable consequences of aggregation; a second implementation does not by itself establish a new tradeoff. A systems paper need not outperform every predecessor, but it must deliver a new, defensible insight, capability, or systematic characterization.

**Concrete fix:** choose one contribution and organize the evaluation around it. For a boundary-characterization paper, compare CUDA interception, eager ATen, shipped graphs, and a whole-step RPC on matched workloads and hardware; explain the measured compatibility/performance crossover. Alternatively, narrow to the Python path and establish a precise semantic or recovery result with a comprehensive experiment. Move unmeasured CUDA material to background/appendix if it is not part of that result. Merely adding more citations or another GPU timing will not resolve this issue.

### 2. Section 6.8 promotes plausible hypotheses and accounting calculations into an established client/network bottleneck diagnosis

**Location:** Abstract, line 7: “client-side per-operation work, not the network, dominates”; Section 6.8, lines 309–315: “The counters say where the time goes,” “confirms,” “hides the network entirely”; Section 7.1, lines 324–326: network cost “bounded by” other rows.

The 6399.59/3214.55 = 1.99 ratio is real. Its cause is not isolated. The WAN placement changes the CPU architecture, Python version, OS, client/server colocation, and tunnel, and runs in a different process/time period. The 1.23/0.53 = 2.32 microbenchmark ratio measures a lambda loop, ordinary CPU tensor dispatch, allocation, and a one-element add. It does not measure RemoteTensor dispatch, meta kernels, serialization, autograd, optimizer traversal, or server execution. Agreement between two ratios is suggestive, not a bottleneck diagnosis. At roughly one microsecond per ordinary add versus 88 microseconds of aggregate excess time per protocol message, it is especially inappropriate to treat the former as a calibrated model of the latter.

Concrete competing explanations are client/server competition for CPU resources, cache/memory bandwidth or a container CPU quota; NUMA placement and frequency differences; different Python/runtime overhead; server-side decoding/dispatch and tensor lifetime costs; launch starvation or changed GPU utilization; and time-varying shared-host load. These are alternatives to test, not findings that any one occurred. A server bottleneck could also change when its colocated client is removed. Identical message counts do not distinguish these explanations.

Three further deductions are invalid:

- `(T_colocated − T_native)/M = 87.78 us` is an **A calculation on G**, not measured client service time. It includes server overhead, local transport, scheduling, different execution choices, and changed overlap. Graph shipping removes server decoding, argument reconstruction, ID-table operations, and repeated protocol dispatch as well as client work; its speedup cannot localize overhead to the client.
- One wait and 5.6 MB of payload do not establish that the link is absent from the critical path. Streaming requests still experience transfer, backpressure, buffering, and dependency/arrival effects. Section 3.2 explicitly warns about this; Section 6.8 then disregards the warning. The graph delta of 48.608733 ms is consistent with one additional RTT, but a cross-client subtraction is not a clean network decomposition. Other workloads cannot provide a quantitative bound on eager training's network cost without an additional model.
- “Host observations are free on loopback” is contradicted by the colocated chain: 92.717946 − 55.742010 = 36.975936 ms for 20 extra waits, or 1.848797 ms per added wait. The training medians differ by only 0.772611 ms, but that is below the observed variability and is compatible with hidden overhead; it does not show zero-cost barriers. A loss download is also not equivalent to a device-wide synchronization.

**Concrete fix:** replace the abstract and Section 7.1 conclusions with: “Eager execution has substantial non-WAN overhead. Faster WAN execution is consistent with client CPU effects, but client speed, server contention, runtime differences, and network cost have not been separated.” Label 88 us/message as aggregate excess time, retain the measured ratios, and replace “free,” “confirms,” “entirely,” and the cross-workload bound with scoped observations. Settle causality with the controlled profiling/RTT experiment in Section 10 below. Withdraw the unsupported attribution of the eager-chain residual to SSH/jitter and the unexplained “75 ms of client work” floor.

### 3. The native training baseline does not hold the optimizer execution path constant

**Location:** Section 6.8, lines 287–289: “same model, inputs, precision, software”; benchmark `paper/gpu_benchmark.py:221–232`, especially `AdamW(model.parameters(), lr=3e-3)` at line 223.

AdamW leaves `foreach` and `fused` unspecified. The installed Torch 2.14 implementation defaults to foreach for eligible ordinary CUDA parameters but excludes tensor subclasses through an exact-type check. rGPU model parameters remain `RemoteTensor` objects even though `isinstance(p, nn.Parameter)` is true. Thus the baseline can use the multi-tensor optimizer while rGPU takes the single-tensor path. This changes Python dispatch, ATen operation count, and GPU launch behavior before transport cost is considered.

Evidence: `python/.venv/lib/python3.14/site-packages/torch/optim/optimizer.py:45,170–195`; `torch/optim/adam.py:933–968`; `torch/optim/adamw.py:30–47` in the same site-packages directory. A local probe with network calls disabled and `Connection.post` replaced by a no-op constructed `nn.Linear(2,2).to("rgpu")`: its parameter type was `RemoteTensor`, and `_default_to_fused_or_foreach([p], False)` returned `(False, False)`. The native CUDA selection follows the source predicate; I did not run it on a GPU or inspect the historical cloud wheel itself.

This does not invalidate the reported **user-visible** slowdown against normal native defaults. It does invalidate interpreting the entire difference as transport/metadata overhead under an otherwise identical execution path. It also reinforces that the message count partly reflects framework decisions rather than an intrinsic property of the model.

**Concrete fix:** retain the current default-native result as a practical baseline, add a matched `foreach=False, fused=False` comparison on both paths, and explicitly trace which optimizer implementation and kernels execute. If a supported foreach rGPU path exists, measure it separately. Attribute the difference between native variants rather than silently replacing the faster baseline.

### 4. The empirical scope and numerical check do not validate the principal training and two-boundary claims

**Location:** Sections 4.5 and 6.7–6.8, lines 142–146, 281–283, 289, 311; conclusion, lines 346–348.

The only fresh graph-performance measurements are a 1,024-element arithmetic chain. There is no G graph-shipped transformer training result, no real-GPU CUDA-path experiment, no cross-boundary comparison, and no startup/amortization measurement. The H graph run is explicitly a different workload/deployment and was slower; it cannot fill these holes. The current measurements therefore cannot tell a user when the advertised design becomes a worthwhile alternative to a coarse remote step.

The G numerical check is a single aggregate loss from an evaluation-mode forward with dropout zero (`gpu_benchmark.py:198–211`). Timed training uses dropout 0.2 and AdamW; the only check on its final loss is `final == final and final < 10` (lines 214–253). This does not check gradients, parameter updates, optimizer state, trajectory, or even comparable outputs across placements. A lost or wrong update could pass. CPU semantic tests are relevant evidence for their tested paths, but they are not CUDA validation of this model/optimizer combination.

**Concrete fix:** add graph-shipped training, startup/steady-state/break-even measurements, and a practical coarse-RPC baseline; add at least one paired real-GPU CUDA/Python workload if retaining the two-boundary contribution. For correctness, initialize native and remote from an identical archived state, disable dropout or supply identical masks, and compare logits/loss, gradients, parameters, and AdamW moments over several steps. Then separately test stochastic training with appropriate tolerances. Restrict existing claims to a successful forward sanity check and completed short training blocks until these checks exist.

### 5. Important semantic qualifications are missing, including a verified compiled-output metadata gap

**Location:** Section 4.3, line 128: “explicitly preserves conjugate and negative metadata bits”; Sections 4.6–4.7, lines 150–162; Section 5, line 172: “Unimplemented entry points return an unsupported result.”

The eager wrapper copies conjugate/negative bits from its input metadata, but compiled-output reconstruction creates fresh metadata from only dtype, shape, stride, and offset. A socket-free probe of `_compiler` with a valid FX conjugate-output graph produced `captured is_conj=True`, `returned is_conj=False`. This is a verified gap in the helper used by compiled execution, not an end-to-end numerical failure demonstrated for the benchmark. The manuscript's metadata discussion needs to distinguish eager preservation from reconstructed graph outputs. Similarly, some CUDA compatibility entry points return success while doing nothing, contrary to the blanket unsupported-result statement. Deferred-error poisoning and recovery also have narrower contracts than a reader could infer; details are in Section 6 of this review.

**Concrete fix:** qualify these claims, enumerate deliberate no-op/unsupported semantics, and add a compatibility matrix that includes compiled complex views, mutation failures, and CUDA-specific behavior. Before claiming compiled metadata preservation, repair or explicitly reject missing metadata cases and run a focused end-to-end correctness check. No manuscript or implementation changes were made in this review.

### 6. The artifact does not substantiate the stated level of experimental control or provenance

**Location:** Section 6.8, lines 289–291; Appendix A, lines 382–418; `paper/README.md:45`.

The WAN client is Python 3.14.7/macOS/Torch 2.14.0; the host client is Python 3.10.12/Linux/Torch 2.14.0+cu130. Matching PyTorch release versions avoids the H schema-version mismatch, but it is not matching software. The colocated JSON's `rgpu_commit` is empty, the native JSON has no benchmark revision, and the WAN JSON identifies the client checkout rather than the server package. The script sets `PYTHONPATH=/root/rgpu_pkg` without recording that package's hash. `env.txt`, raw RTT samples and their measurement command, and raw dispatch samples are absent from the delivered artifact. Consequently I cannot independently verify the driver/core/CPU/RTT provenance or the cloud package from the JSONs.

The manuscript also says each G JSON has 35 samples, but `native.json` has 14; it says two warmup blocks, but the instrument performs two individual calls/steps. Placement order is fixed native→colocated→WAN, while only configurations within a placement are shuffled. All seven trials share a process/session for each leg. These do not make the timings fabricated or useless; they constrain the uncertainty and causal claims that can be attached to them.

**Concrete fix:** correct the descriptions and publish a manifest with hashes of the actual benchmark and client/server source packages, both endpoint environments, CPU affinity/quota/NUMA/frequency information, the original RTT and dispatch samples, and timestamped host-load telemetry. Preserve the existing samples as one run. Add randomized process-level repetitions across placements; do not present within-process repeats as independently controlled deployment trials.

## 4. Verification of numbers

### Method and scope

I grouped every raw sample by configuration, recomputed median/minimum/maximum elapsed time and medians of all four counters, and asserted exact agreement with each JSON's stored summary. All assertions passed. Each configuration has seven samples. Counts are F=35, G-native=14, G-colocated=35, G-WAN=35: **119 raw timing samples total**, not 140. Message and wait counts are constant within every configuration, not merely equal in the medians.

The following standard-library check is the compact form of the recomputation I ran from the repository root; it requires no benchmark execution or network service:

```python
import json
import statistics as st
from pathlib import Path

D = {}
for leg in ("results", "native", "colocated", "wan"):
    d = json.loads(Path(f"paper/{leg}.json").read_text())
    D[leg] = {}
    for key, recorded in d["summary"].items():
        rows = [r for r in d["samples"] if
                (f'{r["workload"]}/{r["case"]}' if "workload" in r
                 else r["case"]) == key]
        assert len(rows) == 7
        times = [r["elapsed_ms"] for r in rows]
        calc = dict(median_ms=st.median(times), min_ms=min(times),
                    max_ms=max(times))
        calc.update({k: st.median(r[k] for r in rows)
                     for k in ("messages", "waits", "bytes_out", "bytes_in")})
        assert calc == recorded, (leg, key, calc, recorded)
        D[leg][key] = calc
        print(leg, key, calc)

f, n, c, w = (D[k] for k in ("results", "native", "colocated", "wan"))
def t(d, k): return d[k]["median_ms"]
for label, value in {
    "F batch ratio": t(f, "eager_flush1") / t(f, "eager_flush64"),
    "F sync ratio": t(f, "eager_sync_each_call") / t(f, "eager_flush64"),
    "F message ratio": 601 / 41,
    "F byte ratio": 45273 / 1968,
    "F graph speedup": t(f, "eager_flush64") / t(f, "graph_eager"),
    "F graph/native": t(f, "graph_eager") / t(f, "native_cpu"),
    "F eager/native": t(f, "eager_flush64") / t(f, "native_cpu"),
    "G train colocated/native": t(c, "train/eager_flush64") / t(n, "train/native"),
    "G train WAN/native": t(w, "train/eager_flush64") / t(n, "train/native"),
    "G train colocated/WAN": t(c, "train/eager_flush64") / t(w, "train/eager_flush64"),
    "dispatch ratio": 1.23 / .53,
    "G chain eager/native": t(c, "chain/eager_flush64") / t(n, "chain/native"),
    "G chain graph/native": t(c, "chain/graph_eager") / t(n, "chain/native"),
    "G graph WAN delta": t(w, "chain/graph_eager") - t(c, "chain/graph_eager"),
    "G chain ms/extra wait": (t(w, "chain/eager_sync_each_call") - t(w, "chain/eager_flush64")) / 20,
    "G train ms/observation": (t(w, "train/eager_sync_each_step") - t(w, "train/eager_flush64")) / 10,
    "G messages/step": 64941 / 10,
    "G excess ms/step": (t(c, "train/eager_flush64") - t(n, "train/native")) / 10,
    "G excess us/message": (t(c, "train/eager_flush64") - t(n, "train/native")) * 1000 / 64941,
}.items():
    print(label, value)
```

### Table 4: all medians, ranges, and counts

Values below are recomputed, in milliseconds. More digits are shown for auditability, not as a claim of experimental precision.

| Configuration | Median | Minimum | Maximum | M | W | Median bytes out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native CPU | 0.663459 | 0.584417 | 1.353792 | 0 | 0 | 0 |
| Eager flush 1 | 51.560458 | 42.232250 | 168.917417 | 601 | 1 | 47,228 |
| Eager flush 64 | 45.421166 | 38.362417 | 62.474250 | 601 | 1 | 45,273 |
| Eager sync each call | 68.554084 | 50.223041 | 129.562083 | 621 | 21 | 45,788 |
| Graph server eager | 2.065208 | 1.859709 | 31.840500 | 41 | 1 | 1,968 |

**All Table 4 values match at the displayed precision**, accepting the even last digit at the 31.8405 rounding tie. Incoming bytes are respectively 0, 24, 24, 504, 24; they also match the JSON summaries.

| Section 6.4 ratio | Recomputed | Paper | Assessment |
| --- | ---: | ---: | --- |
| Flush 1 / flush 64 time | 1.135164 | about 1.1 | Matches |
| Sync / default eager time | 1.509298 | about 1.5 | Matches |
| Eager / graph messages | 14.658537 | about 15 | Matches |
| Eager / graph bytes | 23.004573 | about 23 | Matches |
| Eager / graph time | 21.993507 | about 22 | Matches |
| Graph / native time | 3.112789 | about 3.1 | Matches |
| Eager / native time | 68.461150 | about 68 | Matches |

### Table 6: all medians, caption ranges, and counts

| Placement / configuration | Median ms | Minimum ms | Maximum ms | M | W |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native chain | 2.024261 | 1.991510 | 2.070752 | 0 | 0 |
| Native train | 698.983318 | 688.634143 | 700.832646 | 0 | 0 |
| Colocated chain eager | 55.742010 | 55.373434 | 56.595703 | 601 | 1 |
| Colocated chain sync | 92.717946 | 92.217839 | 96.531314 | 621 | 21 |
| Colocated chain graph | 7.779267 | 6.014541 | 7.802497 | 41 | 1 |
| Colocated train eager | 6399.585394 | 6363.317498 | 6471.587379 | 64,941 | 1 |
| Colocated train observations | 6400.358005 | 6266.546250 | 6460.228928 | 64,951 | 11 |
| WAN chain eager | 150.470416 | 108.700000 | 202.565292 | 601 | 1 |
| WAN chain sync | 1194.064375 | 1116.926792 | 1384.360125 | 621 | 21 |
| WAN chain graph | 56.388000 | 50.674875 | 58.117166 | 41 | 1 |
| WAN train eager | 3214.548625 | 3159.027042 | 3320.611083 | 64,941 | 1 |
| WAN train observations | 3931.325333 | 3804.813833 | 4344.650750 | 64,951 | 11 |

**Every Table 6 median, every range in its caption, and every M/W entry matches the raw samples at the displayed precision.** Median outgoing bytes for each remote placement are 45,273; 45,788; 1,968; 5,569,105; and 5,569,470 in the row order above. Incoming medians are 24; 504; 24; 24; and 624. Eager training's outgoing count is not constant across trials: both placements contain 5,565,964 as well as 5,569,105 bytes. The “same 5,569,105 bytes” statement is correct as a median comparison, not as an assertion about every timed block. These counters are protocol payload accounting, not SSH/TCP wire volume.

### Section 6.8 arithmetic

| Claimed quantity | Calculation from unrounded medians | Result | Assessment |
| --- | --- | ---: | --- |
| 48.6 ms graph WAN increment | 56.388000 − 7.779267 | 48.608733 ms | Arithmetic matches; network attribution confounded |
| 52 ms per added chain wait | (1194.064375 − 150.470416) / 20 | 52.179698 ms | Matches; difference of medians, not measured wait duration |
| 717 ms for ten loss observations | 3931.325333 − 3214.548625 | 716.776708 ms | Matches |
| 72 ms per observation | previous / 10 | 71.677671 ms | Matches; queue/overlap attribution unmeasured |
| WAN training about half colocated / 2.0x | 6399.585394 / 3214.548625 | 1.990819x | Matches |
| M3 dispatch 2.3x | 1.23 / 0.53 | 2.320755x | Matches rounded `dispatch.txt` inputs; raw samples unavailable |
| WAN training 4.6x native | 3214.548625 / 698.983318 | 4.598892x | Matches |
| Colocated training 9.2x native | 6399.585394 / 698.983318 | 9.155562x | Matches |
| Colocated eager chain 27.5x native | 55.742010 / 2.024261 | 27.536968x | Matches |
| Colocated graph 3.8x native | 7.779267 / 2.024261 | 3.843016x | Matches |
| About 6,500 messages/step | 64,941 / 10 | 6,494.1 | Matches; amortized block count, including lifetime/control messages |
| Roughly 570 ms overhead/step | (6399.585394 − 698.983318) / 10 | 570.060208 ms | Matches aggregate excess elapsed time |
| About 90 us/message | (6399.585394 − 698.983318) × 1000 / 64,941 | 87.781249 us | Matches arithmetic; not measured client dispatch time |
| Less than one part in a thousand | (6400.358005 − 6399.585394) / 6399.585394 | 0.000120728 | Matches medians; does not establish equivalence or zero cost |

These calculations use G inputs but their cost decompositions are **A**, not independently measured G components. The 48.8 ms RTT, 46.3–60.6 ms range, and nine-sample count cannot be recomputed from the supplied files because the RTT samples are absent. The “75 ms of client work” floor at line 315 has no reproducible derivation. Even the unjustified scaling `55.742010 / (1.23/0.53)` gives 24.018915 ms, or 72.818915 ms **including** 48.8 ms RTT, not 75 ms of client work plus RTT. Withdraw that estimate rather than treating this alternative scaling as a valid model.

For scale, the median of within-trial sync-minus-eager differences is 11.105117 ms for colocated training, with a −205.041129 to 37.040507 ms range; it is not the 0.772611 ms difference of configuration medians. WAN training's paired median difference is 681.488083 ms, versus 716.776708 ms from separate medians. These are different estimands, not arithmetic errors in Table 6. The paper should specify which it intends and should not call either an instrumented per-wait latency.

### Abstract and numerical mismatches

The abstract's F numbers are correct: 601→41 messages, 45.421166→2.065208 ms, displayed as 45.42→2.07 ms. Graph execution remains slower than native CPU (3.112789x). Its G 9.2x, 4.6x, and approximately 6,500 messages/step match the calculations above. “Same training step” is acceptable at the model/task level, but not as a claim of identical optimizer execution. The H 20-step, 20.8 s, 60.8 s, and reported 116 ms figures agree with `docs/performance-notes.md:14–28`; there are no H raw trials to independently reconstruct. H is not used in any G ratio here. The abstract's RTT and bottleneck claims have the evidentiary limitations already described.

| Location | Paper/artifact value | Verified value or status | Correction |
| --- | --- | --- | --- |
| Appendix A, line 402; README artifact inventory | Each G JSON has 35 samples | Native 14, colocated 35, WAN 35 | State counts individually; total G is 84 |
| Section 6.8, line 289 | Two warmup blocks each | Two calls for the chain; two individual training steps | Correct unit; not two 20-call or 10-step blocks |
| Section 6.8, line 289; README line 45 | Torch 2.14.0+cu130 both ends | Host 2.14.0+cu130; laptop 2.14.0; Python 3.10.12 vs 3.14.7 | Say matching Torch release, disclose builds/runtimes |
| Appendix A, line 384 | Original eager/graph ratio 21.9 | 21.993506707 | 22 at two significant figures, or 22.0 at one decimal |
| Section 6.1, line 180 | Three evidence classes | Four: H, F, A, G | Correct count |
| Section 6.8, line 315 | Expected floor near 75 ms of client work plus RTT | Unsupported; no identifiable measured component | Remove or derive with a validated model |

No other Table 4/Table 6 arithmetic mismatch was found. The RTT and dispatch raw data gaps are **unverifiable inputs**, not demonstrated numerical errors. The measured G model has 10,788,929 parameters (locally counted without executing the benchmark), correctly rounded to 10.8M. Its dropout-free losses all round to 4.33796, but the exact CPU reference is 4.337957859 on the host and 4.337956905 on the laptop; device loss is 4.337957382 in all three records. “Equal” must mean the specified tolerance/rounding, not bitwise equality.

## 5. Claims versus evidence

“Sufficient” below means sufficient for the stated, narrow claim, not proof of generality. Source support is marked explicitly because implementation facts cannot be assigned to a timing dataset. A on G denotes arithmetic based on G; it does not combine G with H.

| Substantive claim in abstract, introduction, or conclusion | Evidence class / source | Support | Assessment |
| --- | --- | --- | --- |
| Application placement can differ from accelerator placement while control stays on the client (abstract; introduction lines 11–15) | Source; H and G examples | Sufficient | Implemented for supported programs; no measured necessity or usability advantage over remote whole-program execution |
| Two independent CUDA and PyTorch interfaces exist (abstract; introduction line 15; conclusion line 346) | Source, not H/F/A/G | Sufficient | Separate clients, servers and protocols exist |
| CUDA interception accommodates existing supported binaries (introduction line 15; conclusion line 346) | Source; limited H CUDA observation | Partial | Mechanism exists; no fresh supported-binary compatibility or GPU evaluation matrix |
| Local metadata and client-assigned IDs allow subsequent operations without individual replies (abstract; introduction lines 13–15; conclusion line 346) | Source; F/G M/W counters | Sufficient | Demonstrated for measured operations, not arbitrary operators or every metadata reconstruction path |
| Fixed graphs can be registered once and invoked by ID (introduction line 15) | Source; F/G chain | Sufficient | Supported static graph path; registration excluded from timed results |
| Graph shipping moves repeated forward/backward computation server-side (abstract) | Source; selected CPU tests; H training observation | Partial | Forward/backward support exists; fresh G only measures a forward arithmetic chain |
| Both paths separate connection from longer-lived session (abstract) | Source; recorded CPU reconnect tests | Sufficient | Live-session mechanism exists; not durability or arbitrary-failure recovery |
| Ordering, deferred errors and bounded recovery make streaming useful (introduction lines 13,19; conclusion line 346) | Source; selected CPU tests | Partial | Mechanisms verified in limited paths; no full fault matrix, latency bounds or CUDA validation |
| Boundary choice determines compatibility/performance tradeoffs (introduction lines 17–19; conclusion line 346) | Source and prior work; F/G only Python | Partial | Conceptual argument, not a controlled two-boundary characterization |
| H transformer loop: 20.8 s remote A40 vs 60.8 s local MPS at reported 116 ms RTT (abstract) | H | Sufficient for a reported historical observation | Not reproducible from raw trials, not native A40 overhead, not combined with G |
| Compilation can reduce messages yet slow a short training run (introduction line 19) | H | Sufficient for that observation | Does not determine warm speed or break-even |
| Tiny operations remain cheaper locally (introduction line 19; abstract CPU qualification) | F; G chain | Sufficient for measured chain | Avoid universalizing across operations/hardware |
| CPU graph shipping cuts 601 to 41 messages and 45.42 to 2.07 ms, but loses to native (abstract) | F | Sufficient | Warmed chain; excludes compile/startup; substantial variability |
| L40S eager training takes 9.2x colocated and 4.6x WAN versus native (abstract) | G; A on G ratios | Sufficient as configuration ratios | Partial as isolated remoting cost; optimizer and placement confounds remain |
| About 6,500 messages per training step (abstract) | G counters; A on G | Sufficient | Amortized ten-step count, including control/lifetime messages |
| WAN contributes roughly one RTT per observation; graph shipping costs one RTT plus execution (abstract) | G deltas; A interpretation | Partial | Consistent small-chain observations; no same-client RTT sweep or training decomposition |
| Client per-operation work dominates eager remoting, rather than the network (abstract) | G plus ordinary-add microbenchmark | Absent as an established causal conclusion | Evidence motivates the hypothesis; does not locate the bottleneck or bound network cost |
| Graph shipping substantially reduces software overhead for repeated computations (conclusion line 348) | F/G chain; source | Sufficient narrowly | Not demonstrated for fresh GPU training; not exclusively client overhead |
| Remoting can be useful when enough work stays remote between observations (conclusion line 348) | H feasibility; F/G sensitivity; A | Partial | Plausible conditional result; no controlled usefulness frontier against practical alternatives |
| Low message count or async submission alone does not establish end-to-end performance (abstract qualifications; conclusion line 348) | H negative graph result; F/G native comparisons | Sufficient | This limitation is supported and should govern Section 6.8 too |
| No near-native-GPU or superiority-over-prior-systems result is established (abstract) | G slowdowns; absent head-to-head evaluation | Sufficient | Appropriate boundary on the claims |

The H/G separation is generally maintained: Sections 6.2, 6.6, 6.8's final paragraph and the abstract do not numerically combine the A40 and L40S measurements. The principal evidence-class violations are **A being presented as causal G measurement** in lines 309–315 and 324–326, and F/G graph results being used to “confirm” a client-only mechanism. The CPU-versus-GPU chain comparison at line 309 also attributes its difference “only” to native launch overhead without a decomposition; keep it as a cross-experiment observation, not a causal conclusion.

## 6. Implementation accuracy

### Confirmed design descriptions

The ordinary meta-first dispatch, data-dependent fallback, write-alias handling, ordered post/request path, encode-before-commit behavior, 64-message/256-KiB thresholds, separate ACK/SYNC handlers, ATen-only graph reconstruction, generation-protected session attachment, and process-global RNG limitation are described substantially accurately. Relevant anchors are `python/rgpu/dispatch.py:100–217`, `session.py:135–217`, `server/session.py:185–223`, `server/__main__.py:100–135`, `compile.py:44–98,127–168`, and `server/graphs.py:19–60`. I did not find evidence that the paper conflates the Python and C++ wire formats.

### Material qualifications and discrepancies

| Manuscript claim | Source evidence | Required qualification |
| --- | --- | --- |
| Section 4.3 preserves conjugate/negative bits | `tensor.py:76–92` copies bits, but `compile.py:143–153` reconstructs outputs through `meta_like`; `tensor.py:66–71` has no bit arguments | Eager-wrapper preservation does not guarantee compiled-output preservation. The socket-free conjugate-output probe returned a cleared bit. Test complete compiled complex-view behavior; do not claim a benchmark numerical failure from this helper-only probe |
| Metadata returned by synchronous fallback is sufficient | `server/session.py:174–183` transmits dtype/shape/stride/offset only; `dispatch.py:168–173` rebuilds from those | The same metadata-bit fields are absent from this wire description. I did not establish a supported data-dependent operator exposing a wrong result here; document/audit the contract rather than assert a reproduced bug |
| Section 4.6 poisoning prevents dependent error cascades | `dispatch.py:118–120` sends `None` for written-input returns; `server/session.py:91–114` poisons only non-None output IDs | Failed in-place operations do not poison the written input through this mechanism. Later operations on that input can continue before observation; partial mutation is not rolled back. The existing no-transaction caveat is correct but should explicitly address this case |
| Later SYNC additionally waits for the device | `server/session.py:211–216` reports and clears `first_error` instead of executing the requested handler | A SYNC that raises a deferred error is not a completed-device barrier. Successful benchmark barriers are unaffected; describe success and error semantics separately |
| Ordering implies timely streaming | `session.py:206–221,287–307` flushes on thresholds or explicit request/flush; there is no periodic sender | A sub-threshold tail can remain buffered during client-only work until a later flush/observation. Ordering is implemented, but prompt execution/progress and overlap are not automatic |
| Default reconnect budget is 60 seconds | `session.py:104–105,114–121,243–260,309–323` uses 10-second connect and 300-second receive timeouts; deadline is checked outside these calls | This is a retry-loop budget, not an end-to-end 60-second failure-detection/recovery bound. A handshake/receive can overrun it. Report detection and recovery separately |
| C++ recovery is analogous to Python recovery | `client/rpc.cpp:208–219,418–439,557–568,620–637` | C++ disables replay after its 64-MiB cap; no re-enable assignment exists in this file, unlike Python's acknowledgment reset. A waited call gets two transport attempts and can return an uncertain error/abandon the call. Do not imply identical retry/replay contracts |
| Unimplemented CUDA entry points return unsupported | `client/cudart_impl.cpp:926–931`: profiler start/stop return success without action; `client/shim.cpp:698–711`: non-DEVICEMAP host registration/unregistration also return success without registration | State that generated unsupported stubs return errors, but some compatibility functions are intentional no-ops. Distinguish them in coverage reporting; do not present symbol coverage as semantic transparency |
| Graph lifetime follows tensor lifetime, by implication of Sections 4.3/4.5 | `server/session.py:43–47,124–126,134–139`: FREE releases tensors; graph table retains registered callables | Graph reclamation is session-scoped in the inspected implementation. Repeated recompilation can accumulate graph entries in a long-lived session. This is material to interactive use, even though long-run memory is already listed as unmeasured |
| Protocol counters locate client cost/network traffic | `session.py:175–178,212–217,274–278,290–295,324`; `wire.py:380–384` | M counts logical queued messages including FREE/control; W counts explicit requests, not all socket blocking; payload bytes omit frame-length headers and TCP/SSH overhead. They cannot identify where elapsed time was spent |

The C++ thread-state claim in Section 4.6 is supported by `server/client_threads.h:78–118` and `server/main.cpp:760–835`: deferred errors live in session thread state and are not simply connection-local. Error delivery is conditional—only a suitable successful reply consumes a held error—so it should not be generalized to “every next call” or arbitrary GPU asynchronous failure attribution.

For reproducibility, the essential compiled-metadata probe I ran was:

```python
# Run with python/.venv/bin/python -B; no files are written.
import socket
def blocked(*args, **kwargs):
    raise AssertionError("network forbidden in review probe")
socket.create_connection = blocked
import torch, rgpu
from rgpu import session
from rgpu.compile import _compiler
from rgpu.tensor import register
session.Connection.post = lambda *args, **kwargs: None
g = torch.fx.Graph()
x = g.placeholder("x")
y = g.call_function(torch.ops.aten._conj.default, (x,))
g.output((y,))
y.meta["val"] = torch.empty(2, dtype=torch.complex64, device="meta").conj()
fn = _compiler(None, "eager")(torch.fx.GraphModule(torch.nn.Module(), g), [])
m = torch.empty(2, dtype=torch.complex64, device="meta")
register(m)
out = fn([m])[0]
assert y.meta["val"].is_conj() and not out.is_conj()
```

## 7. Methodology

**Statistical adequacy.** Seven trials are adequate to expose gross effects such as a 4.6–9.2x configuration slowdown or the repeated-chain graph reduction. They are not adequate to establish tails, equivalence, or precise cost components. Medians and full sample plots are appropriate descriptive summaries; min/max are not uncertainty intervals. F has a 31.8405 ms graph outlier versus a 2.065208 ms median and a 168.917417 ms flush-1 outlier versus 51.560458 ms median. This supports the paper's caution about tails, but also makes the modest 1.1x batching effect inconclusive without uncertainty on a paired comparison. F's later reported rerun is another run, not evidence that only the preferred original timings should drive the paper. Archive it rather than preserving only its narrative summary.

The G trials are repeated blocks within one process and, for each remote leg, one session. Reusing server, allocator, graph/compiler caches and the same GPU creates correlated conditions. Training reconstructs its model/optimizer each case, which reduces some state carry-over, but does not create an independent deployment trial. Report both within-run dispersion and independent process-level repetitions. With only seven independent observations, even a distribution-free min-to-max interval for a population median would be very wide; more importantly, independence is not established here. Do not infer “free” from a sub-millisecond difference among multi-second medians with ranges spanning hundreds of milliseconds.

**Warmup.** F performs two function calls per case per trial; G does two chain calls or two training steps (`benchmark.py:66–68`; `gpu_benchmark.py:180–182,235–237`). The G text incorrectly says two blocks. Warmup and barriers exclude graph registration/tracing and initial optimizer state creation from timed measurements. This is appropriate for a warm-throughput measurement but not for end-to-end interactive latency or amortization. Two steps do not demonstrate stable GPU clocks, allocator behavior, or Inductor warm state; the measured graph backend is server eager, so do not describe its warm time as an Inductor result.

**Timing boundaries and barriers.** The timed remote windows include a final successful `torch.rgpu.synchronize`; source inspection confirms the server calls `torch.cuda.synchronize(self.device)`. Native CUDA also ends with a device barrier. Initial uploads, reference checks, compilation and final downloads occur outside timing; these are completed warmed blocks, not deployment/run startup costs. The repeated chain starts from the same input on each call; it is not a 200-stage stateful chain. Training reuses already-uploaded token batches, so the result does not measure streaming fresh minibatches or input-pipeline traffic.

The training observation variant executes `loss.item()`, not a device-wide synchronization on every step. It downloads a loss and waits for dependencies needed to read it; it need not wait for every later optimizer operation in the step (`dispatch.py:257–258`; `server/session.py:166–167`). The final barrier establishes completion for the block. Rename “sync each step” to “loss observation each step” and keep it separate from the chain's explicit SYNC variant.

**Randomization.** F shuffles five configurations within each of seven trials with seed zero. G shuffles chain configurations and training configurations separately within each leg; native has one configuration per workload. All chain trials precede training trials. `run_on_gpu_host.sh:41–44` fixes native before colocated, and the WAN leg runs afterwards. Thus “seven trials in randomized order” must not imply randomized placements or network conditions. A paired, interleaved design is needed to separate drift from treatment effects.

**Host and software controls.** One physical GPU removes a major hardware difference and disabling TF32 is useful. It does not fix client CPU/software, optimizer implementation, GPU frequency/power/load, CPU scheduling/NUMA/quota, allocator history, or threading. The GPU harness does not set Torch/OMP/MKL thread counts, while F and the dispatch microbenchmark do. A stated 256 host cores is not evidence of 256 available cores or uncontended scheduling inside the rented environment. The native and colocated legs do not run concurrently, but the colocated client and its server do; that difference specifically matters to the proposed explanation.

**WAN tunnel and network measurement.** The experiment measures rGPU over an SSH tunnel, not bare TCP. Record the tunnel command, SSH implementation, encryption/compression settings, and endpoint CPUs, then compare with an otherwise equivalent untunneled trusted test path or a local tunnel control. No such comparison is needed to report the deployment timing, but it is necessary to blame its residual on SSH. The nine RTT samples and probe definition are absent: a connection-establishment RTT is not necessarily the latency distribution of an application reply inside a persistent tunnel. There is no throughput, socket-blocking, packet-loss, or frame-timing record. Roughly ten chain flush batches are plausible from 601 messages and a 64-message threshold; they are not ten serialized RTTs. The existing text's batching/wait distinction is correct and should remain intact.

**Numerical checking.** The chain asserts output closeness against a CPU reference, outside the timer. This is a useful end-result sanity check, although it checks the final repeated output rather than all calls. The transformer checks the scalar evaluation loss within `rtol=atol=1e-4`; differences of approximately 4.8e-7 satisfy that check. It establishes agreement for one aggregate forward result. It does not establish training correctness, dropout equivalence, or matching AdamW behavior. Seeding alone cannot promise the same dropout masks across CPU/native-CUDA/remote operation decompositions. Use deterministic gradient/update comparisons with explicitly controlled masks, then a separate stochastic check.

**Correctness transcript and numerical rendering check.** `checks.txt` reports 40 tests passed in 42.54 s with one warning; the inspected tests include CPU gradient/update comparisons and reconnect cases. I verified the transcript and source, not a fresh execution. `render.py:23–32` checks a handful of stored summary values and the literal text “40 passed”; it neither recomputes raw-sample summaries nor parses numbers out of the manuscript. G figures consume summaries without assertions (`render.py:54–98`). These checks catch some accidental input changes; they are not independent numerical validation of the paper. The raw-sample audit above supplies that limited validation for this review, not a validation of causal interpretations or measurement provenance.

## 8. Related work

**Section 2.4 is defensible as a scope statement, not as a demonstration of novelty.** Explicitly avoiding incompatible benchmark rankings and distinguishing live-session reconnection from checkpointing is appropriate. The most consequential comparison is to mechanisms already acknowledged, not another long catalogue.

- **Wang et al. are closer than the current one-paragraph treatment makes operationally clear.** Their paper develops outstanding asynchronous requests, client-side shadow resources, and local handling of queries; it also studies batching tradeoffs and validates a latency/bandwidth model. The draft cites this correctly, but should explicitly say what its tensor-metadata contract or experiments add beyond those principles. The required advance cannot simply be that messages and waits differ. [Primary paper, Sections 3–5](https://arxiv.org/html/2401.13354v1).
- **Genie overlaps at the implementation-mechanism level, not just in broad motivation.** Its Section 3.2 describes a PyTorch `__torch_dispatch__` frontend with LazyTensor proxies, dependency/tensor-property capture, and FX processing. The phase/residency distinction in rGPU's discussion is real, but it should compare these shared mechanisms and clarify which claims in Genie are proposed versus evaluated. Do not imply rGPU is the implemented counterpart to a purely conceptual framework argument. [Genie paper, Sections 3.2 and 4](https://vincen.tl/files/hong25lost.pdf).
- **Kernel-as-a-Service is a relevant omission for graph shipping.** It separates host code from remotely executed graphs of GPU-only work. It is a different programming interface and serverless setting, not a like-for-like transparency competitor; it nevertheless weakens any implication that keeping host code separate while shipping GPU graphs is new. [KaaS author preprint](https://arxiv.org/abs/2212.08146).
- **Torpor is a relevant recent omission for asynchronous API redirection.** Its design includes asynchronous redirection and pipelined model execution in a serverless setting. Add it to the discussion of what rGPU does not attempt in resource sharing/management, without importing its performance numbers. [ATC 2025 proceedings entry](https://www.usenix.org/conference/atc25/presentation/yu).
- **Singularity is relevant if recovery remains a major contribution.** It addresses transparent migration and elasticity, a broader failure/state-management problem than maintaining a live remote session. A brief comparison would help delineate transport continuity from actual relocation/recovery of execution. [Author preprint](https://arxiv.org/abs/2202.07848).
- **Tessera's architectural separation is reasonable, but pin the cited version.** The arXiv abstract entry I retrieved retains the Tessera title, while the v3 full text calls the system *FluidGPU*. The full text still studies PTX-derived kernel dependencies and heterogeneous-GPU placement, so this does not overturn the stated distinction. The version/title inconsistency needs a precise versioned citation; “cluster fabric” should not imply an experimentally established interconnect requirement absent a specific cited result. [Abstract entry](https://arxiv.org/abs/2604.10180), [v3 full text](https://arxiv.org/html/2604.10180v3).

**Djinn:** the existing treatment is acceptable for this draft. It names the unavailable comparison, reports retrieval limitations, and refuses to invent mechanisms from a title. I do not count the missing full-text comparison as a new finding or penalize the authors as though a verified technical comparison had been omitted. Before submission they should revisit availability and keep priority claims conditional. Four retrieval-route details belong mainly in an artifact note; an inability to retrieve a paper through those routes is not a universal proof that nobody can access it. This qualification does not require pretending that the current review has read Djinn.

PyTorch RPC and whole-program remote execution are already acknowledged, but listing them as future work is insufficient: they are the immediate empirical alternatives to the problem this paper claims to solve. A coarse RPC does not need to match operator transparency; the evaluation should quantify what transparency costs and which client-local requirements justify that cost. I have not independently re-audited every statement about every older citation and do not assert that the remaining descriptions are exhaustive.

## 9. Minor issues

1. **Stale narrative after adding G.** The introduction still describes a preliminary evaluation whose required controlled GPU experiments are all future work (line 21); Section 6.8 says it fills the gap “Section 7.1 identified” (line 287), but the current Section 7.1 already describes G. The conclusion asks the next evaluation to measure same-GPU completed work (line 348), which this draft now does in a limited way. Reconcile these passages.
2. **Evidence-class count and visual labels.** Section 6.1 says three classes but defines four. Preserve all four; add explicit A-on-G labels to derived cost estimates so they cannot be mistaken for measured components. Do not merge H and G to simplify presentation.
3. **Table 6 caption placement.** The source places Figure 3 and its caption between Table 6 and its caption (lines 293–307). Put the table caption directly next to the table, and put milliseconds/block and block sizes in or directly above the table. The native rows and remote columns otherwise make a reader work to identify comparable units.
4. **Figure 3 compares unlike block lengths on one axis.** Twenty chain calls and ten training steps are correctly disclosed, but separate panels or explicit row labels would prevent cross-workload comparisons of block times. F/G plots show all seven samples, which is more informative than naked bars; overlapping G points still obscure multiplicity.
5. **Overprecise language.** Two-decimal millisecond values are useful for auditability, but do not interpret a 0.77 ms median difference between six-second runs as a scientifically resolved cost. Keep “ratio of medians” distinct from “median speedup” and from per-request latency.
6. **“Network contributes nothing” is too literal even on loopback.** There is no WAN propagation in that placement, but loopback TCP still has transport and scheduling costs. Use “no off-host link” when that is what is meant.
7. **“Pinned transformer” lacks an explicit model-source revision in the harness.** The implementation is in the benchmark itself, so archive its hash and describe it as that exact implementation rather than implying an externally pinned upstream model. Parameter count is correct; kernel choice and optimizer path still need disclosure.
8. **Introductory contribution list omits the new experiment.** Line 19 still foregrounds H plus a CPU mechanism experiment. Update it to match the final evaluation, with the causal caveats retained.
9. **README/render metadata has drifted.** README says “two figures” although the renderer produces three, and repeats the unjustified dispatch explanation as fact. Treat the manuscript as source of truth, but bring artifact instructions into agreement on the next author revision.
10. **Precise definition of “message.”** State in table notes that it includes FREE and control operations; otherwise readers can mistake 6,494 messages/step for 6,494 GPU arithmetic operators or launches. Transport frame counts are defined in Section 3 but not measured in the reported summaries.
11. **Reference status versus technical novelty.** Keep the explicit preliminary/experience positioning, but do not repeatedly defer the decisive experiments to Section 7 while making their hoped-for conclusions in the abstract. A limitations section cannot retract a stronger headline inference.

## 10. The three experiments that would most improve the paper

### 1. A factorial bottleneck and RTT experiment on a fixed server and controlled client

**Establishes:** whether eager training is client-bound, server-bound, or limited by their interaction; the causal contribution of RTT; and whether client CPU speed explains the surprising WAN result.

**Requires:** one L40S allocation, a separate Linux client with a matched Python/Torch stack, and controlled latency/bandwidth injection on a trusted test path. First hold client CPU/software and server fixed while sweeping RTT from loopback/LAN-scale through roughly 50 and 116 ms, bandwidth independently, and observation frequency. Measure eager training and the chain. Then vary client CPU resources/affinity at fixed network conditions. Run the colocated case with client/server pinned to disjoint cores, record NUMA placement and CPU quotas, and compare it with a separate-client condition. If retaining the M3 comparison, present it as another factor with its software explicitly recorded, not the network treatment itself.

Instrument client meta/encode/autograd time and blocked send/receive time, server decode/ATen/ID-lifetime work and queue depth, plus GPU utilization and launch gaps. Use low-overhead profiling and report its cost. Record direct and SSH-tunneled controls and interleave process-level repetitions. Force matched optimizer choices for the mechanistic comparison while preserving ordinary native defaults as a practical baseline. A flat response to RTT plus a client-rate-sensitive throughput ceiling and corresponding server/GPU idle periods would support the client-bound claim; a server-rate/affinity-sensitive ceiling would support the contention/server alternative. Counts alone cannot distinguish these outcomes.

### 2. A matched boundary/granularity comparison with compilation amortization

**Establishes:** what research value the two interfaces provide, when graph shipping improves an actual training workload, and what preserving client-side control costs relative to simple remote execution.

**Requires:** the same GPU/software and at least the transformer plus one contrasting workload with host observations, such as generation with explicit sampling placement. Compare native eager/default optimizer, matched native optimizer, CUDA interception where supported, eager ATen, shipped graph with server eager, shipped graph with Inductor, and a manually coarse whole-step RPC. Use the same observation/input/output contract across relevant variants; do not let the RPC hide a required client dependency. Report compatibility failures as failures, not speedups. Include startup/upload, tracing/registration/compile, first-step and warm costs, then enough iterations to measure an actual crossover. Add at least a small batch-size sweep. This can establish a useful characterization even if rGPU never wins on raw runtime; the missing result is the measured tradeoff, not a mandatory performance victory.

### 3. A GPU correctness and transport-fault matrix for stateful training

**Establishes:** that the work being timed preserves the intended state updates and that reconnect/deferred-error claims hold at the precise boundaries the design describes.

**Requires:** one GPU; a deterministic small network and the measured transformer; native/reference snapshots of parameters, gradients and optimizer state; a fault-injection proxy or narrowly placed server hooks. First compare several dropout-free AdamW steps across native, eager rGPU and compiled rGPU, including complex-view/output-metadata cases separately. Then inject disconnection before receipt, mid-batch, after an update but before reply delivery, and around replay-cap transitions; separately restart the server and expire sessions. Check non-idempotent updates for neither duplication nor omission, deferred errors across reconnection, explicit session-loss failures, and resource reclamation. Include detection time and recovery time as separate distributions. Add one real-CUDA-path case if the paper continues to claim correctness mechanisms across both boundaries. This is a bounded matrix tied to the paper's claims, not a request to prove all of CUDA or all PyTorch operators correct.
