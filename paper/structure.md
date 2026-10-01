# Structure and positioning for an rGPU systems paper

The recommended organizing question is **how the interception boundary and host-observation frequency determine the cost of remote GPU execution**. This gives the paper a systems argument that connects both implementations, rather than making it a catalogue of supported APIs. The draft follows this structure and labels the present evaluation as preliminary.

## What to borrow from prior papers

| Prior paper | Observed presentation pattern | Application to rGPU |
| --- | --- | --- |
| [rCUDA](https://www.mcs.anl.gov/papers/P5137-0514.pdf) | Related work, architecture and implementation, then correctness, benchmarks, and applications | Establish what remoting already achieves before describing rGPU; do not lead with a compatibility checklist |
| [Cricket](https://doi.org/10.1002/cpe.6474) | Virtualization layer and checkpoint/restart are separate mechanisms, followed by application and microbenchmark evaluation | Explain live-session reconnect separately from execution; distinguish it explicitly from checkpoint/restart |
| [AvA](https://oscarlab.github.io/papers/ava-asplos20.pdf) | Background and system design precede the specification/compiler details, implementation, and evaluation | Put the boundary choice before code-generation details; treat generated API coverage as an implementation property |
| [DGSF](https://www.cs.utexas.edu/~witchel/pubs/fingler23osr-dgsf.pdf) | Deployment challenges motivate mechanisms; evaluation asks explicit questions about overhead and optimizations | Separate remote execution overhead, resource benefits, and mechanism ablations |
| [DxPU](https://arxiv.org/abs/2310.04648) | Background and motivation, design with a performance model, evaluation, discussion | Use a small latency/throughput model to formulate testable hypotheses before presenting results |
| [Prism](https://www.usenix.org/conference/nsdi25/presentation/yang) | Workload requirements drive design; evaluation separates native, local disaggregation, and remote disaggregation | Adopt the baseline decomposition on one GPU, with the laptop-versus-server comparison in a separate figure |

These are structural lessons, not instructions to imitate each paper's scope. rGPU has no production fleet study, durable checkpoint system, or demonstrated resource scheduler. Adding sections that imply those capabilities would weaken the paper.

## Recommended conference layout

The following is a target allocation for a roughly 12-page main paper, excluding references and artifact material. It is an editorial budget, not a claim about a particular conference's current page limit.

1. **Introduction (1 page).** Begin with the cost of keeping application control on the client. State the tension between compatibility and information available for asynchronous execution. Give bounded contributions and two representative results, including their baselines.
2. **Background and Prior Work (1.5 pages).** Group work by interception layer and objective. Include CUDA remoting, generated virtualization, serverless disaggregation, framework graphs, explicit RPC, and notebook execution. Finish with the precise gap being investigated. Do not repeat this survey in a second long related-work section.
3. **Problem Statement and Performance Model (0.75 page).** Define messages, waits, frames, and bytes. Explain that waits times RTT is a conditional network sensitivity term. State the workload and failure assumptions before readers encounter implementation details.
4. **Design (3 pages).** Start with one architecture diagram showing the independent paths. Then explain metadata prediction, remote identities, ordered streaming, graph shipping, errors, and recovery. These mechanisms should answer the problems just introduced.
5. **Implementation and Supported Semantics (0.75 page).** Record version assumptions, unsupported behavior, library interception, code generation, and security boundaries. Give measured API coverage only when there is a reproducible supported/unsupported inventory.
6. **Evaluation (3.5 pages).** Use questions to organize results: end-to-end overhead; sensitivity to network conditions and host reads; mechanism ablations; graph amortization; compatibility and recovery. Place experimental setup before charts. Include adverse workloads and native/coarse-RPC baselines.
7. **Limitations and Discussion (0.35 page).** Explain what results do not generalize to and what remains outside scope. In the current draft this section is longer because the GPU evaluation is incomplete.
8. **Conclusion (0.15 page).** State the demonstrated lesson, without introducing a stronger performance or novelty claim.

The current draft retains expanded methodology, negative results, and a concrete missing-experiment section. Once the missing GPU evidence is collected, replace that section with actual results and shorten the limitations discussion. A formatting pass should then apply the selected venue's official template.

## Figures and tables that earn their space

The architecture diagram should show where metadata lives and where tensor values live, and make clear that the Python and CUDA protocols are independent. The primary performance chart should use identical GPU hardware and distinguish native, colocated remote, LAN, and WAN execution. A second chart should vary host-observation frequency at controlled RTT. A third should decompose eager, graph-shipped server-eager, and graph-shipped Inductor execution, including startup and break-even. The present artifact supplies the architecture diagram and a CPU mechanism chart with every sample visible.

A compatibility table should report tested operations/workloads, versions, and unsupported outcomes. A recovery table should name the fault location, expected state transition, correctness result, and recovery duration. Performance counters belong beside latency: message count alone cannot show whether the application improved. Avoid a headline bar chart that places published numbers from unrelated systems on a common axis.

## The novelty question to resolve

Remote CUDA, batching, shadow state, graph execution, and semantic disaggregation all have substantial prior art. [Wang et al.](https://arxiv.org/abs/2401.13354) already connect asynchronous requests and shadow state to network requirements. [Genie](https://vincen.tl/files/hong25lost.pdf) directly argues for framework-level semantic disaggregation. The [Djinn OSDI 2026 listing](https://www.usenix.org/conference/osdi26/presentation/hong) identifies a particularly relevant work, but a full technical comparison remains outstanding because only its listing was verified.

The present defensible position is an implementation and characterization of two remoting boundaries for client-resident programs. A stronger conference submission needs a result beyond combining familiar techniques. Possibilities to investigate include a systematic account of which framework semantics remove WAN dependencies, a correctness contract for reconnecting asynchronous tensor execution with bounded state, or a controlled study identifying when operator-level remoting is preferable to whole-program and coarse-RPC execution. These are candidate research claims, not demonstrated contributions in the current draft.

Two same-GPU sessions now measure remoting overhead directly: eager training on an L40S is 8.9x slower than native with a matched optimizer, the client process is CPU-bound while the server is idle about half the time, client and server placement on the host's cores makes no difference, and the network costs about one round trip per wait. That locates the cost in the client's per-operation path, which is also where a contribution beyond characterization could come from: reducing that cost and showing it. That replaces an absence of evidence and argues for the honest positioning rather than against it. In particular, a 2.92x remote-A40-versus-local-MPS ratio does not establish low remoting overhead; a 21.99x graph-versus-eager CPU-loopback ratio does not establish a GPU speedup; and five recorded waits do not prove negligible network cost. The manuscript is strongest when these distinctions are part of its explanation, not hidden in footnotes.
