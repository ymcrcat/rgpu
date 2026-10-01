"""Follow-up experiments to gpu_benchmark.py, answering the external review.

    --exp optimizer    native only: AdamW default against foreach=False, and
                       which implementation each device's parameters select
    --exp train        rGPU against a server started elsewhere (so it can be
                       pinned and delayed): eager, loss read each step, eager
                       with foreach forced, and the graph-shipped step. Records
                       client CPU time and, with --server-pid, server CPU time
    --exp correctness  several deterministic AdamW steps from one archived
                       initial state, comparing loss, first gradients, final
                       parameters and AdamW moments against native

Native CUDA and rGPU never train in one process: with the rgpu device
registered, a native CUDA backward() fails an assertion in PyTorch's autograd
engine (engine.cpp, opt_ready_stream). So --exp optimizer takes no --server,
and --exp correctness runs twice, the native run saving its reference with
--save-ref for the rGPU run to --load-ref.

The server is never started here. gpu_benchmark.py is left as it was, since
the published JSONs came from it; the model is imported from it.
"""
import argparse
import json
import os
import platform
import random
import statistics
import sys
import time

import torch
from torch.optim.optimizer import _default_to_fused_or_foreach

import gpu_benchmark as gb

LR = 3e-3


def optimizer(params, kind):
    kw = {"default": {}, "single": {"foreach": False, "fused": False},
          "foreach": {"foreach": True}}[kind]
    return torch.optim.AdamW(params, lr=LR, **kw)


def selected(model):
    """What AdamW picks for these parameters when nothing is specified."""
    fused, foreach = _default_to_fused_or_foreach(list(model.parameters()), False, False)
    return {"fused": bool(fused), "foreach": bool(foreach),
            "parameter_type": type(next(model.parameters())).__name__}


def server_cpu_ms(pid):
    if not pid:
        return None
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) * 1000 / os.sysconf("SC_CLK_TCK")


def batch():
    g = torch.Generator().manual_seed(1)
    return (torch.randint(0, gb.VOCAB, (gb.BATCH, gb.BLOCK), generator=g),
            torch.randint(0, gb.VOCAB, (gb.BATCH, gb.BLOCK), generator=g))


def timed_train(dev, cases, args, barrier, stats, graph_backend=None):
    xb, yb = batch()
    rng, rows = random.Random(0), []
    for trial in range(args.trials):
        order = cases[:]
        rng.shuffle(order)
        for name in order:
            torch.manual_seed(3)
            if graph_backend:
                torch._dynamo.reset()
            model = gb.GPT(dropout=0.2).to(dev)
            kind = {"native_default": "default", "native_single": "single",
                    "eager_foreach": "foreach"}.get(name, "default")
            opt = optimizer(model.parameters(), kind)
            xd, yd = xb.to(dev), yb.to(dev)
            fwd = model
            if name == "graph_eager":
                fwd = torch.compile(model, backend=graph_backend, dynamic=False)

            def step():
                loss = fwd(xd, yd)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                return loss

            barrier()
            t0 = time.perf_counter_ns()
            for _ in range(2):
                step()
            barrier()
            warm_ms = (time.perf_counter_ns() - t0) / 1e6   # includes compilation
            before, c0, s0 = stats(), time.process_time(), server_cpu_ms(args.server_pid)
            t0 = time.perf_counter_ns()
            for _ in range(args.train_steps):
                loss = step()
                if name == "eager_loss_read":
                    loss.item()
            barrier()
            ms = (time.perf_counter_ns() - t0) / 1e6
            after, c1, s1 = stats(), time.process_time(), server_cpu_ms(args.server_pid)
            final = loss.item()
            assert final == final and final < 10, f"loss diverged: {final}"
            rows.append({"case": name, "trial": trial, "elapsed_ms": ms,
                         "warmup_2_steps_ms": warm_ms, "final_loss": final,
                         "client_cpu_ms": (c1 - c0) * 1e3,
                         "server_cpu_ms": None if s0 is None else s1 - s0,
                         **{k: after[k] - before[k] for k in before}})
            del model, opt, xd, yd, fwd
    return rows


def summarize(rows):
    out = {}
    for name in sorted({r["case"] for r in rows}):
        sel = [r for r in rows if r["case"] == name]
        med = lambda k: (None if sel[0][k] is None
                         else statistics.median(r[k] for r in sel))
        t = [r["elapsed_ms"] for r in sel]
        out[name] = {"median_ms": statistics.median(t), "min_ms": min(t), "max_ms": max(t),
                     **{k: med(k) for k in sel[0] if k not in
                        ("case", "trial", "elapsed_ms", "final_loss")}}
    return out


def correctness(args, rgpu_mod):
    """Every variant starts from the same state and takes the same steps."""
    xb, yb = batch()
    torch.manual_seed(2)
    state = gb.GPT(dropout=0.0).state_dict()
    variants = [("cpu_single", "cpu", "single", False)]
    if args.device != "cpu" and not rgpu_mod:
        variants += [("native_single", args.device, "single", False),
                     ("native_default", args.device, "default", False)]
    if rgpu_mod:
        variants += [("rgpu_eager", "rgpu", "default", False),
                     ("rgpu_graph", "rgpu", "default", True)]
    got = {}
    for name, dev, kind, graph in variants:
        torch._dynamo.reset()
        model = gb.GPT(dropout=0.0)
        model.load_state_dict(state)
        model = model.to(dev)
        opt = optimizer(model.parameters(), kind)
        fwd = (torch.compile(model, backend=rgpu_mod.compile_backend(compiler="eager"),
                             dynamic=False) if graph else model)
        xd, yd = xb.to(dev), yb.to(dev)
        losses, grads = [], None
        for i in range(args.check_steps):
            loss = fwd(xd, yd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if i == 0:
                grads = [p.grad.detach().cpu().double() for p in model.parameters()]
            opt.step()
            losses.append(loss.item())
        moments = [(opt.state[p]["exp_avg"].cpu().double(),
                    opt.state[p]["exp_avg_sq"].cpu().double()) for p in model.parameters()]
        got[name] = {"losses": losses, "grads": grads, "moments": moments,
                     "params": [p.detach().cpu().double() for p in model.parameters()]}
        del model, opt, fwd

    ref_name = "native_single" if "native_single" in got else "cpu_single"
    ref = got[ref_name]
    if args.save_ref:
        torch.save({"name": ref_name, "ref": ref}, args.save_ref)
    if args.load_ref:
        loaded = torch.load(args.load_ref)
        ref_name, ref = loaded["name"] + " (loaded)", loaded["ref"]

    def worst(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    report = {}
    for name, g in got.items():
        report[name] = {
            "losses": g["losses"],
            "max_abs_loss_diff": max(abs(a - b) for a, b in zip(g["losses"], ref["losses"])),
            "max_abs_first_grad_diff": worst(g["grads"], ref["grads"]),
            "max_abs_param_diff": worst(g["params"], ref["params"]),
            "max_abs_exp_avg_diff": worst([m[0] for m in g["moments"]],
                                          [m[0] for m in ref["moments"]]),
            "max_abs_exp_avg_sq_diff": worst([m[1] for m in g["moments"]],
                                             [m[1] for m in ref["moments"]])}
    # The reference has to have moved, or agreeing with it proves nothing.
    start = gb.GPT(dropout=0.0)
    start.load_state_dict(state)
    moved = worst(ref["params"], [p.detach().double() for p in start.parameters()])
    grad_scale = max(float(g.abs().max()) for g in ref["grads"])
    assert len(set(ref["losses"])) > 1 and grad_scale > 0 and moved > 0
    # Only rGPU is held to the tolerance, and only against the same kernels on
    # the same device. CPU against CUDA is reported, not asserted: Adam divides
    # by the root of the second moment, so where a gradient is near zero a
    # rounding difference flips the sign of a full lr-sized step.
    for name, r in report.items():
        if not name.startswith("rgpu"):
            continue
        assert r["max_abs_loss_diff"] < args.tolerance, (name, r)
        assert r["max_abs_param_diff"] < args.tolerance, (name, r)
        assert r["max_abs_first_grad_diff"] < args.tolerance, (name, r)
    return {"reference": ref_name, "steps": args.check_steps, "tolerance": args.tolerance,
            "reference_max_abs_first_grad": grad_scale,
            "reference_max_abs_param_change": moved, "variants": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["optimizer", "train", "correctness"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--server", help="host:port of a running rgpu server")
    ap.add_argument("--server-pid", type=int, help="its pid, when it is on this host")
    ap.add_argument("--label", default="", help="recorded as given: pinning, delay")
    ap.add_argument("--cases", default="eager,eager_loss_read,eager_foreach,graph_eager")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--train-steps", type=int, default=10)
    ap.add_argument("--check-steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--save-ref")
    ap.add_argument("--load-ref")
    ap.add_argument("--smoke", action="store_true", help="a tiny model, to test the script")
    args = ap.parse_args()
    if args.smoke:
        gb.BATCH, gb.BLOCK, gb.N_EMBD, gb.N_HEAD, gb.N_LAYER = 4, 32, 48, 2, 1

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rgpu_mod = None
    if args.server:
        os.environ["RGPU_OPSERVER"] = args.server
        import rgpu as rgpu_mod

    env = {"exp": args.exp, "label": args.label, "device": args.device,
           "server": args.server, "platform": platform.platform(),
           "python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(),
           "affinity": (sorted(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity") else None),
           "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
           "trials": args.trials, "train_steps": args.train_steps, "smoke": args.smoke}

    none = lambda: {"messages": 0, "waits": 0, "bytes_out": 0, "bytes_in": 0}
    if args.exp == "optimizer":
        barrier = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
        assert not rgpu_mod, "native training and the rgpu device do not share a process"
        env["adamw_selects"] = selected(gb.GPT().to(args.device))
        rows = timed_train(args.device, ["native_default", "native_single"], args,
                           barrier, none)
        out = {"environment": env, "summary": summarize(rows), "samples": rows}
    elif args.exp == "train":
        assert rgpu_mod, "--exp train needs --server"
        env["adamw_selects"] = selected(gb.GPT().to("rgpu"))
        rows = timed_train("rgpu", args.cases.split(","), args, torch.rgpu.synchronize,
                           rgpu_mod.stats, rgpu_mod.compile_backend(compiler="eager"))
        out = {"environment": env, "summary": summarize(rows), "samples": rows}
    else:
        out = {"environment": env, "result": correctness(args, rgpu_mod)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
