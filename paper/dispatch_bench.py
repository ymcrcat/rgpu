# Client-side dispatch speed: how fast this CPU runs the PyTorch Python/C++
# dispatch path, independent of any device. Used to explain why a faster client
# can beat a colocated one when the client is the bottleneck.
import time, torch, platform, statistics
torch.set_num_threads(1)
a = torch.ones(1); b = torch.ones(1)
def us_per(fn, n=20000):
    out = []
    for _ in range(5):
        t = time.perf_counter_ns()
        for _ in range(n): fn()
        out.append((time.perf_counter_ns() - t) / n / 1e3)
    return statistics.median(out)
print("host", platform.machine(), platform.processor() or "", "| torch", torch.__version__)
print("torch.add(1-elem cpu)  %.2f us/op" % us_per(lambda: torch.add(a, b)))
print("python loop            %.3f us/iter" % us_per(lambda: None))
