---
name: rcuda-runpod
description: Rent, use and release a RunPod GPU for rgpu work without wasting money. Use whenever a task needs a real NVIDIA GPU - running the PyTorch ladder, testing a kernel launch, checking a driver behaviour - or when asked to start, stop, check or delete the test pod. Covers the cost rules, the disk-is-wiped-on-stop trap, and the deploy and verify loop.
---

# Renting a GPU for rgpu

A GPU costs money by the hour and this project only needs one in short bursts.
The whole point of this skill is that the pod is up for minutes, not hours.

## The cost rules

| State | GPU charge | Disk charge | What survives |
|---|---|---|---|
| RUNNING | yes, ~$0.17/hr | yes | everything |
| EXITED | **no** | yes, small | **nothing on the container disk** |
| deleted | no | no | nothing |

Two things follow, and both are easy to get wrong:

**Stopping does not preserve your work.** A community-cloud pod's container
disk is wiped between stop and start. The PyTorch install, roughly seven
gigabytes and ten minutes, has to be redone every time. Stopping is still right
when the next use is hours away, because the GPU charge dwarfs the disk charge,
but do not stop it to "save money" during a five minute pause.

**So batch the work.** Decide everything you want to check before starting the
pod, then start it, run all of it, and stop it. A stopped pod that you restart
three times costs three PyTorch installs.

If the pod is genuinely finished with, delete it rather than leaving it
stopped, since a stopped pod still bills for its disk.

## Commands

```sh
./scripts/runpod.sh status    # is anything billing right now
./scripts/runpod.sh create    # rent a new one
./scripts/runpod.sh start     # restart a stopped one, disk will be empty
./scripts/runpod.sh stop      # release the GPU
./scripts/runpod.sh delete    # destroy it, stops the disk charge too
```

The API key comes from 1Password at `op://YOUR_VAULT/Runpod/api-key`. If a
command fails with an authorization timeout the vault has locked; the user has
to unlock it, and `! op signin` in the session is the quickest way to ask.

**Always run `status` first.** It is one call, it costs nothing, and it stops
you renting a second pod while one is already running.

## The loop

Once the pod is up, `status` prints the ssh and deploy commands with the
current address. The port changes on every start.

```sh
./scripts/deploy_server.sh root@HOST -i ~/.ssh/rgpu_runpod -p PORT
```

That copies the source, builds the server and the shims on the host, and
reports what it built. Then, on the host, PyTorch needs installing into
`~/rgpu/venv` if the disk was wiped:

```sh
python3 -m venv venv && ./venv/bin/pip install -q torch==2.9.1 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

Then run the ladder. The shims have to go in front of PyTorch's own CUDA
libraries, and `LD_LIBRARY_PATH` will not do it: PyTorch's libraries carry
`RPATH`, which the loader consults first, and PyTorch also preloads the CUDA
libraries by absolute path. `LD_PRELOAD` wins over both.

```sh
NV=$PWD/venv/lib/python3.10/site-packages/nvidia
RGPU_CUBLAS=$NV/cublas/lib/libcublas.so.12 \
RGPU_CUBLASLT=$NV/cublas/lib/libcublasLt.so.12 \
  ./build/rgpu-server > server.log 2>&1 &

LD_PRELOAD=$PWD/build/libcudart.so.12:$PWD/build/libcublas.so.12:$PWD/build/libcublasLt.so.12:$PWD/build/libcuda.so.1 \
RGPU_SERVER=127.0.0.1:9713 \
  ./venv/bin/python ladder.py
```

Then stop the pod.

## Things that will bite you

**Multi-line ssh commands get lost.** Long inline scripts over ssh have come
back empty in this project. Write the script to a file, `scp` it, and run it
with `bash /root/name.sh`.

**Do not ship host-specific binaries.** `third_party/cudart` and
`third_party/nvcc` hold binaries for the client's architecture. On an x86_64
host the linker rejects an aarch64 library with "file in wrong format" and the
build fails partway, leaving some targets built and others missing. The deploy
script already excludes them.

**A partly failed build looks like a working one.** If some shims are missing
from `build/`, the run fails with "cannot be preloaded" rather than anything
about the build. Check `ls build/*.so*` before blaming the runtime.

**nvcc may not be on PATH** even when it is installed, commonly at
`/usr/local/cuda/bin/nvcc`. Only the vector-add test kernel needs it.

## Choosing hardware

An RTX A4000 or A5000 at around twenty cents an hour is more than enough. Both
are Ampere, `sm_86`, which the test fatbin already targets. Ask for several
types so the request is filled from whatever the community cloud has.

The driver on these boxes is often older than the CUDA headers the shims are
generated from. That is handled: the server resolves driver entry points on
first use, so one the driver lacks costs a single unsupported call rather than
stopping the server from loading.
