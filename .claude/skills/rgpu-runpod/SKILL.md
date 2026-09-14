---
name: rgpu-runpod
description: Rent, use and release a RunPod GPU for rgpu work without wasting money. Use whenever a task needs a real NVIDIA GPU - running the PyTorch ladder, testing a kernel launch, checking a driver behaviour - or when asked to start, stop, check or delete the test pod. Covers the cost rules, the disk-is-wiped-on-stop trap, and the deploy and verify loop.
---

# Renting a GPU for rgpu

A GPU costs money by the hour and this project only needs one in short bursts.
The whole point of this skill is that the pod is up for minutes, not hours.

## The cost rules

| State | GPU charge | Disk charge | What survives |
|---|---|---|---|
| RUNNING | yes, twenty cents or so per hour | yes | everything |
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
stopped, since a stopped pod still bills for its disk. Given that stopping
preserves nothing anyway, deleting is usually strictly better.

**Confirm, do not assume.** A delete that printed success is not evidence that
nothing is billing. Finish by listing every pod on the account and seeing it
empty. This has already gone wrong once here: a pod billed unnoticed for forty
minutes because nothing ever looked at the whole list.

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

**Name the pod `rgpu-dev`, which is what `NAME` already defaults to.**
Every subcommand in `scripts/runpod.sh` - `status`, `start`, `stop`, `delete` -
finds the pod by matching `NAME`. Creating one as `rgpu-yangpt` wasted five minutes here: the wait loop
watched for the ssh block, which `status` only printed for a name match, so a
pod that was up and healthy looked like one that never started. If you do
create a pod under another name, put `NAME=that-name` in front of every later
command, including the delete.

**Always run `status` first.** It is one call, it costs nothing, and it stops
you renting a second pod while one is already running.

**What the `NAME` filter does and does not cover.** `status`'s summary table
is unfiltered and always lists every pod. `start`, `stop` and `delete` are
not: they match on `NAME`. Before this was fixed, `stop` and `delete` printed
"no pod named rgpu-dev" and exited **0** - success - while a differently named
pod went on billing. They now list any such pod and exit non-zero, and
`status` prints the ssh block for every pod rather than only a name match.

To see the whole account without going through the script at all:

```sh
curl -s https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $(op read 'op://YOUR_VAULT/Runpod/api-key')" |
  python3 -c "import json,sys
for p in json.load(sys.stdin): print(p['id'], p['name'], p.get('desiredStatus'), p.get('costPerHr'))"
```

Finish every session with that call, not with `status`.

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

The ladder needs no settings now: cuBLAS, cuBLASLt and cuDNN are all
forwarded. `RGPU_NO_CUDNN=1` falls back to PyTorch's own convolution kernels
and `RGPU_NO_CUBLASLT=1` to plain cuBLAS, which is how to tell a library
problem apart from a problem underneath it. Batching is on by default;
`RGPU_BATCH=0` turns it off, which is worth doing as a control when a result
looks wrong, since it makes every call report itself where it happened.

Then run the ladder. The shims have to go in front of PyTorch's own CUDA
libraries, and `LD_LIBRARY_PATH` will not do it: PyTorch's libraries carry
`RPATH`, which the loader consults first, and PyTorch also preloads the CUDA
libraries by absolute path. `LD_PRELOAD` wins over both.

```sh
NV=$(echo $PWD/venv/lib/python3*/site-packages/nvidia)   # version varies
RGPU_CUBLAS=$NV/cublas/lib/libcublas.so.12 \
RGPU_CUBLASLT=$NV/cublas/lib/libcublasLt.so.12 \
RGPU_CUDNN=$NV/cudnn/lib/libcudnn.so.9 \
  ./build/rgpu-server > server.log 2>&1 &

LD_PRELOAD=$PWD/build/libcudart.so.12:$PWD/build/libcublas.so.12:$PWD/build/libcublasLt.so.12:$PWD/build/libcudnn.so.9:$PWD/build/libcuda.so.1 \
RGPU_SERVER=127.0.0.1:9713 \
  ./venv/bin/python ladder.py
```

Then delete the pod, and confirm the pod list is empty.

## The op-level backend needs torch ON THE POD

There are two backends and they put PyTorch in opposite places. Getting this
wrong wastes a pod's worth of time.

| Backend | Port | Where torch runs | What the pod needs |
|---|---|---|---|
| Driver-API shim (`device="cuda"`) | 9713 | the client | `rgpu-server`, no torch |
| Op-level device (`device="rgpu"`) | 9720 | **the server** | a matching torch |

The op-level path ships ATen operations, not driver calls, so `rgpu-opserver`
is a PyTorch process and the client version-checks it on handshake. Note the
client cannot be macOS for the driver-API path: it needs a CUDA-linked torch,
which is why that client is a linux container.

**The version bind.** `pyproject.toml` declares `torch>=2.14`, which the
*client* really does need for PrivateUse1 device registration. But there is no
`2.14+cu128` wheel, and the stock pod image's driver is 12.8 (`570.x`), so
`2.14+cu130` installs happily and then reports `cuda False` with "driver is too
old". What works today: `torch==2.11.0+cu128` on the pod from the cu128 index,
and `RGPU_ALLOW_VERSION_MISMATCH=1` on the client. The server only executes
ops, so the older torch is fine there.

**Never pip-install one CUDA flavour over another.** Installing cu128 on top of
cu130 leaves a half-replaced tree that imports as
`libtorch_nvshmem.so: undefined symbol`. Delete the venv and build it again
rather than trying to correct it in place.

**The image name lies about torch.** `runpod/pytorch:...-torch291-...` shipped
with no torch module at all, and no conda - only system `python3.9` through
`python3.13`. Budget for installing torch every time; it is several minutes.

**Getting it running**, heeding the multi-line-ssh warning below:

```sh
tar czf rgpu_pkg.tgz -C python rgpu && scp -P PORT rgpu_pkg.tgz root@HOST:/workspace/
# then a scp'd script, not an inline heredoc:
#   PYTHONPATH=/workspace nohup venv/bin/python -m rgpu.server \
#     --device cuda --bind 127.0.0.1 --port 9720 > opserver.log 2>&1 &
ssh -N -L 9721:127.0.0.1:9720 root@HOST -p PORT      # never expose 9720
RGPU_OPSERVER=127.0.0.1:9721 RGPU_ALLOW_VERSION_MISMATCH=1 python train.py
```

Measured this way from a Mac to an A40 in EU-SE-1, RTT 116 ms: nanoGPT
(10.8M params, batch 64, block 256) trained at 1.04 s/iter, against 3.04 s/iter
on local MPS. 139,931 messages but only **5 waits** for 20 iterations - the
queue is what makes a transatlantic GPU beat a local one, so if a workload ever
looks latency-bound, count waits first.

## Things that will bite you

**A create that looks like it failed may have worked.** The API can return an
error, or something that is not JSON at all, after having created the pod. If
`create` does not print an id, do not assume nothing happened: run `status` and
delete whatever is there. This is the forty-minute mistake above, and it is why
`create` now points at `status` instead of just exiting.

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
`/usr/local/cuda/bin/nvcc`. The vector-add test kernel needs it, and so does
the benchmark if you want launch numbers: a real driver rejects the synthetic
module image the fake server accepts, so pass a real fatbin.

```sh
NVCC=/usr/local/cuda/bin/nvcc ./scripts/build_fatbin.sh
./build/bench 2000 build/vecadd.fatbin      # launch numbers need this argument
```

**`create` asks for COMMUNITY cloud, which has repeatedly had no capacity.**
`scripts/runpod.sh create` hardcodes `"cloudType": "COMMUNITY"` and fails with
"creation did not return an id". Secure cloud works: post the same body with
`"cloudType": "SECURE"` and an A40 in `gpuTypeIds`, keeping
`"name": "rgpu-dev"` so the script's other subcommands can still find it.

**A pod that stays RUNNING but never gets a machine means the account is out
of money.** It does not say so. `status` shows RUNNING, and the API shows no
`publicIp`, no `portMappings` and `runtime` false, which looks exactly like a
slow start. The message only appears if you try to create one by hand:

```sh
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d @body.json https://rest.runpod.io/v1/pods
# {"error":"create pod: Your account balance is too low to rent a pod. ..."}
```

`create` in the script reports this as "creation did not return an id", so
that message means check the balance before suspecting the platform. There is
no balance endpoint in the REST v1 API; the console is the place to look, and
adding funds is the only fix. Three pods in a row did this on 2026-09-09 and
were blamed on placement; the real cause was a pod left running overnight
after a failed delete, which drained the account.

**A new pod reuses an old pod's address.** RunPod hands out host and port
pairs from a pool, so ssh refuses with "Host key verification failed" on a pod
that is perfectly healthy. Drop the stale entry and take the new key:

```sh
ssh-keygen -R "[HOST]:PORT"
ssh-keyscan -p PORT -H HOST >> ~/.ssh/known_hosts
```

**Check the shim's symbols against the real library before running anything.**
A missing one is an import failure with an unhelpful message, and it costs one
command to know. The host's library is in the venv, not the system:

```sh
nm -D --defined-only $NV/cudnn/lib/libcudnn.so.9 | awk '$2 ~ /[TWi]/ {print $3}' \
  | sed 's/@.*//' | sort -u > /tmp/real.txt
nm -D --undefined-only venv/lib/python3*/site-packages/torch/lib/libtorch_cuda.so \
  | awk '{print $2}' | sed 's/@.*//' | grep '^cudnn' | sort -u > /tmp/needed.txt
```

Strip the `@version` suffix on both sides or every symbol looks missing: nm
spells a definition `foo@@lib` and a reference `foo@lib`. This found
`cudnnGetLastErrorString`, which torch links by name and which the stub
generator had skipped because it does not return a status.

**Two things on the laptop side fail quietly.** Docker Desktop stops running
and `scripts/build_client.sh` then fails with a daemon socket error; reopen it
and wait. The 1Password session times out and `op read` prints an `[ERROR]`
string to stdout rather than failing, which reaches the API as a malformed
token; if a runpod command behaves strangely, check the key looks like a key
before debugging anything else. This is worst when it happens on the way out:
a delete that fails this way leaves the pod billing, so always re-check the
pod list after any command that printed "Malformed Bearer token". Two accounts
are configured on this machine, so the signin has to name one, and a bare
`op signin` fails with "multiple accounts found":

```sh
op signin --account my.1password.com   # the Personal vault lives here
```

## Choosing hardware

An RTX A4000 or A5000 at around twenty cents an hour is more than enough. Both
are Ampere, `sm_86`, which the test fatbin already targets. Ask for several
types so the request is filled from whatever the community cloud has.

The driver on these boxes is often older than the CUDA headers the shims are
generated from. That is handled: the server resolves driver entry points on
first use, so one the driver lacks costs a single unsupported call rather than
stopping the server from loading.
