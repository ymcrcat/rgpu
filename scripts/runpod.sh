#!/usr/bin/env bash
# Start, stop, inspect or delete the RunPod GPU box used for testing.
#
#   ./scripts/runpod.sh create     # rent one and print the ssh command
#   ./scripts/runpod.sh status     # what is running, and how to reach it
#   ./scripts/runpod.sh stop       # release the GPU, keep the disk
#   ./scripts/runpod.sh delete     # destroy it and everything on it
#
# The API key comes from 1Password, so it never lands in a file or the shell
# history. A running pod bills by the hour; a stopped one still bills for its
# disk, so delete it when you are done for good.
set -euo pipefail

cd "$(dirname "$0")/.."

OP_ITEM=${OP_ITEM:-op://YOUR_VAULT/Runpod/api-key}
NAME=${NAME:-rgpu-dev}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/rgpu_runpod}
# An RTX A4000 or A5000 is around twenty cents an hour and has more than enough
# memory for this. Ampere is sm_86, which the test fatbin already targets.
GPUS=${GPUS:-'["NVIDIA RTX A5000","NVIDIA RTX A4000","NVIDIA GeForce RTX 3090"]'}
# CUDA 12.8.1 with torch 2.9.1: matches the headers the shims are generated
# from. Includes an ssh server, which a bare nvidia/cuda image does not.
IMAGE=${IMAGE:-runpod/pytorch:1.2.0-rc.162-cu1281-torch291-ubuntu2204}

key() { op read "$OP_ITEM"; }

api() {
  local method=$1 path=$2 body=${3:-}
  if [[ -n "$body" ]]; then
    curl -s -X "$method" "https://rest.runpod.io/v1$path" \
      -H "Authorization: Bearer $(key)" -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -s -X "$method" "https://rest.runpod.io/v1$path" \
      -H "Authorization: Bearer $(key)"
  fi
}

pod_id() {
  api GET /pods | python3 -c "
import json,sys
for p in json.load(sys.stdin):
    if p.get('name') == '$NAME':
        print(p['id']); break
"
}

show() {
  api GET "/pods/$1" | python3 -c "
import json,sys
p = json.load(sys.stdin)
m = p.get('machine') or {}
print('id        ', p['id'])
print('status    ', p.get('desiredStatus'))
print('gpu       ', m.get('gpuTypeId'), 'in', m.get('location'))
print('cost      ', p.get('costPerHr'), 'per hour')
ip = p.get('publicIp'); port = (p.get('portMappings') or {}).get('22')
if ip and port:
    print()
    print('ssh -i ${SSH_KEY} -p %s root@%s' % (port, ip))
    print()
    print('deploy:  ./scripts/deploy_server.sh root@%s -i ${SSH_KEY} -p %s' % (ip, port))
else:
    print('ssh       not ready yet; run status again in a moment')
"
}

case "${1:-status}" in
  create)
    if [[ ! -f "$SSH_KEY.pub" ]]; then
      echo "creating an ssh key at $SSH_KEY"
      ssh-keygen -t ed25519 -N "" -C "rgpu-runpod" -f "$SSH_KEY"
    fi
    body=$(python3 - "$(cat "$SSH_KEY.pub")" <<PY
import json, sys
print(json.dumps({
  "name": "$NAME",
  "imageName": "$IMAGE",
  "gpuTypeIds": json.loads('$GPUS'),
  "gpuCount": 1,
  "cloudType": "COMMUNITY",
  "computeType": "GPU",
  "containerDiskInGb": 40,
  "volumeInGb": 0,
  "ports": ["22/tcp"],
  "supportPublicIp": True,
  "interruptible": False,
  # cuFuncGetParamInfo, which the kernel launch path needs, arrived in 12.4.
  "allowedCudaVersions": ["12.4","12.5","12.6","12.7","12.8","12.9"],
  "env": {"PUBLIC_KEY": sys.argv[1]},
}))
PY
)
    id=$(api POST /pods "$body" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))")
    [[ -n "$id" ]] || { echo "creation failed" >&2; exit 1; }
    echo "created $id; waiting for ssh"
    for _ in $(seq 1 60); do
      sleep 5
      out=$(show "$id")
      grep -q "^ssh -i" <<<"$out" && { echo "$out"; exit 0; }
    done
    show "$id"
    ;;
  status)
    id=$(pod_id)
    [[ -n "$id" ]] || { echo "no pod named $NAME"; exit 0; }
    show "$id"
    ;;
  stop)
    id=$(pod_id)
    [[ -n "$id" ]] || { echo "no pod named $NAME"; exit 0; }
    api POST "/pods/$id/stop" >/dev/null
    echo "stopped $id; the GPU is released, the disk still costs a little"
    ;;
  delete)
    id=$(pod_id)
    [[ -n "$id" ]] || { echo "no pod named $NAME"; exit 0; }
    read -r -p "Delete pod $id and everything on it? [y/N] " a
    [[ "$a" == "y" || "$a" == "Y" ]] || { echo aborted; exit 1; }
    api DELETE "/pods/$id" >/dev/null
    echo "deleted $id"
    ;;
  *)
    echo "usage: $0 {create|status|stop|delete}" >&2
    exit 1
    ;;
esac
