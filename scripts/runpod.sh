#!/usr/bin/env bash
# Start, stop, inspect or delete the RunPod GPU box used for testing.
#
#   ./scripts/runpod.sh create     # rent one and print the ssh command
#   ./scripts/runpod.sh start      # restart one that was stopped
#   ./scripts/runpod.sh status     # what is running, and how to reach it
#   ./scripts/runpod.sh stop       # release the GPU, keep the disk
#   ./scripts/runpod.sh delete     # destroy it and everything on it
#   ./scripts/runpod.sh delete --yes
#
# Set RUNPOD_API_KEY, or let the script read OP_ITEM from 1Password. A running
# pod bills by the hour; a stopped one still bills for its disk, so delete it
# when you are done for good.
set -euo pipefail

cd "$(dirname "$0")/.."

OP_ITEM=${OP_ITEM:-}
NAME=${NAME:-rgpu-dev}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/rgpu_runpod}
# Prefer the affordable 16-24 GB cards, with A40 as a last fallback. Keeping a
# broad pool avoids creation failures when one GPU family has no capacity.
GPUS=${GPUS:-'["NVIDIA L4","NVIDIA GeForce RTX 4090","NVIDIA GeForce RTX 4080 SUPER","NVIDIA RTX 4000 Ada Generation","NVIDIA RTX A4500","NVIDIA RTX A5000","NVIDIA RTX A4000","NVIDIA GeForce RTX 3090","NVIDIA A40"]'}
# Community cloud has repeatedly had no capacity for these cards, and a create
# that finds none fails without saying why. Secure cloud costs more but is
# actually available; override with CLOUD=COMMUNITY to try the cheap one first.
CLOUD=${CLOUD:-SECURE}
# CUDA 12.8.1 with torch 2.9.1: matches the headers the shims are generated
# from. Includes an ssh server, which a bare nvidia/cuda image does not.
IMAGE=${IMAGE:-runpod/pytorch:1.2.0-rc.162-cu1281-torch291-ubuntu2204}

case "${1:-status}" in
  create|start|status|stop|delete) ;;
  *) echo "usage: $0 {create|start|status|stop|delete}" >&2; exit 1 ;;
esac

if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  RUNPOD_TOKEN=$RUNPOD_API_KEY
elif [[ -z "$OP_ITEM" ]]; then
  echo "set RUNPOD_API_KEY or set OP_ITEM to a 1Password secret reference" >&2
  exit 1
elif ! RUNPOD_TOKEN=$(op read "$OP_ITEM"); then
  echo "set RUNPOD_API_KEY or unlock 1Password, then retry" >&2
  exit 1
fi
if [[ -z "$RUNPOD_TOKEN" ]]; then
  echo "the RunPod API key is empty" >&2
  exit 1
fi

api() {
  local method=$1 path=$2 body=${3:-}
  if [[ -n "$body" ]]; then
    curl -s -X "$method" "https://rest.runpod.io/v1$path" \
      -H "Authorization: Bearer $RUNPOD_TOKEN" -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -s -X "$method" "https://rest.runpod.io/v1$path" \
      -H "Authorization: Bearer $RUNPOD_TOKEN"
  fi
}

# Parses an API response, or explains it if it is not JSON. Every call goes
# through this so an auth failure or a gateway error reads as itself rather
# than as a JSON decode traceback.
parse() {
  python3 -c "
import json,sys
raw = sys.stdin.read()
try:
    doc = json.loads(raw)
except Exception:
    sys.stderr.write('runpod API did not return JSON:\n  %s\n' % raw[:300].strip())
    sys.exit(1)
$1
"
}

all_ids() {
  api GET /pods | parse "print(' '.join(p['id'] for p in doc))"
}

# Pods on the account that this NAME does not match. Printed by stop and
# delete, which would otherwise report success while one of these bills on.
others() {
  api GET /pods | parse "
for p in doc:
    if p.get('name') != '$NAME':
        print('  %s %s %s \$%s/hr' % (p['id'], p.get('name'),
              p.get('desiredStatus'), p.get('costPerHr')))
"
}

pod_id() {
  api GET /pods | parse "
for p in doc:
    if p.get('name') == '$NAME':
        print(p['id']); break
"
}

show() {
  api GET "/pods/$1" | parse "
p = doc
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
    body=$(python3 - "$(cat "$SSH_KEY.pub")" "$NAME" "$IMAGE" "$GPUS" "$CLOUD" <<'PY'
import json, sys
public_key, name, image, gpu_types, cloud = sys.argv[1:]
print(json.dumps({
  "name": name,
  "imageName": image,
  "gpuTypeIds": json.loads(gpu_types),
  "gpuCount": 1,
  "cloudType": cloud,
  "computeType": "GPU",
  "containerDiskInGb": 40,
  "volumeInGb": 0,
  "ports": ["22/tcp"],
  "supportPublicIp": True,
  "interruptible": False,
  # cuFuncGetParamInfo, which the kernel launch path needs, arrived in 12.4.
  "allowedCudaVersions": ["12.4","12.5","12.6","12.7","12.8","12.9"],
  "env": {"PUBLIC_KEY": public_key},
}))
PY
)
    id=$(api POST /pods "$body" | parse "
if not isinstance(doc, dict) or not doc.get('id'):
    sys.stderr.write('runpod create failed:\n  %s\n' % raw[:1000].strip())
    sys.exit(1)
print(doc['id'])
") || id=""
    if [[ -z "$id" ]]; then
      # The request may still have created something. Say so loudly rather
      # than exiting and leaving it to bill unnoticed.
      echo "creation did not return an id. Checking whether one was made anyway:" >&2
      "$0" status >&2
      echo "if a pod is listed above, delete it with: $0 delete" >&2
      exit 1
    fi
    echo "created $id; waiting for ssh"
    for _ in $(seq 1 60); do
      sleep 5
      out=$(show "$id")
      grep -q "^ssh -i" <<<"$out" && { echo "$out"; exit 0; }
    done
    show "$id"
    ;;
  status)
    # Every pod, not just the first match. A create that looked like it failed
    # but did not leaves a second pod with the same name, and showing only one
    # of them is how it goes on billing unnoticed.
    api GET /pods | parse "
if not doc:
    print('no pods: nothing is billing')
for p in doc:
    m = p.get('machine') or {}
    print('%-16s %-12s %-9s \$%s/hr %s' % (p.get('id'), p.get('name'),
          p.get('desiredStatus'), p.get('costPerHr'), m.get('gpuTypeId') or ''))
"
    for one in $(all_ids); do
      echo
      show "$one"
    done
    exit 0
    ;;
  start)
    id=$(pod_id)
    [[ -n "$id" ]] || { echo "no pod named $NAME; use create" >&2; exit 1; }
    api POST "/pods/$id/start" >/dev/null
    echo "starting $id; waiting for ssh"
    for _ in $(seq 1 60); do
      sleep 5
      out=$(show "$id")
      grep -q "^ssh -i" <<<"$out" && { echo "$out"; exit 0; }
    done
    show "$id"
    ;;
  stop)
    id=$(pod_id)
    if [[ -z "$id" ]]; then
      echo "no pod named $NAME"
      rest=$(others)
      [[ -z "$rest" ]] || {
        echo "but these pods ARE on the account and still billing:" >&2
        echo "$rest" >&2
        echo "stop one with: NAME=<its-name> $0 stop" >&2
        exit 1
      }
      exit 0
    fi
    api POST "/pods/$id/stop" >/dev/null
    echo "stopped $id; the GPU is released, the disk still costs a little"
    ;;
  delete)
    # Every pod of this name, for the same reason status lists them all.
    ids=$(api GET /pods | parse "
print(' '.join(p['id'] for p in doc if p.get('name') == '$NAME'))
")
    if [[ -z "$ids" ]]; then
      echo "no pod named $NAME"
      rest=$(others)
      [[ -z "$rest" ]] || {
        echo "but these pods ARE on the account and still billing:" >&2
        echo "$rest" >&2
        echo "delete one with: NAME=<its-name> $0 delete" >&2
        exit 1
      }
      exit 0
    fi
    id=$ids
    # Nothing on a community pod's container disk survives a stop anyway, so
    # deleting usually costs nothing that stopping would have kept.
    if [[ "${2:-}" != "--yes" ]]; then
      read -r -p "Delete pod $id and everything on it? [y/N] " a
      [[ "$a" == "y" || "$a" == "Y" ]] || { echo aborted; exit 1; }
    fi
    for one in $ids; do
      code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
             -H "Authorization: Bearer $RUNPOD_TOKEN" \
             "https://rest.runpod.io/v1/pods/$one")
      echo "deleted $one (http $code)"
    done
    # Confirm rather than assume: a delete that silently failed is exactly the
    # case that leaves something billing.
    sleep 3
    left=$(api GET /pods | parse "print(len(doc))")
    if [[ "$left" == "0" ]]; then
      echo "pods remaining on the account: 0, nothing is billing"
    else
      echo "WARNING: $left pod(s) still on the account, still billing:" >&2
      others >&2
      api GET /pods | parse "
for p in doc:
    if p.get('name') == '$NAME':
        print('  %s %s %s' % (p['id'], p.get('name'), p.get('desiredStatus')))
" >&2
      exit 1
    fi
    ;;
esac
