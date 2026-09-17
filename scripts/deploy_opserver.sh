#!/usr/bin/env bash
# Copy the Python backend to a GPU host and start rgpu-opserver there.
#
#   ./scripts/deploy_opserver.sh user@host -p 2222 -i ~/.ssh/key
#   ./scripts/deploy_opserver.sh user@host --torch-version 2.11 -p 2222
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  sed -n '2,5p' "$0"
  exit 0
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@host [--torch-version MAJOR.MINOR] [ssh options...]" >&2
  exit 2
fi

target=$1
shift
torch_version=
ssh_opts=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch-version)
      [[ $# -ge 2 ]] || { echo "--torch-version needs MAJOR.MINOR" >&2; exit 2; }
      torch_version=$2
      shift 2
      ;;
    *)
      ssh_opts+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$torch_version" ]]; then
  torch_version=$(python3 -c \
    'import torch; print(".".join(torch.__version__.split(".")[:2]))' 2>/dev/null) || {
      echo "cannot detect local Torch; activate the client venv or pass --torch-version" >&2
      exit 2
    }
fi
[[ "$torch_version" =~ ^[0-9]+\.[0-9]+$ ]] || {
  echo "invalid Torch version '$torch_version'; expected MAJOR.MINOR" >&2
  exit 2
}

echo "copying the Python backend to $target:rgpu"
tar czf - \
  --exclude='__pycache__' --exclude='*.pyc' \
  python/pyproject.toml python/rgpu python/rgpu_run.py scripts/opserver_pod.sh \
  | ssh "${ssh_opts[@]}" "$target" 'mkdir -p rgpu && tar xzf - -C rgpu'

echo "starting rgpu-opserver with Torch $torch_version"
ssh "${ssh_opts[@]}" "$target" \
  "cd rgpu && bash scripts/opserver_pod.sh '$torch_version'"
