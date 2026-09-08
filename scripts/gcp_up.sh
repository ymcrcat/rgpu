#!/usr/bin/env bash
# Create (or start) the GPU host that runs rgpu-server.
#
#   ./scripts/gcp_up.sh              # create if absent, then start
#   GPU=nvidia-tesla-t4 MACHINE=n1-standard-4 ./scripts/gcp_up.sh
#
# The instance bills while it is RUNNING. Stop it with ./scripts/gcp_down.sh.
set -euo pipefail

NAME=${NAME:-rgpu-gpu}
ZONE=${ZONE:-us-central1-a}
# L4 is Ada (sm_89) and is the cheapest current-generation option. A T4
# (sm_75) is cheaper still and works for everything here.
MACHINE=${MACHINE:-g2-standard-4}
GPU=${GPU:-nvidia-l4}
GPU_COUNT=${GPU_COUNT:-1}
DISK=${DISK:-100GB}
# The Deep Learning VM images ship the NVIDIA driver and CUDA already, which
# saves a long and failure-prone driver install.
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu124-ubuntu-2204-py310}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
  echo "No GCP project set. Pick one:" >&2
  gcloud projects list --format="value(projectId,name)" >&2
  echo >&2
  echo "Then: gcloud config set project PROJECT_ID" >&2
  exit 1
fi
echo "project: $PROJECT  zone: $ZONE  machine: $MACHINE  gpu: ${GPU_COUNT}x${GPU}"

if gcloud compute instances describe "$NAME" --zone "$ZONE" \
     --project "$PROJECT" >/dev/null 2>&1; then
  echo "instance $NAME exists; starting it"
  gcloud compute instances start "$NAME" --zone "$ZONE" --project "$PROJECT"
else
  echo "creating $NAME ..."
  # A preemptible/spot instance is a fraction of the price and this workload
  # tolerates being interrupted: nothing here is long-running yet.
  gcloud compute instances create "$NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --machine-type "$MACHINE" \
    --accelerator "type=${GPU},count=${GPU_COUNT}" \
    --image-family "$IMAGE_FAMILY" \
    --image-project "$IMAGE_PROJECT" \
    --boot-disk-size "$DISK" \
    --boot-disk-type pd-balanced \
    --maintenance-policy TERMINATE \
    --provisioning-model SPOT \
    --instance-termination-action STOP \
    --metadata install-nvidia-driver=True \
    --scopes cloud-platform
fi

EXT_IP=$(gcloud compute instances describe "$NAME" --zone "$ZONE" \
  --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo
echo "instance is up: $EXT_IP"
echo
echo "The rgpu protocol has no authentication, so do not open port 9713 to the"
echo "internet. Reach it over an SSH tunnel instead:"
echo
echo "  gcloud compute ssh $NAME --zone $ZONE --project $PROJECT -- -L 9713:localhost:9713"
echo
echo "then point the client at RGPU_SERVER=host.docker.internal:9713"
echo
echo "Remember to stop it when you are done: ./scripts/gcp_down.sh"
