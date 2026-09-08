#!/usr/bin/env bash
# Stop the GPU host so it stops billing. Add --delete to remove it entirely.
#
#   ./scripts/gcp_down.sh
#   ./scripts/gcp_down.sh --delete
set -euo pipefail

NAME=${NAME:-rgpu-gpu}
ZONE=${ZONE:-us-central1-a}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}

if [[ "${1:-}" == "--delete" ]]; then
  # Deleting destroys the boot disk and everything on it.
  read -r -p "Delete instance $NAME and its disk? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted"; exit 1; }
  gcloud compute instances delete "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet
  echo "deleted"
else
  gcloud compute instances stop "$NAME" --zone "$ZONE" --project "$PROJECT"
  echo "stopped; the boot disk still costs a little until deleted"
fi
