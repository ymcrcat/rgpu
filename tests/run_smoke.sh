#!/usr/bin/env bash
# Start the fake-driver server, run the remoting smoke test against it, stop it.
# Invoked by ctest; takes the build directory as $1.
set -uo pipefail

BUILD=${1:-build}
PORT=${RGPU_TEST_PORT:-9713}

"$BUILD/rgpu-server-fake" "$PORT" &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# Wait for the listener rather than sleeping a fixed amount, so a slow start
# does not make this flaky.
for _ in $(seq 1 50); do
  if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    exec 3<&- 3>&-
    break
  fi
  sleep 0.1
done

rc=0
LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" "$BUILD/rpc_smoke" || rc=1

# The runtime API path, if it was built. Our libcudart must come first so the
# loader picks it over any stock one.
if [[ -x "$BUILD/cudart_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/cudart_smoke" 2>&1 | grep -v "no version information" || rc=1
fi
exit $rc
