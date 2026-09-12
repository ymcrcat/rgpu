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

if [[ -x "$BUILD/launch_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/launch_smoke" || rc=1
fi

if [[ -x "$BUILD/cublas_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/cublas_smoke" || rc=1
fi

if [[ -x "$BUILD/graph_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/graph_smoke" || rc=1
fi

if [[ -x "$BUILD/cudnn_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/cudnn_smoke" || rc=1
fi

# A server that breaks the connection partway through, to show the client can
# pick the session back up. Its own server, since the drop is a server-wide
# setting and the other tests want a connection that stays up.
if [[ -x "$BUILD/reconnect_smoke" ]]; then
  echo
  DROP_PORT=$((PORT + 1))
  RGPU_DROP_AFTER=12 RGPU_SESSION_GRACE=30 "$BUILD/rgpu-server-fake" "$DROP_PORT" &
  DROP_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$DROP_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  # RGPU_BATCH=1 is already the default, but pinned here because the test
  # depends on it: with batching off the launch that has to fail without a
  # reply becomes a round trip, fails at the call site, and nothing is ever
  # deferred. A stray environment variable should not produce a red that reads
  # like a server regression.
  drop_out=$(LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$DROP_PORT" \
    RGPU_BATCH=1 "$BUILD/reconnect_smoke" 2>&1) || rc=1
  printf '%s\n' "$drop_out"
  # The break has to have actually happened. The test's checks all pass on a
  # connection that was never dropped, so without this it would quietly stop
  # testing anything the day the drop point moved.
  if ! printf '%s\n' "$drop_out" | grep -q "and resumed"; then
    echo "FAIL: the connection was never dropped and resumed, so reconnect_smoke"
    echo "      proved nothing; check where RGPU_DROP_AFTER lands"
    rc=1
  fi
  kill $DROP_SRV 2>/dev/null
fi

# Hostile requests get a server of their own: if one of them does take the
# server down, the other tests should not be the ones that notice.
if [[ -x "$BUILD/hostile_smoke" ]]; then
  echo
  HOSTILE_PORT=$((PORT + 2))
  "$BUILD/rgpu-server-fake" "$HOSTILE_PORT" &
  HOSTILE_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$HOSTILE_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$HOSTILE_PORT" RGPU_RECONNECT_SECONDS=1 \
    "$BUILD/hostile_smoke" || rc=1
  kill $HOSTILE_SRV 2>/dev/null
fi

# The runtime API path, if it was built. Our libcudart must come first so the
# loader picks it over any stock one.
if [[ -x "$BUILD/cudart_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/cudart_smoke" 2>&1 | grep -v "no version information" || rc=1
fi
exit $rc
