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

# What an expired session leaves behind. Its own server, with a grace period
# short enough to wait out and a file the fake driver publishes its outstanding
# resource counts to; both are server-wide settings that the other tests want
# left alone.
if [[ -x "$BUILD/expiry_smoke" ]]; then
  echo
  EXPIRY_PORT=$((PORT + 3))
  EXPIRY_STATS=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
  RGPU_SESSION_GRACE=3 RGPU_FAKE_STATS="$EXPIRY_STATS" \
    "$BUILD/rgpu-server-fake" "$EXPIRY_PORT" &
  EXPIRY_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$EXPIRY_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$EXPIRY_PORT" \
    RGPU_SESSION_GRACE=3 RGPU_FAKE_STATS="$EXPIRY_STATS" \
    "$BUILD/expiry_smoke" || rc=1
  kill $EXPIRY_SRV 2>/dev/null
  rm -f "$EXPIRY_STATS" "$EXPIRY_STATS.tmp"

  # Sessions that destroy a primary context, then expire. Each scenario gets a
  # server of its own, since each needs nobody else on it.
  # Everything has to be back to zero afterwards - including the count of
  # releases the fake driver had to refuse, and the count of frees of things
  # already destroyed.
  #
  #   reset:   the reset is allowed (the other half of the ruling above). It
  #            destroys what is in the context but releases nothing, so the
  #            session still owes both its retains and expiry has to pay them:
  #            a retain left held is a leak, one released twice an
  #            over-release.
  #   release: releasing the only retain destroys what is in the context, so
  #            expiry must not free any of it again, which would be stale.
  #   tenants: three sessions share the primary context, and one that released
  #            its retain but kept its resources has them destroyed by
  #            another's last release, or another's expiry; its own expiry
  #            must then free nothing (expiry_smoke.cpp says more). Three
  #            sessions to wait for.
  #   derived: graphs, clones and executables made from device 0's objects
  #            while device 1 is current go with device 0's primary context,
  #            and expiry must not free them again. Two devices.
  then_expire() {
    local mode=$1 port=$2 sessions=$3 devices=${4:-1}
    local stats log
    stats=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
    log=$(mktemp "${TMPDIR:-/tmp}/rgpu-$mode-log.XXXXXX")
    RGPU_SESSION_GRACE=3 RGPU_FAKE_STATS="$stats" RGPU_FAKE_DEVICES="$devices" \
      "$BUILD/rgpu-server-fake" "$port" >"$log" 2>&1 &
    local srv=$!
    for _ in $(seq 1 50); do
      if (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
        exec 3<&- 3>&-
        break
      fi
      sleep 0.1
    done
    # $mode may be two words.
    # shellcheck disable=SC2086
    LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$port" \
      RGPU_SESSION_GRACE=3 RGPU_FAKE_STATS="$stats" RGPU_SERVER_LOG="$log" \
      "$BUILD/expiry_smoke" $mode || rc=1
    # Wait for every session to expire rather than for the counters, which may
    # already look right: the mistakes being looked for happen at expiry.
    for _ in $(seq 1 300); do
      [[ $(grep -c " expired;" "$log") -ge $sessions ]] && break
      sleep 0.1
    done
    local want="allocs=0 retains=0 contexts=0 modules=0 streams=0 events=0"
    want="$want graphs=0 execs=0 cublas=0 cublaslt=0 cudnn=0"
    want="$want overreleases=0 stale=0"
    local got
    got=$(cat "$stats" 2>/dev/null)
    if [[ "$got" != "$want" ]]; then
      echo "FAIL: after a $mode and an expiry the server should hold nothing"
      echo "      and have released nothing it no longer owned"
      echo "  expected: $want"
      echo "     found: $got"
      grep "session cleanup\|expired" "$log" | sed 's/^/  /'
      rc=1
    else
      grep "expired" "$log" | sed 's/^/  /'
    fi
    kill $srv 2>/dev/null
    rm -f "$stats" "$stats.tmp" "$log"
  }
  then_expire reset $((PORT + 4)) 1
  then_expire release $((PORT + 5)) 1
  then_expire "tenants release" $((PORT + 6)) 3
  then_expire "tenants expire" $((PORT + 7)) 3
  then_expire derived $((PORT + 8)) 1 2
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

# A device reset on one thread must invalidate every other thread's cached
# context selection. Its own server, because it relies on nobody else holding
# a retain on the primary context it releases.
if [[ -x "$BUILD/thread_id_smoke" ]]; then
  echo
  THREAD_PORT=$((PORT + 9))
  "$BUILD/rgpu-server-fake" "$THREAD_PORT" &
  THREAD_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$THREAD_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$THREAD_PORT" \
    "$BUILD/thread_id_smoke" runtime || rc=1
  kill $THREAD_SRV 2>/dev/null
fi

# The runtime API path, if it was built. Our libcudart must come first so the
# loader picks it over any stock one.
if [[ -x "$BUILD/cudart_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/cudart_smoke" 2>&1 | grep -v "no version information" || rc=1
fi
exit $rc
