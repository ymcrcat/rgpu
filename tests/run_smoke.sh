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

  # The same break, in the middle of a batch two client threads queued under
  # two devices' contexts. Its own server: two devices, a stats file the test
  # reads the fake's cross-context count from, and a drop point of its own.
  echo
  THREADS_DROP_PORT=$((PORT + 16))
  THREADS_DROP_STATS=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
  RGPU_DROP_AFTER=16 RGPU_SESSION_GRACE=30 RGPU_FAKE_DEVICES=2 \
    RGPU_FAKE_STATS="$THREADS_DROP_STATS" \
    "$BUILD/rgpu-server-fake" "$THREADS_DROP_PORT" &
  THREADS_DROP_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$THREADS_DROP_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  threads_drop_out=$(LD_LIBRARY_PATH="$BUILD" \
    RGPU_SERVER="127.0.0.1:$THREADS_DROP_PORT" RGPU_BATCH=1 \
    RGPU_FAKE_STATS="$THREADS_DROP_STATS" \
    "$BUILD/reconnect_smoke" threads 2>&1) || rc=1
  printf '%s\n' "$threads_drop_out"
  # Frames 1-8 are the setup (cuInit, the device count, then each thread's
  # retain, selection and allocation), 9-24 thread A's memsets, 25-40 thread
  # B's, 41 the call that flushes them. The server drops on reading frame 16,
  # in the middle of A's run, so the client has to send 16-41 again: 26 calls,
  # both threads' among them. Any other count means the break did not land
  # inside the batch, and the test's checks would pass without a replay that
  # crosses threads.
  if ! printf '%s\n' "$threads_drop_out" |
    grep -q "and resumed; 26 call(s) to send again"; then
    echo "FAIL: reconnect_smoke threads did not resume sending exactly the 26"
    echo "      calls from the middle of the two threads' batch; check where"
    echo "      RGPU_DROP_AFTER lands"
    printf '%s\n' "$threads_drop_out" | grep "and resumed" | sed 's/^/  /'
    rc=1
  fi
  kill $THREADS_DROP_SRV 2>/dev/null
  rm -f "$THREADS_DROP_STATS" "$THREADS_DROP_STATS.tmp"

  # The same break, but the client stays away until its session has expired.
  # Its own server: a grace period short enough to wait out, a stats file
  # counting the calls that ran, a log the test waits on, and a drop point of
  # its own. Frames 1-5 are cuInit, the device, the context, the allocation
  # and cuDeviceTotalMem; the server drops on reading frame 6, the
  # asynchronous copy, which goes out on its own and wants no reply.
  echo
  EXPIRED_PORT=$((PORT + 19))
  EXPIRED_STATS=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
  EXPIRED_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-expired-log.XXXXXX")
  RGPU_DROP_AFTER=6 RGPU_SESSION_GRACE=1 RGPU_FAKE_STATS="$EXPIRED_STATS" \
    "$BUILD/rgpu-server-fake" "$EXPIRED_PORT" >"$EXPIRED_LOG" 2>&1 &
  EXPIRED_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$EXPIRED_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  expired_out=$(LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$EXPIRED_PORT" \
    RGPU_BATCH=1 RGPU_RECONNECT_SECONDS=20 RGPU_FAKE_STATS="$EXPIRED_STATS" \
    RGPU_SERVER_LOG="$EXPIRED_LOG" "$BUILD/reconnect_smoke" expired 2>&1) || rc=1
  printf '%s\n' "$expired_out"
  kill $EXPIRED_SRV 2>/dev/null
  # Said by the client, and never undone: no resume on either side.
  if ! printf '%s\n' "$expired_out" | grep -q "no longer has our session"; then
    echo "FAIL: the client never said the server no longer has its session"
    rc=1
  fi
  if grep -q "resumed" "$EXPIRED_LOG" ||
    printf '%s\n' "$expired_out" | grep -q "and resumed"; then
    echo "FAIL: a session that had expired was resumed"
    grep "resumed" "$EXPIRED_LOG" | sed 's/^/  /'
    rc=1
  fi
  if ! grep -q "dropping the connection after 6 frames" "$EXPIRED_LOG"; then
    echo "FAIL: the server never broke the connection, so reconnect_smoke"
    echo "      expired proved nothing"
    rc=1
  fi
  rm -f "$EXPIRED_STATS" "$EXPIRED_STATS.tmp" "$EXPIRED_LOG"
fi

# A peer that connects and never sends its handshake must not tie up an accept
# thread and an fd forever: the handshake read has a bounded deadline
# (RGPU_HANDSHAKE_TIMEOUT_SECONDS), after which the server closes the
# connection. Its own server, with a short deadline, so the wait is quick and
# does not depend on any client binary. A silent connection has to be closed
# from the server side within the deadline, and a real client has to be served
# after it, proving the accept path stayed healthy.
echo
HS_PORT=$((PORT + 22))
HS_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-handshake-log.XXXXXX")
RGPU_HANDSHAKE_TIMEOUT_SECONDS=1 "$BUILD/rgpu-server-fake" "$HS_PORT" \
  >"$HS_LOG" 2>&1 &
HS_SRV=$!
for _ in $(seq 1 50); do
  if (exec 3<>/dev/tcp/127.0.0.1/"$HS_PORT") 2>/dev/null; then
    exec 3<&- 3>&-
    break
  fi
  sleep 0.1
done
# Open a connection, send nothing, and read: the server's close arrives as EOF,
# so `cat` returns. With no deadline it would block until `timeout` kills it,
# which is the failure. The deadline is 1s; 8s is generous slack for a loaded CI
# box while still catching a hang.
hs_start=$(date +%s)
if timeout 8 bash -c "exec 3<>/dev/tcp/127.0.0.1/$HS_PORT; cat <&3 >/dev/null"; then
  hs_closed=1
else
  hs_closed=0
fi
hs_elapsed=$(( $(date +%s) - hs_start ))
if [[ $hs_closed -ne 1 ]]; then
  echo "FAIL: the server did not close a silent connection within the handshake"
  echo "      deadline; a peer that sends nothing ties up an accept thread forever"
  rc=1
else
  echo "a silent connection was closed by the server after ${hs_elapsed}s"
fi
# The server is still healthy: a real client is served after the silent peer.
LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$HS_PORT" "$BUILD/rpc_smoke" \
  || { echo "FAIL: the server did not serve a client after a silent connection"; rc=1; }
kill $HS_SRV 2>/dev/null
rm -f "$HS_LOG"

# A request still running when its client reconnects, sent again by the client
# and never run again by the server. Its own server: one fake call made slow
# enough to reconnect in the middle of, and a stats file counting how often it
# ran.
if [[ -x "$BUILD/replay_smoke" ]]; then
  echo
  REPLAY_PORT=$((PORT + 13))
  REPLAY_STATS=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
  REPLAY_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-replay-log.XXXXXX")
  RGPU_FAKE_SLOW_TOTALMEM_MS=1500 RGPU_FAKE_STATS="$REPLAY_STATS" \
    "$BUILD/rgpu-server-fake" "$REPLAY_PORT" >"$REPLAY_LOG" 2>&1 &
  REPLAY_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$REPLAY_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  RGPU_SERVER="127.0.0.1:$REPLAY_PORT" RGPU_FAKE_STATS="$REPLAY_STATS" \
    "$BUILD/replay_smoke" || rc=1
  kill $REPLAY_SRV 2>/dev/null
  # Any client can send frames naming no request id, as many as it likes, so
  # the server says so once per session rather than once per frame.
  no_id=$(grep -c "names no request id" "$REPLAY_LOG")
  if [[ "$no_id" != 1 ]]; then
    echo "FAIL: the server logged frames naming no request id $no_id times;"
    echo "      once per session is what keeps a client from flooding the log"
    rc=1
  fi
  rm -f "$REPLAY_STATS" "$REPLAY_STATS.tmp" "$REPLAY_LOG"
fi

# Sessions lost while the server was busy with them. Its own server: a grace
# period short enough to run out in the middle of a reconnect's hand-over, a
# hand-over held back past it (RGPU_TEST_HANDOFF_DELAY_MS, a test hook), a cap
# of two client threads so that a client can be refused at it, and a log to
# count lines in. replay_smoke.cpp says more about each case.
if [[ -x "$BUILD/replay_smoke" ]]; then
  echo
  LOST_PORT=$((PORT + 18))
  LOST_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-lost-log.XXXXXX")
  RGPU_SESSION_GRACE=1 RGPU_TEST_HANDOFF_DELAY_MS=2500 \
    RGPU_MAX_CLIENT_THREADS=2 \
    "$BUILD/rgpu-server-fake" "$LOST_PORT" >"$LOST_LOG" 2>&1 &
  LOST_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$LOST_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  RGPU_SERVER="127.0.0.1:$LOST_PORT" "$BUILD/replay_smoke" handoff || rc=1
  echo
  RGPU_SERVER="127.0.0.1:$LOST_PORT" "$BUILD/replay_smoke" flood || rc=1
  # Both sessions expire; what the flood leaves is only said at expiry.
  for _ in $(seq 1 100); do
    [[ $(grep -c " expired;" "$LOST_LOG") -ge 2 ]] && break
    sleep 0.1
  done
  kill $LOST_SRV 2>/dev/null
  # A client can make these failures as fast as it can send frames, so each
  # kind is said once per session, and the count comes at expiry. The flood
  # holds 201 failures: thread 1's first and one for each of the 200 refused
  # threads. Two reach their thread; the other 199 are evicted, or still held
  # when the session expires.
  held_lines=$(grep -c "had no reply to report it in" "$LOST_LOG")
  lost_lines=$(grep -c "without learning that a call it sent" "$LOST_LOG")
  if [[ "$held_lines" != 1 || "$lost_lines" != 1 ]]; then
    echo "FAIL: the server logged $held_lines held failures and $lost_lines"
    echo "      failures that never reached their thread; once per session each"
    rc=1
  fi
  if ! grep "expired;" "$LOST_LOG" |
    grep -q "201 failure(s) of calls sent without a reply were held, 199 of them never reached"; then
    echo "FAIL: the flood's session did not report at expiry how many failures"
    echo "      it held (201) and how many never reached their thread (199)"
    grep "session .* expired\|without a reply were held" "$LOST_LOG" | sed 's/^/  /'
    rc=1
  fi
  rm -f "$LOST_LOG"
fi

# How many sessions a server keeps, and sessions it made but could never serve.
# Its own server: a cap of two sessions, a grace period short enough to wait
# out, every new session's handshake reply held back long enough for a client
# to be gone before it is written (RGPU_TEST_HANDSHAKE_DELAY_MS, a test hook),
# and a log to count lines in. replay_smoke.cpp says more.
if [[ -x "$BUILD/replay_smoke" ]]; then
  echo
  SESSIONS_PORT=$((PORT + 22))
  SESSIONS_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-sessions-log.XXXXXX")
  RGPU_MAX_SESSIONS=2 RGPU_SESSION_GRACE=5 RGPU_TEST_HANDSHAKE_DELAY_MS=1000 \
    "$BUILD/rgpu-server-fake" "$SESSIONS_PORT" >"$SESSIONS_LOG" 2>&1 &
  SESSIONS_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$SESSIONS_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  RGPU_SERVER="127.0.0.1:$SESSIONS_PORT" RGPU_TEST_HANDSHAKE_DELAY_MS=1000 \
    "$BUILD/replay_smoke" sessions || rc=1
  kill $SESSIONS_SRV 2>/dev/null
  # The lost handshake replies have to have been lost at the write, or the
  # case proved nothing: a server that never read the hello makes no session
  # either.
  lost=$(grep -c "could not answer its handshake" "$SESSIONS_LOG")
  if [[ "$lost" != 2 ]]; then
    echo "FAIL: the server failed to write $lost handshake replies, not 2, so"
    echo "      replay_smoke sessions did not test a lost handshake reply"
    rc=1
  fi
  if ! grep -q "refusing a new session" "$SESSIONS_LOG"; then
    echo "FAIL: the server never said it refused a session past its cap"
    rc=1
  fi
  rm -f "$SESSIONS_LOG"
fi

# Reconnects that land while the thread serving a session is between
# connections. Its own server, and nothing else on it: the case depends on a
# reconnect being given the descriptor number of the connection that just
# closed, which is the lowest free one only when nothing else is open. The gap
# is held open by RGPU_TEST_RECONNECT_GAP_MS, a test hook.
if [[ -x "$BUILD/replay_smoke" ]]; then
  echo
  GAP_PORT=$((PORT + 23))
  GAP_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-gap-log.XXXXXX")
  RGPU_SESSION_GRACE=30 RGPU_TEST_RECONNECT_GAP_MS=1500 \
    "$BUILD/rgpu-server-fake" "$GAP_PORT" >"$GAP_LOG" 2>&1 &
  GAP_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$GAP_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  RGPU_SERVER="127.0.0.1:$GAP_PORT" "$BUILD/replay_smoke" gap || rc=1
  kill $GAP_SRV 2>/dev/null
  # A reconnect has to have been given a descriptor number a closed connection
  # had, or the first case passes without testing anything. The test keeps the
  # lower numbers occupied to make that happen; this is where it shows.
  first_closed=$(grep -o "connection closed (descriptor [0-9]*" "$GAP_LOG" |
    grep -o "[0-9]*$" | head -1)
  first_handed=$(grep -o "handed it over on descriptor [0-9]*" "$GAP_LOG" |
    grep -o "[0-9]*$" | head -1)
  if [[ -z "$first_closed" || "$first_closed" != "$first_handed" ]]; then
    echo "FAIL: no reconnect was given the descriptor number of a connection"
    echo "      that had closed, so replay_smoke gap did not test the reuse"
    grep "descriptor" "$GAP_LOG" | sed 's/^/  /'
    rc=1
  fi
  rm -f "$GAP_LOG"
fi

# The handshake read is bounded by an elapsed-time deadline, and the
# connections in the pre-handshake read are capped. Its own server, with a
# short deadline (RGPU_HANDSHAKE_TIMEOUT_SECONDS) and a small cap
# (RGPU_MAX_PENDING_HANDSHAKES) so a dribbling peer is closed quickly and the
# cap can be reached; a log to check the refusal really was the server's.
# replay_smoke.cpp explains the cases.
if [[ -x "$BUILD/replay_smoke" ]]; then
  echo
  HSCAP_PORT=$((PORT + 24))
  HSCAP_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-hscap-log.XXXXXX")
  RGPU_HANDSHAKE_TIMEOUT_SECONDS=2 RGPU_MAX_PENDING_HANDSHAKES=4 \
    "$BUILD/rgpu-server-fake" "$HSCAP_PORT" >"$HSCAP_LOG" 2>&1 &
  HSCAP_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$HSCAP_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  RGPU_SERVER="127.0.0.1:$HSCAP_PORT" RGPU_HANDSHAKE_TIMEOUT_SECONDS=2 \
    RGPU_MAX_PENDING_HANDSHAKES=4 "$BUILD/replay_smoke" handshake || rc=1
  kill $HSCAP_SRV 2>/dev/null
  # The connection past the cap has to have been refused by the server, or the
  # case proved nothing: a healthy server that simply closed it at the deadline
  # would look the same to a client that did not measure the wait.
  if ! grep -q "the most allowed (RGPU_MAX_PENDING_HANDSHAKES)" "$HSCAP_LOG"; then
    echo "FAIL: the server never refused a connection past the pre-handshake"
    echo "      cap, so replay_smoke handshake did not test the cap"
    rc=1
  fi
  rm -f "$HSCAP_LOG"
fi

# Request ids wrapping past 0xFFFFFFFF, with the connection broken in a batch
# that spans the wrap. Its own server: a break by frame count, which only means
# something with one client on the server, and a stats file counting the calls
# that ran. wrap_smoke.cpp explains the numbers.
if [[ -x "$BUILD/wrap_smoke" ]]; then
  echo
  WRAP_PORT=$((PORT + 14))
  WRAP_STATS=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
  RGPU_DROP_AFTER=16 RGPU_SESSION_GRACE=30 RGPU_FAKE_STATS="$WRAP_STATS" \
    "$BUILD/rgpu-server-fake" "$WRAP_PORT" &
  WRAP_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$WRAP_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  wrap_out=$(LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$WRAP_PORT" \
    RGPU_BATCH=1 RGPU_TEST_FIRST_REQ_ID=4294967286 \
    RGPU_FAKE_STATS="$WRAP_STATS" "$BUILD/wrap_smoke" 2>&1) || rc=1
  printf '%s\n' "$wrap_out"
  # The break has to have happened, and the client has to have sent again
  # exactly the 7 frames the server had not run (16-22). The count is the only
  # place the client's own trimming shows: one that forgot nothing would send
  # all 21, the server would skip the 14 that already ran, and every other
  # check would still pass - while a real client's replay buffer never shrank.
  if ! printf '%s\n' "$wrap_out" | grep -q "and resumed; 7 call(s) to send again"; then
    echo "FAIL: wrap_smoke's client did not resume sending exactly the 7 calls"
    echo "      the server had not run; either the break never happened (check"
    echo "      where RGPU_DROP_AFTER lands) or the client trimmed the wrong frames"
    printf '%s\n' "$wrap_out" | grep "and resumed" | sed 's/^/  /'
    rc=1
  fi
  kill $WRAP_SRV 2>/dev/null
  rm -f "$WRAP_STATS" "$WRAP_STATS.tmp"
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
  #            while device 1 is current belong to no context, so releasing
  #            device 0 leaves them and their maker destroys them; expiry must
  #            free nothing again. Two devices.
  #   foreign: a graph captured on another session's stream is of unknown
  #            placement, so the capturing session's expiry leaves it alone.
  #            The other session cleans it up. Two sessions to wait for.
  #   twoctx:  two created contexts, one destroyed while the other is current;
  #            the teardown must forget exactly the destroyed context's entries,
  #            so expiry leaks none of the other's and frees none of the gone
  #            one's again. The regression guard for the by-context index.
  #   detach:  cuCtxDetach destroys a created context and what is in it, so
  #            expiry must not destroy or free any of it again, which would
  #            be stale.
  #   capture: a session killed in the middle of two stream captures, its
  #            thread left in GLOBAL mode (strict) or RELAXED mode (relaxed),
  #            while another session holds the primary context; its expiry has
  #            to end both captures and give back everything, which a thread
  #            still restricted by its own capture cannot. Two sessions.
  #   threads: two client threads on two devices, one with a created context
  #            on top of its primary one, exit holding everything; the slots
  #            the server keeps for them must not stand in the way of giving
  #            back both devices' retains. Two devices.
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
    want="$want graphs=0 execs=0 cublas=0 cublaslt=0 cudnn=0 captures=0"
    want="$want libraries=0 overreleases=0 stale=0"
    local got
    # The call counters at the end of the line count calls, not resources, so
    # they are not part of what has to come back to zero.
    got=$(sed -e 's/ ctxsets=[0-9]*//' -e 's/ crossctx=[0-9]*//' \
      -e 's/ totalmem=[0-9]*//' -e 's/ modeswaps=[0-9]*//' \
      -e 's/ memsets=[0-9]*//' "$stats" 2>/dev/null)
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
  then_expire foreign $((PORT + 11)) 2
  then_expire detach $((PORT + 15)) 1
  then_expire twoctx $((PORT + 16)) 1
  then_expire threads $((PORT + 17)) 1 2
  then_expire "capture strict" $((PORT + 20)) 2
  then_expire "capture relaxed" $((PORT + 21)) 2
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

# Each client thread keeps its own context on the server. Its own server: two
# devices, a stats file the test reads the fake's call counters from, low caps
# on live client threads and on each one's context stack so the caps can be
# reached, and destroyed
# context handles handed out again so a stale one can be shown never to bind
# the context that took its address. RGPU_BATCH=1 is the default, pinned
# because the batch-flushed-by-another-thread case means nothing without
# batching.
if [[ -x "$BUILD/threadctx_smoke" ]]; then
  echo
  CTX_PORT=$((PORT + 10))
  CTX_STATS=$(mktemp "${TMPDIR:-/tmp}/rgpu-stats.XXXXXX")
  CTX_LOG=$(mktemp "${TMPDIR:-/tmp}/rgpu-threadctx-log.XXXXXX")
  RGPU_FAKE_DEVICES=2 RGPU_FAKE_STATS="$CTX_STATS" RGPU_MAX_CLIENT_THREADS=16 \
    RGPU_MAX_CONTEXT_STACK=16 RGPU_FAKE_REUSE_CONTEXTS=1 \
    "$BUILD/rgpu-server-fake" "$CTX_PORT" >"$CTX_LOG" 2>&1 &
  CTX_SRV=$!
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$CTX_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      break
    fi
    sleep 0.1
  done
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$CTX_PORT" RGPU_BATCH=1 \
    RGPU_FAKE_STATS="$CTX_STATS" RGPU_MAX_CLIENT_THREADS=16 \
    RGPU_MAX_CONTEXT_STACK=16 "$BUILD/threadctx_smoke" || rc=1
  kill $CTX_SRV 2>/dev/null
  # Said once per session, and said at all: nobody should have to diagnose
  # serialized multithreaded clients as a mystery.
  if ! grep -q "has more than one client thread" "$CTX_LOG"; then
    echo "FAIL: the server never said a session had more than one client thread"
    rc=1
  elif ! grep "has more than one client thread" "$CTX_LOG" |
    grep -q "own current context, context stack and stream capture mode"; then
    # And says what it keeps per thread now, rather than what it used to.
    echo "FAIL: the server's word on more than one client thread does not name"
    echo "      the thread state it keeps: the context, stack and capture mode"
    grep "has more than one client thread" "$CTX_LOG" | head -1 | sed 's/^/  /'
    rc=1
  fi
  # The fake has no cuCtxAttach and no green contexts either, so the result
  # alone cannot show whose refusal it was. The server says it, once each.
  if [[ $(grep -c "refusing cuCtxAttach" "$CTX_LOG") -ne 1 ]]; then
    echo "FAIL: the server did not say, once, that it refuses cuCtxAttach"
    rc=1
  fi
  if [[ $(grep -c "refusing green contexts" "$CTX_LOG") -ne 1 ]]; then
    echo "FAIL: the server did not say, once, that it refuses green contexts"
    rc=1
  fi
  rm -f "$CTX_STATS" "$CTX_STATS.tmp" "$CTX_LOG"

  # A reset is refused while another session is live, and the case above
  # leaves several waiting out their grace, so the reset case has a server of
  # its own.
  echo
  RESET_PORT=$((PORT + 12))
  RGPU_FAKE_DEVICES=2 "$BUILD/rgpu-server-fake" "$RESET_PORT" &
  RESET_SRV=$!
  reset_up=0
  for _ in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$RESET_PORT") 2>/dev/null; then
      exec 3<&- 3>&-
      reset_up=1
      break
    fi
    sleep 0.1
  done
  # Said, rather than left to the client's connection error: a server that
  # never listened is a different failure from a reset that went wrong.
  if [[ $reset_up -ne 1 ]]; then
    echo "FAIL: the reset case's server never came up on port $RESET_PORT"
    rc=1
  else
    LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$RESET_PORT" \
      "$BUILD/threadctx_smoke" reset || rc=1
  fi
  kill $RESET_SRV 2>/dev/null
  # Reaped before going on, so it is gone - port and all - before anything
  # after it starts.
  wait $RESET_SRV 2>/dev/null
fi

# The runtime API path, if it was built. Our libcudart must come first so the
# loader picks it over any stock one.
if [[ -x "$BUILD/cudart_smoke" ]]; then
  echo
  LD_LIBRARY_PATH="$BUILD" RGPU_SERVER="127.0.0.1:$PORT" \
    "$BUILD/cudart_smoke" 2>&1 | grep -v "no version information" || rc=1
fi
exit $rc
