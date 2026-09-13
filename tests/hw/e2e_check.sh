#!/usr/bin/env bash
# End-to-end checks of the real rgpu-server on a GPU host, through the rgpu
# client shims. Run on the pod from the repo root, after
# scripts/deploy_server.sh has built build/rgpu-server and build/libcuda.so.1
# (scripts/hw_check.sh runs this for you):
#
#   tests/hw/e2e_check.sh
#
# Scenarios, each on a server of its own:
#   E1  an expired session gives its GPU memory back, and a live one keeps its
#   E2  a device reset is refused while another session is live, allowed alone
#   E3  two client threads keep two contexts apart on one GPU
#   E4  kernel launches across a dropped connection run exactly once
#   E5  a session lost to a server restart stops cleanly and replays nothing
#
# Prints PASS / FAIL / SKIP lines and exits 1 if anything failed. Leaves no
# processes behind.
#
# Settings (environment):
#   BUILD=build             where rgpu-server, libcuda.so.1 and the fatbin are
#   HW_OUT=$BUILD/hw        where the client binary goes
#   HW_PORT_BASE=9811       ports HW_PORT_BASE+1 .. +9 are used
#   HW_IMAGE=$BUILD/vecadd.fatbin   kernel for E4 (memsets without it)
#   HW_HOG_MIB=1024         what the dead client of E1 allocates
#   HW_GRACE=6              RGPU_SESSION_GRACE for E1 and E2
#   HW_ONLY="E1 E3"         run only these scenarios
#   RGPU_E2E_SERVER=path    another server binary; only for trying this script
#                           out against rgpu-server-fake, where the memory
#                           checks cannot pass
set -uo pipefail

cd "$(dirname "$0")/../.."
ROOT=$PWD
BUILD=${BUILD:-build}
BUILD_ABS=$(cd "$BUILD" 2>/dev/null && pwd) || {
  echo "FAIL  no $BUILD directory; run scripts/deploy_server.sh first"
  exit 1
}
OUT=${HW_OUT:-$BUILD_ABS/hw}
PORT_BASE=${HW_PORT_BASE:-9811}
IMAGE=${HW_IMAGE:-$BUILD_ABS/vecadd.fatbin}
HOG_MIB=${HW_HOG_MIB:-1024}
GRACE=${HW_GRACE:-6}
ONLY=${HW_ONLY:-}
SERVER=${RGPU_E2E_SERVER:-$BUILD_ABS/rgpu-server}
CXX=${CXX:-g++}
CUDA_INC=${CUDA_INC:-$ROOT/third_party/cuda_include}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/rgpu-e2e.XXXXXX")
PIDS=()

descendants() {
  local p
  for p in $(pgrep -P "$1" 2>/dev/null); do
    echo "$p"
    descendants "$p"
  done
}
cleanup() {
  local p
  for p in ${PIDS[@]+"${PIDS[@]}"} $(descendants $$); do
    kill -9 "$p" 2>/dev/null
  done
  wait 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

passes=0
fails=0
skips=0
pass() { echo "PASS  $*"; passes=$((passes + 1)); }
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }
skip() { echo "SKIP  $*"; skips=$((skips + 1)); }
note() { echo "      $*"; }
indent() { sed 's/^/      | /' "$@"; }

want() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }

# --- preconditions ------------------------------------------------------------

if [[ ! -x "$SERVER" ]]; then
  fail "no $SERVER; run scripts/deploy_server.sh first"
  exit 1
fi
if [[ ! -e "$BUILD_ABS/libcuda.so.1" ]]; then
  fail "no $BUILD_ABS/libcuda.so.1 (the client shim); run scripts/deploy_server.sh first"
  exit 1
fi
if [[ ! -f "$CUDA_INC/cuda.h" ]]; then
  fail "no cuda.h in $CUDA_INC"
  exit 1
fi

mkdir -p "$OUT"
echo "== building the e2e client against the shim"
if ! "$CXX" -std=c++17 -O1 -g -Wall -Wno-deprecated-declarations \
  -I"$CUDA_INC" -I"$ROOT" tests/hw/e2e_client.cpp -o "$OUT/e2e_client" \
  -L"$BUILD_ABS" -l:libcuda.so.1 -Wl,-rpath,"$BUILD_ABS" -ldl -lpthread; then
  fail "the e2e client did not compile"
  exit 1
fi
CLIENT=$OUT/e2e_client

HAVE_SMI=0
if command -v nvidia-smi >/dev/null 2>&1 &&
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >/dev/null 2>&1; then
  HAVE_SMI=1
fi

# --- helpers -------------------------------------------------------------------

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# start_server NAME PORT [VAR=value ...]; sets SERVER_PID and SERVER_LOG.
start_server() {
  local name=$1 port=$2
  shift 2
  SERVER_LOG=$WORK/$name.server.log
  if port_open "$port"; then
    note "port $port is already in use; not starting $name (set HW_PORT_BASE)"
    return 1
  fi
  # The server must load the real driver, never the shim it would otherwise
  # find through a LD_LIBRARY_PATH that names the build directory.
  local ldpath
  ldpath=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' |
    grep -vxF "$BUILD_ABS" | grep -vxF "$BUILD" | paste -sd: -)
  env -u LD_PRELOAD LD_LIBRARY_PATH="$ldpath" "$@" "$SERVER" "$port" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  PIDS+=("$SERVER_PID")
  local i
  for i in $(seq 1 150); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      note "server $name exited while starting:"
      indent "$SERVER_LOG"
      return 1
    fi
    if port_open "$port"; then
      local lib
      lib=$(grep -m1 -o '/[^ ]*libcuda\.so[^ ]*' "/proc/$SERVER_PID/maps" 2>/dev/null)
      if [[ -n "$lib" && "$lib" == "$BUILD_ABS"/* ]]; then
        fail "server $name loaded the shim ($lib) instead of the driver"
        return 1
      fi
      note "server $name up on $port (pid $SERVER_PID, driver ${lib:-unknown})"
      return 0
    fi
    sleep 0.1
  done
  note "server $name never listened on $port"
  return 1
}

stop_server() {
  local pid=$1 i
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
}

# client PORT args...  (foreground)
client() {
  local port=$1
  shift
  LD_LIBRARY_PATH="$BUILD_ABS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    RGPU_SERVER="127.0.0.1:$port" "$CLIENT" "$@"
}

# client_bg PORT OUTFILE args...; sets BG_PID, which is the client's own pid
# (exec), so killing it kills the client and not a subshell around it.
client_bg() {
  local port=$1 out=$2
  shift 2
  (
    export LD_LIBRARY_PATH="$BUILD_ABS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export RGPU_SERVER="127.0.0.1:$port"
    exec "$CLIENT" "$@"
  ) >"$out" 2>&1 &
  BG_PID=$!
  PIDS+=("$BG_PID")
}

wait_file() {
  local path=$1 seconds=$2 i
  for i in $(seq 1 $((seconds * 10))); do
    [[ -e "$path" ]] && return 0
    sleep 0.1
  done
  return 1
}

# count_log FILE PATTERN
count_log() { grep -c -- "$2" "$1" 2>/dev/null || true; }

# wait_log FILE PATTERN COUNT SECONDS
wait_log() {
  local file=$1 pattern=$2 want=$3 seconds=$4 i
  for i in $(seq 1 $((seconds * 10))); do
    (($(count_log "$file" "$pattern") >= want)) && return 0
    sleep 0.1
  done
  return 1
}

# keeper DIR COMMAND: one command to a `hold` client, prints its reply.
keeper() {
  local dir=$1 cmd=$2 i
  rm -f "$dir/reply"
  printf '%s\n' "$cmd" >"$dir/cmd.tmp" && mv "$dir/cmd.tmp" "$dir/cmd"
  for i in $(seq 1 1200); do
    if [[ -e "$dir/reply" ]]; then
      cat "$dir/reply"
      rm -f "$dir/reply"
      return 0
    fi
    sleep 0.1
  done
  echo "timeout"
  return 1
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 |
    head -1 | tr -d ' '
}

field() { printf '%s\n' "$1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }

# --- E1: expiry releases real GPU memory ---------------------------------------
# Verifies issue #4 (issue-4-report.md): release_inventory in
# server/inventory.cpp frees a dead session's allocations and releases exactly
# its own primary-context retains, never one more.
scenario_expiry() {
  echo
  echo "== E1: an expired session gives its GPU memory back; a live one keeps its"
  local port=$((PORT_BASE + 1)) dir=$WORK/e1
  mkdir -p "$dir"
  start_server e1 "$port" RGPU_SESSION_GRACE="$GRACE" || {
    fail "E1: the server did not start"
    return
  }
  local srv=$SERVER_PID log=$SERVER_LOG
  client_bg "$port" "$dir/keeper.out" hold "$dir"
  local keeper_pid=$BG_PID
  if ! wait_file "$dir/ready" 60; then
    fail "E1: the live session never became ready"
    indent "$dir/keeper.out"
    stop_server "$srv"
    return
  fi
  sleep 1
  local base=0 held=0 now=0 base_free after_free
  ((HAVE_SMI)) && base=$(gpu_used_mib)
  base_free=$(field "$(keeper "$dir" free)" free)
  base_free=${base_free:-0}
  note "baseline with the live session: nvidia-smi ${base} MiB used, cuMemGetInfo free $((base_free >> 20)) MiB"

  client "$port" hog "$HOG_MIB" >"$dir/hog.out" 2>&1
  if ! grep -q ALLOCATED "$dir/hog.out"; then
    fail "E1: the dying client could not allocate ${HOG_MIB} MiB"
    indent "$dir/hog.out"
    keeper "$dir" exit >/dev/null
    stop_server "$srv"
    return
  fi
  local died=$SECONDS
  sleep 1
  if ((HAVE_SMI)); then
    held=$(gpu_used_mib)
    if ((held - base >= HOG_MIB * 9 / 10)); then
      pass "E1: within its grace period the dead client's ${HOG_MIB} MiB are still held ($base -> $held MiB used)"
    else
      fail "E1: the dead client's ${HOG_MIB} MiB never showed up in nvidia-smi ($base -> $held MiB used), so the rest proves nothing"
    fi
  fi
  if wait_log "$log" " expired;" 1 $((GRACE + 60)); then
    note "server: $(grep -m1 ' expired;' "$log" | sed 's/^\[rgpu-server\] //')"
    note "the session expired about $((SECONDS - died))s after the client died (grace ${GRACE}s)"
  else
    fail "E1: the dead client's session never expired"
  fi

  if ((HAVE_SMI)); then
    local i
    for i in $(seq 1 60); do
      now=$(gpu_used_mib)
      ((now - base <= 128)) && break
      sleep 0.5
    done
    if ((now - base <= 128)); then
      pass "E1: after expiry device memory is back to baseline ($held -> $now MiB used, baseline $base)"
    else
      fail "E1: after expiry device memory did not come back ($held -> $now MiB used, baseline $base)"
    fi
  else
    skip "E1: no nvidia-smi; relying on cuMemGetInfo from the live session only"
  fi
  after_free=$(field "$(keeper "$dir" free)" free)
  if ((base_free > 0)) && [[ -n "$after_free" ]] &&
    (((base_free - after_free) >> 20 <= 128 && (after_free - base_free) >> 20 <= 128)); then
    pass "E1: the live session's cuMemGetInfo agrees: free $((base_free >> 20)) -> $((after_free >> 20)) MiB"
  else
    fail "E1: the live session's cuMemGetInfo disagrees: free ${base_free:-?} -> ${after_free:-?} bytes"
  fi
  local v
  v=$(keeper "$dir" verify)
  if [[ "$v" == ok* ]]; then
    pass "E1: the live session's memory still reads back and it can still allocate (no retain released too many)"
  else
    fail "E1: the live session was damaged by the other's expiry: $v"
  fi
  note "live session exits: $(keeper "$dir" exit)"
  wait "$keeper_pid" 2>/dev/null
  if grep -q "session cleanup" "$log"; then
    fail "E1: the server could not release something at expiry:"
    grep "session cleanup" "$log" | indent
  fi
  stop_server "$srv"
}

# --- E2: primary-context reset refusal -----------------------------------------
# Verifies w_cuDevicePrimaryCtxReset_v2 (server/inventory.cpp): refused with
# CUDA_ERROR_NOT_SUPPORTED (801) while another session is live, allowed alone,
# and the session that reset keeps working.
scenario_reset() {
  echo
  echo "== E2: a device reset is refused next to another session, allowed alone"
  local port=$((PORT_BASE + 2)) dir=$WORK/e2
  mkdir -p "$dir"
  start_server e2 "$port" RGPU_SESSION_GRACE="$GRACE" || {
    fail "E2: the server did not start"
    return
  }
  local srv=$SERVER_PID log=$SERVER_LOG
  client_bg "$port" "$dir/keeper.out" hold "$dir"
  local keeper_pid=$BG_PID
  if ! wait_file "$dir/ready" 60; then
    fail "E2: the live session never became ready"
    indent "$dir/keeper.out"
    stop_server "$srv"
    return
  fi
  local out code
  out=$(client "$port" reset 2>&1)
  code=$(printf '%s\n' "$out" | sed -n 's/^RESET_RC //p')
  if [[ "$code" == 801 ]]; then
    pass "E2: with two sessions live, cuDevicePrimaryCtxReset returned 801 (CUDA_ERROR_NOT_SUPPORTED)"
  else
    fail "E2: with two sessions live, cuDevicePrimaryCtxReset returned '${code:-nothing}', not 801"
    printf '%s\n' "$out" | indent
  fi
  local v
  v=$(keeper "$dir" verify)
  if [[ "$v" == ok* ]]; then
    pass "E2: the other session's memory was untouched by the refused reset"
  else
    fail "E2: the other session was damaged: $v"
  fi
  # The resetting client's session stays live for its grace period.
  if ! wait_log "$log" " expired;" 1 $((GRACE + 60)); then
    fail "E2: the resetting client's session never expired, so the keeper is never alone"
  fi
  local r="" i
  for i in $(seq 1 40); do
    r=$(field "$(keeper "$dir" reset)" rc)
    [[ "$r" == 801 ]] || break
    sleep 0.25 # the live count drops a moment after the expiry line
  done
  if [[ "$r" == 0 ]]; then
    pass "E2: alone on the server, cuDevicePrimaryCtxReset succeeded"
    v=$(keeper "$dir" after-reset)
    if [[ "$v" == ok* ]]; then
      pass "E2: after its reset the session still has the primary context current, allocates in it, and reads back"
    else
      fail "E2: after its reset the session does not work: $v"
    fi
  else
    fail "E2: alone on the server, cuDevicePrimaryCtxReset returned '${r:-nothing}'"
  fi
  note "session exits after the reset (release of its retain, see probe check 7): $(keeper "$dir" exit)"
  wait "$keeper_pid" 2>/dev/null
  if wait_log "$log" " expired;" 2 $((GRACE + 60)); then
    note "server: $(grep ' expired;' "$log" | tail -1 | sed 's/^\[rgpu-server\] //')"
  fi
  if grep -q "session cleanup" "$log"; then
    note "server cleanup messages (expected at most a retain leak after a reset):"
    grep "session cleanup" "$log" | indent
  fi
  stop_server "$srv"
}

# --- E3: two client threads, two contexts ------------------------------------
# Verifies issue #2 on hardware: server/client_threads.cpp keeps each client
# thread's context, and placement is read with CU_POINTER_ATTRIBUTE_CONTEXT.
scenario_threads() {
  echo
  echo "== E3: two client threads keep two contexts apart on one GPU"
  local port=$((PORT_BASE + 3))
  start_server e3 "$port" || {
    fail "E3: the server did not start"
    return
  }
  local srv=$SERVER_PID out rc
  out=$(RGPU_BATCH=1 client "$port" threads 2>&1)
  rc=$?
  printf '%s\n' "$out" | grep -E '^(PASS|FAIL)' | sed 's/^PASS /PASS  E3: /; s/^FAIL /FAIL  E3: /'
  passes=$((passes + $(printf '%s\n' "$out" | grep -c '^PASS')))
  fails=$((fails + $(printf '%s\n' "$out" | grep -c '^FAIL')))
  if ((rc != 0)) && ! printf '%s\n' "$out" | grep -q '^FAIL'; then
    fail "E3: the client exited $rc without saying why"
    printf '%s\n' "$out" | indent
  fi
  stop_server "$srv"
}

# --- E4: at-most-once across a dropped connection ------------------------------
# Verifies the serving loop's at-most-once check and the client's resume
# (server/main.cpp serve, client/rpc.cpp): RGPU_DROP_AFTER breaks the
# connection on reading frame N, wherever that falls.
scenario_drop() {
  echo
  echo "== E4: work across a dropped connection runs exactly once"
  local mode=drop args
  if [[ -f "$IMAGE" ]]; then
    args=("$IMAGE" 300)
  else
    mode=drop-memset
    args=(300)
    skip "E4: no kernel image at $IMAGE; memsets catch lost work but not duplicated work"
  fi
  local n i=0
  for n in 30 120 250; do
    i=$((i + 1))
    local port=$((PORT_BASE + 3 + i))
    start_server "e4-$n" "$port" RGPU_DROP_AFTER="$n" RGPU_SESSION_GRACE=60 || {
      fail "E4: the server did not start"
      continue
    }
    local srv=$SERVER_PID log=$SERVER_LOG out rc
    out=$(RGPU_BATCH=1 RGPU_RECONNECT_SECONDS=30 client "$port" "$mode" "${args[@]}" 2>&1)
    rc=$?
    if ! grep -q "dropping the connection after $n frames" "$log"; then
      fail "E4 (drop at frame $n): the server never broke the connection, so this proves nothing"
    elif ! printf '%s\n' "$out" | grep -q "and resumed"; then
      fail "E4 (drop at frame $n): the client never resumed its session"
      printf '%s\n' "$out" | indent
    elif ((rc == 0)); then
      pass "E4 (drop at frame $n): $(printf '%s\n' "$out" | grep -m1 '^PASS' | sed 's/^PASS //')"
    else
      fail "E4 (drop at frame $n): $(printf '%s\n' "$out" | grep '^FAIL' | sed 's/^FAIL //' | paste -sd';' -)"
    fi
    note "client: $(printf '%s\n' "$out" | grep -m1 'and resumed' | sed 's/^\[rgpu\] //')"
    local skipped
    skipped=$(count_log "$log" "already ran")
    note "server: $(grep -m1 'resumed' "$log" | sed 's/^\[rgpu-server\] //'); $skipped copy(ies) of already-run requests skipped"
    stop_server "$srv"
  done
}

# --- E5: a lost session stops cleanly --------------------------------------------
# Verifies client/rpc.cpp: told the session is gone (a restarted server has
# never heard of it), the client fails every call from then on and replays
# nothing into the new server.
scenario_lost() {
  echo
  echo "== E5: a session lost to a server restart stops cleanly"
  local port=$((PORT_BASE + 8)) dir=$WORK/e5
  mkdir -p "$dir"
  start_server e5-before "$port" || {
    fail "E5: the server did not start"
    return
  }
  local first=$SERVER_PID
  client_bg "$port" "$dir/client.out" lost "$dir"
  local client_pid=$BG_PID
  if ! wait_file "$dir/ready" 60; then
    fail "E5: the client never became ready"
    indent "$dir/client.out"
    stop_server "$first"
    return
  fi
  kill -9 "$first" 2>/dev/null
  wait "$first" 2>/dev/null
  local i
  for i in $(seq 1 50); do
    port_open "$port" || break
    sleep 0.1
  done
  start_server e5-after "$port" RGPU_VERBOSE=1 || {
    fail "E5: the restarted server did not start"
    return
  }
  local second=$SERVER_PID log=$SERVER_LOG
  touch "$dir/go"
  local rc=124
  for i in $(seq 1 900); do
    if ! kill -0 "$client_pid" 2>/dev/null; then
      wait "$client_pid"
      rc=$?
      break
    fi
    sleep 0.1
  done
  if ((rc == 124)); then
    fail "E5: the client did not finish within 90s"
    kill -9 "$client_pid" 2>/dev/null
  fi
  grep -E '^(PASS|FAIL)' "$dir/client.out" | sed 's/^PASS /PASS  E5: /; s/^FAIL /FAIL  E5: /'
  passes=$((passes + $(grep -c '^PASS' "$dir/client.out")))
  fails=$((fails + $(grep -c '^FAIL' "$dir/client.out")))
  if grep -q "no longer has our session" "$dir/client.out"; then
    pass "E5: the client said the server no longer has its session"
  else
    fail "E5: the client never said its session was gone"
    indent "$dir/client.out"
  fi
  if grep -q "is not here" "$log"; then
    pass "E5: the restarted server told the client its session is not here"
  else
    fail "E5: the restarted server never saw the client come back"
  fi
  if grep -qE "resumed|session [0-9a-f]+ started| -> " "$log"; then
    fail "E5: the restarted server started a session or ran a call for the lost client:"
    grep -E "resumed|session [0-9a-f]+ started| -> " "$log" | head -10 | indent
  else
    pass "E5: nothing was replayed into the restarted server (no session started, no call ran)"
  fi
  stop_server "$second"
}

want E1 && scenario_expiry
want E2 && scenario_reset
want E3 && scenario_threads
want E4 && scenario_drop
want E5 && scenario_lost

echo
echo "e2e summary: $passes PASS, $fails FAIL, $skips SKIP"
((fails == 0))
