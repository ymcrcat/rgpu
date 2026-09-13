// Checks that each client thread keeps its own CUDA context across the wire,
// without a GPU.
//
// CUDA's current context - and the stack it is the top of - belongs to the
// calling thread, but one server thread serves every thread of a client
// process. So the server keeps each client thread's context stack, puts its
// top back before running that thread's request, and reads it back
// afterwards. Without that, the last thread to
// select a context wins for everybody, silently: issue #2.
//
// Every case asserts what the driver documents - what cuCtxGetCurrent returns,
// which device an allocation is on - and never that a use under the wrong
// context fails, because on hardware with unified addressing it most likely
// does not.
//
// Runs against its own rgpu-server-fake with two devices, destroyed context
// handles reused, and batching on:
//
//   RGPU_FAKE_DEVICES=2 RGPU_FAKE_STATS=/tmp/s RGPU_MAX_CLIENT_THREADS=16
//     RGPU_MAX_CONTEXT_STACK=16 RGPU_FAKE_REUSE_CONTEXTS=1
//     rgpu-server-fake 9723 &
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9723 RGPU_BATCH=1
//     RGPU_FAKE_STATS=/tmp/s RGPU_MAX_CLIENT_THREADS=16
//     RGPU_MAX_CONTEXT_STACK=16 ./threadctx_smoke
//
// `threadctx_smoke reset` runs the device-reset case alone, and needs a server
// of its own with no other session on it.

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"

namespace {

int g_failures = 0;
std::mutex g_fail_mu;

void fail_at(const char* file, int line, const char* what) {
  std::lock_guard<std::mutex> lk(g_fail_mu);
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
  g_failures++;
}

#define EXPECT(cond, what)                          \
  do {                                              \
    if (!(cond)) fail_at(__FILE__, __LINE__, what); \
  } while (0)

#define CHECK(call)                                                    \
  do {                                                                 \
    CUresult r_ = (call);                                              \
    if (r_ != CUDA_SUCCESS) {                                          \
      char msg_[256];                                                  \
      std::snprintf(msg_, sizeof(msg_), "%s -> %d", #call, (int)r_);   \
      fail_at(__FILE__, __LINE__, msg_);                               \
    }                                                                  \
  } while (0)

#define CHECK_RT(call)                                                 \
  do {                                                                 \
    cudaError_t e_ = (call);                                           \
    if (e_ != cudaSuccess) {                                           \
      char msg_[256];                                                  \
      std::snprintf(msg_, sizeof(msg_), "%s -> %d", #call, (int)e_);   \
      fail_at(__FILE__, __LINE__, msg_);                               \
    }                                                                  \
  } while (0)

// One of the fake driver's counters, as it last published them to
// RGPU_FAKE_STATS, or -1 if it cannot be read. The server publishes before it
// replies, so after a call has returned the file includes it.
long fake_counter(const char* name) {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  if (!path) return -1;
  std::FILE* f = std::fopen(path, "r");
  if (!f) return -1;
  char buf[512] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  const std::string line = std::string(" ") + buf;
  const std::string key = std::string(" ") + name + "=";
  const size_t at = line.find(key);
  if (at == std::string::npos) return -1;
  return std::strtol(line.c_str() + at + key.size(), nullptr, 10);
}

int ordinal_of(CUdeviceptr p) {
  int ordinal = -1;
  CHECK(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p));
  return ordinal;
}

// Lets two threads take turns: each waits for the stage it acts in. The
// point of every case is the order in which the threads' requests reach the
// server, so the order is forced rather than hoped for.
class Turns {
 public:
  void advance(int to) {
    std::lock_guard<std::mutex> lk(mu_);
    stage_ = to;
    cv_.notify_all();
  }
  void await(int at) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stage_ >= at; });
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int stage_ = 0;
};

// Test 2 of the design, and the smallest regression test for all of it: a
// context one thread makes current is still current on that thread after
// another thread has made a different one current.
void currency_is_read_back() {
  std::printf("-- a thread's current context survives another thread's\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  EXPECT(p0 && p1 && p0 != p1, "the two devices share a primary context");

  Turns turns;
  CUcontext a_sees = nullptr, b_sees = nullptr;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    turns.advance(1);
    turns.await(2);
    CHECK(cuCtxGetCurrent(&a_sees));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxGetCurrent(&b_sees));
  });
  a.join();
  b.join();
  EXPECT(a_sees == p0,
         "thread A's current context was changed by thread B's selection");
  EXPECT(b_sees == p1,
         "thread B's current context was changed by thread A's query");

  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// Test 1 of the design: issue #2's scenario, through the runtime API, and
// asserted by where the memory went. Thread A's second allocation used to land
// on device 1, because the runtime remembers per thread that its context is
// current and so sends no selection, while the one server thread serving both
// had been moved to device 1 by thread B.
void issue_scenario_by_placement() {
  std::printf("-- issue #2: an allocation lands on its own thread's device\n");
  Turns turns;
  void* a1 = nullptr;
  void* a2 = nullptr;
  void* b1 = nullptr;
  int a1_on = -1, a2_on = -1, b1_on = -1;
  std::thread a([&] {
    CHECK_RT(cudaSetDevice(0));
    CHECK_RT(cudaMalloc(&a1, 64));
    turns.advance(1);
    turns.await(2);
    CHECK_RT(cudaMalloc(&a2, 64));
    a1_on = ordinal_of(reinterpret_cast<CUdeviceptr>(a1));
    a2_on = ordinal_of(reinterpret_cast<CUdeviceptr>(a2));
    CHECK_RT(cudaFree(a1));
    CHECK_RT(cudaFree(a2));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK_RT(cudaSetDevice(1));
    CHECK_RT(cudaMalloc(&b1, 64));
    b1_on = ordinal_of(reinterpret_cast<CUdeviceptr>(b1));
    turns.advance(2);
    turns.await(3);
    CHECK_RT(cudaFree(b1));
  });
  a.join();
  b.join();
  EXPECT(a1_on == 0, "thread A's first allocation is not on device 0");
  EXPECT(b1_on == 1, "thread B's allocation is not on device 1");
  EXPECT(a2_on == 0,
         "thread A's allocation after thread B selected device 1 landed on "
         "device 1");
}

// Test 4 of the design: a run of calls queued without replies by one thread
// and flushed by another thread's call runs under the context of the thread
// that queued it. The memsets succeed under any context, because the driver
// infers placement from the pointer, so what is checked is what the fake saw:
// every device-memory write that ran under a context other than its memory's
// own is counted.
//
// And that they were batched at all. With batching off each memset is a round
// trip that runs under its own thread's context before the other thread calls,
// and every other check here passes without testing anything. So the fake's
// memset count must not move while they are queued, and must move by all of
// them when the other thread's call flushes them. RGPU_BATCH=1 is pinned in
// run_smoke.sh, and this is what says so if it stops meaning batching.
void batch_flushed_by_another_thread() {
  std::printf("-- a batch flushed by another thread runs under its own "
              "thread's context\n");
  constexpr int kQueued = 8;
  constexpr size_t kBytes = 64;
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  long before = -1, after = -1;
  long runs_before = -1, runs_queued = -1, runs_flushed = -1;
  CUdeviceptr mem = 0;
  CUcontext b_sees = nullptr;
  CUresult flushed_by = CUDA_ERROR_UNKNOWN;
  unsigned char last = 0;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    CHECK(cuMemAlloc(&mem, kBytes));
    turns.advance(1);
    turns.await(2);
    // Every reply so far has been read, so nothing is still running.
    before = fake_counter("crossctx");
    runs_before = fake_counter("memsets");
    for (int i = 1; i <= kQueued; i++) {
      CHECK(cuMemsetD8Async(mem, static_cast<unsigned char>(i), kBytes,
                            nullptr));
    }
    // Still in this process: nothing has been written to the server.
    runs_queued = fake_counter("memsets");
    // Nothing from this thread until the other one has flushed the batch.
    turns.advance(3);
    turns.await(4);
    CHECK(cuMemcpyDtoH(&last, mem, 1));
    CHECK(cuMemFree(mem));
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(5);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    turns.advance(2);
    turns.await(3);
    flushed_by = cuCtxGetCurrent(&b_sees);
    after = fake_counter("crossctx");
    runs_flushed = fake_counter("memsets");
    turns.advance(4);
    turns.await(5);
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();

  EXPECT(before >= 0 && after >= 0 && runs_before >= 0,
         "could not read the fake's counters (is RGPU_FAKE_STATS set?)");
  if (runs_queued != runs_before) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "%ld of the memsets ran before another thread's call "
                  "flushed them: they were not batched (is RGPU_BATCH=1?)",
                  runs_queued - runs_before);
    fail_at(__FILE__, __LINE__, msg);
  }
  if (runs_flushed != runs_before + kQueued) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "the other thread's call flushed %ld memsets, not the %d "
                  "queued",
                  runs_flushed - runs_queued, kQueued);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(flushed_by == CUDA_SUCCESS,
         "the call that flushed the batch reported a failure");
  EXPECT(b_sees == p1, "the flushing thread's own context changed");
  EXPECT(last == kQueued, "the queued memsets did not all run");
  if (after != before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "%ld of the queued memsets ran under the flushing thread's "
                  "context instead of their own",
                  after - before);
    fail_at(__FILE__, __LINE__, msg);
  }

  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// A call sent without a reply has nowhere to report its failure, so the server
// holds it for the next call that does reply - from the same client thread. A
// launch that thread A queued and thread B's call flushed fails on the server
// before B's call runs, and B must not be told: B's call succeeds, and A's next
// call that replies carries A's failure, once.
void deferred_error_stays_with_its_thread() {
  std::printf("-- a failure with no reply surfaces on the thread that caused "
              "it\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));

  Turns turns;
  CUresult queued = CUDA_ERROR_UNKNOWN, b_call = CUDA_ERROR_UNKNOWN;
  CUresult a_first = CUDA_SUCCESS, a_second = CUDA_ERROR_UNKNOWN;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    // A null function: the server's driver rejects the launch, and with no
    // reply the failure waits. reconnect_smoke rests on the same launch.
    unsigned char packed[28] = {0};
    size_t packed_size = sizeof(packed);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                     CU_LAUNCH_PARAM_END};
    queued = cuLaunchKernel(nullptr, 8, 1, 1, 256, 1, 1, 0, nullptr, nullptr,
                            extra);
    turns.advance(1);
    turns.await(2);
    a_first = cuCtxSynchronize();
    a_second = cuCtxSynchronize();
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    int count = 0;
    b_call = cuDeviceGetCount(&count);  // flushes the launch ahead of itself
    turns.advance(2);
    turns.await(3);
  });
  a.join();
  b.join();
  EXPECT(queued == CUDA_SUCCESS,
         "the launch was not queued without a reply, so nothing was deferred");
  if (b_call != CUDA_SUCCESS) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "thread B's call returned %d: it was handed the failure of "
                  "a launch thread A queued",
                  (int)b_call);
    fail_at(__FILE__, __LINE__, msg);
  }
  if (a_first != CUDA_ERROR_INVALID_HANDLE) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "thread A's next call that replies returned %d, not its "
                  "launch's CUDA_ERROR_INVALID_HANDLE",
                  (int)a_first);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(a_second == CUDA_SUCCESS,
         "thread A's deferred failure was reported more than once");
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// The stream capture mode is thread state in CUDA too - "A thread's mode is one
// of the following", GLOBAL being "the default mode" - so each client thread
// keeps its own, and the mode one thread exchanges is not the mode another
// thread is given back. Both threads have the same context current, so the
// server cannot learn from the context alone that it has a different thread's
// state to put back.
void capture_modes_are_per_thread() {
  std::printf("-- each thread keeps its own stream capture mode, even on a "
              "shared context\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));

  Turns turns;
  CUstreamCaptureMode a_first = CU_STREAM_CAPTURE_MODE_RELAXED;
  CUstreamCaptureMode a_second = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUstreamCaptureMode b_first = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
  CUstreamCaptureMode b_second = CU_STREAM_CAPTURE_MODE_GLOBAL;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    turns.advance(1);
    turns.await(2);
    a_first = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
    CHECK(cuThreadExchangeStreamCaptureMode(&a_first));
    turns.advance(3);
    turns.await(4);
    a_second = CU_STREAM_CAPTURE_MODE_GLOBAL;
    CHECK(cuThreadExchangeStreamCaptureMode(&a_second));
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(5);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p0));
    turns.advance(2);
    turns.await(3);
    b_first = CU_STREAM_CAPTURE_MODE_RELAXED;
    CHECK(cuThreadExchangeStreamCaptureMode(&b_first));
    turns.advance(4);
    turns.await(5);
    b_second = CU_STREAM_CAPTURE_MODE_GLOBAL;
    CHECK(cuThreadExchangeStreamCaptureMode(&b_second));
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  EXPECT(a_first == CU_STREAM_CAPTURE_MODE_GLOBAL,
         "thread A's capture mode did not start as the default");
  EXPECT(b_first == CU_STREAM_CAPTURE_MODE_GLOBAL,
         "thread B was given back the capture mode thread A chose");
  EXPECT(a_second == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
         "thread A was not given back its own capture mode after thread B "
         "chose another");
  EXPECT(b_second == CU_STREAM_CAPTURE_MODE_RELAXED,
         "thread B was not given back its own capture mode");
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// The price of all this for a client with one thread: nothing. The server only
// makes a thread's context current when it differs from the one it last made
// current, so a thread calling on its own pays no cuCtxSetCurrent beyond the
// ones it asks for itself.
void one_thread_costs_no_switches() {
  std::printf("-- one thread's calls cost no context switches\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuCtxSetCurrent(p0));
  // A mode other than the default, so that a server putting it back before
  // every request would show.
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
  CHECK(cuThreadExchangeStreamCaptureMode(&mode));
  const long before = fake_counter("ctxsets");
  const long swaps_before = fake_counter("modeswaps");
  constexpr size_t kBytes = 256;
  for (int i = 0; i < 20; i++) {
    CUdeviceptr d = 0;
    CHECK(cuMemAlloc(&d, kBytes));
    CHECK(cuMemsetD8Async(d, 0x5a, kBytes, nullptr));
    unsigned char out = 0;
    CHECK(cuMemcpyDtoH(&out, d, 1));
    EXPECT(out == 0x5a, "a memset did not reach the memory");
    CUdevice dev = -1;
    CHECK(cuCtxGetDevice(&dev));
    CUcontext cur = nullptr;
    CHECK(cuCtxGetCurrent(&cur));
    CHECK(cuMemFree(d));
  }
  const long after = fake_counter("ctxsets");
  const long swaps_after = fake_counter("modeswaps");
  EXPECT(before >= 0 && after >= 0 && swaps_before >= 0 && swaps_after >= 0,
         "could not read the fake's counters (is RGPU_FAKE_STATS set?)");
  if (after != before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "a single thread's 120 calls cost %ld context switches",
                  after - before);
    fail_at(__FILE__, __LINE__, msg);
  }
  if (swaps_after != swaps_before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "a single thread's 120 calls cost %ld capture mode exchanges",
                  swaps_after - swaps_before);
    fail_at(__FILE__, __LINE__, msg);
  }
  mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CHECK(cuThreadExchangeStreamCaptureMode(&mode));
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// A context one thread made current and another thread then destroyed, and
// whose address the next context created then got. The first thread's next
// call must not run under that new context, nor under whatever the other
// thread left current: CUDA leaves a destroyed context current to the threads
// it was current to and fails their calls with
// CUDA_ERROR_CONTEXT_IS_DESTROYED, and the thread can select again.
//
// The server runs with RGPU_FAKE_REUSE_CONTEXTS=1, so the fake hands the
// destroyed context's handle to the next create, as a real driver may.
void context_destroyed_by_another_thread() {
  std::printf("-- a context destroyed by another thread is not replaced by "
              "the context that took its address\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, other = nullptr;
  CUresult after_destroy = CUDA_SUCCESS;
  int landed_on = -1;
  CUcontext a_sees = reinterpret_cast<CUcontext>(0xbadull);
  CUresult device_query = CUDA_SUCCESS;
  int reselected_on = -1;
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    turns.advance(1);
    turns.await(2);
    CUdeviceptr d = 0;
    after_destroy = cuMemAlloc(&d, 64);
    if (after_destroy == CUDA_SUCCESS) {
      landed_on = ordinal_of(d);
      cuMemFree(d);
    }
    CUdevice device = -1;
    device_query = cuCtxGetDevice(&device);
    CHECK(cuCtxGetCurrent(&a_sees));
    CHECK(cuCtxSetCurrent(p0));
    CUdeviceptr e = 0;
    CHECK(cuMemAlloc(&e, 64));
    if (e) {
      reselected_on = ordinal_of(e);
      CHECK(cuMemFree(e));
    }
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    CHECK(cuCtxDestroy(made));
    // Pushed on top of p1, on device 1, so that running under it shows.
    CHECK(cuCtxCreate(&other, 0, 1));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxDestroy(other));
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  EXPECT(other == made,
         "the fake did not reuse the destroyed context's handle (is "
         "RGPU_FAKE_REUSE_CONTEXTS=1 set on the server?)");
  if (after_destroy != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "a call after another thread destroyed this thread's context "
                  "returned %d (on device %d), not "
                  "CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)after_destroy, landed_on);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(device_query != CUDA_SUCCESS,
         "a thread whose context was destroyed was bound to another thread's "
         "context");
  // CUDA leaves the destroyed context current, and names it. Here that is
  // also the address the new context took, which is why binding is checked
  // above by what a call does, not by the handle.
  EXPECT(a_sees == made,
         "cuCtxGetCurrent on a thread whose context was destroyed did not name "
         "the destroyed context");
  EXPECT(reselected_on == 0,
         "a thread whose context was destroyed could not select another");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// A thread whose context was destroyed under it recovers the way it would in
// CUDA: its very first call can be a selection, which succeeds, and what it
// pushes on top of the destroyed context can be popped to find the destroyed
// one current again.
void recovery_after_destroy() {
  std::printf("-- a thread recovers from a context destroyed under it\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, popped = nullptr;
  CUresult pushed = CUDA_ERROR_UNKNOWN;
  CUresult under_push = CUDA_ERROR_UNKNOWN, after_pop = CUDA_SUCCESS;
  int pushed_on = -1, set_on = -1;
  CUcontext at_end = reinterpret_cast<CUcontext>(0xbadull);
  CUcontext destroyed_sees = nullptr;
  CUresult about_handle = CUDA_SUCCESS, about_event_ctx = CUDA_SUCCESS;
  CUresult peer_on = CUDA_SUCCESS, peer_off = CUDA_SUCCESS;
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    turns.advance(1);
    turns.await(2);
    pushed = cuCtxPushCurrent(p0);
    CUdeviceptr d = 0;
    under_push = cuMemAlloc(&d, 64);
    if (under_push == CUDA_SUCCESS) {
      pushed_on = ordinal_of(d);
      CHECK(cuMemFree(d));
    }
    CHECK(cuCtxPopCurrent(&popped));
    CHECK(cuCtxGetCurrent(&destroyed_sees));
    // A call that reports on a handle it was handed keeps its own answer
    // about that handle, whatever the thread has current.
    unsigned int version = 0;
    about_handle = cuCtxGetApiVersion(
        reinterpret_cast<CUcontext>(0xdeadbeefull), &version);
    about_event_ctx = cuCtxRecordEvent(
        reinterpret_cast<CUcontext>(0xdeadbeefull), nullptr);
    // Peer access is from the current context, and is refused with
    // CUDA_ERROR_INVALID_CONTEXT "if there is no current context": about the
    // current context too, so corrected like any other call. The peer here is
    // a live context.
    peer_on = cuCtxEnablePeerAccess(p1, 0);
    peer_off = cuCtxDisablePeerAccess(p1);
    CUdeviceptr e = 0;
    after_pop = cuMemAlloc(&e, 64);
    if (after_pop == CUDA_SUCCESS) cuMemFree(e);
    // Replaces the destroyed context at the top of the stack.
    CHECK(cuCtxSetCurrent(p1));
    CUdeviceptr f = 0;
    CHECK(cuMemAlloc(&f, 64));
    if (f) {
      set_on = ordinal_of(f);
      CHECK(cuMemFree(f));
    }
    CHECK(cuCtxSetCurrent(nullptr));
    CHECK(cuCtxGetCurrent(&at_end));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxDestroy(made));
    turns.advance(2);
    turns.await(3);
  });
  a.join();
  b.join();
  if (pushed != CUDA_SUCCESS) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "the first call after this thread's context was destroyed, a "
                  "push, returned %d",
                  (int)pushed);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(under_push == CUDA_SUCCESS && pushed_on == 0,
         "an allocation under the pushed context did not land on its device");
  EXPECT(popped == p0, "the pop did not return the pushed context");
  EXPECT(destroyed_sees == made,
         "after popping back to the destroyed context, cuCtxGetCurrent did not "
         "name it, as CUDA does");
  if (about_handle != CUDA_ERROR_INVALID_CONTEXT) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "cuCtxGetApiVersion of a handle naming no context returned "
                  "%d, not its own CUDA_ERROR_INVALID_CONTEXT",
                  (int)about_handle);
    fail_at(__FILE__, __LINE__, msg);
  }
  if (about_event_ctx != CUDA_ERROR_INVALID_CONTEXT) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "cuCtxRecordEvent in a handle naming no context returned "
                  "%d, not its own CUDA_ERROR_INVALID_CONTEXT",
                  (int)about_event_ctx);
    fail_at(__FILE__, __LINE__, msg);
  }
  if (after_pop != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "after popping back to the destroyed context an allocation "
                  "returned %d, not CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)after_pop);
    fail_at(__FILE__, __LINE__, msg);
  }
  if (peer_on != CUDA_ERROR_CONTEXT_IS_DESTROYED ||
      peer_off != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "peer access from a destroyed current context returned %d "
                  "(enable) and %d (disable), not "
                  "CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)peer_on, (int)peer_off);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(set_on == 1,
         "selecting over a destroyed context did not take effect");
  EXPECT(at_end == nullptr,
         "the thread's stack was not empty after it popped everything");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// cuCtxDetach destroys a created context - its usage count is 1, and the
// server refuses cuCtxAttach, the only call that raises it - so it has to be
// seen as a destroy just as cuCtxDestroy is. Thread A creates a context;
// thread B makes it current, detaches it, and creates another on device 1,
// which the fake hands the detached context's handle. A's next call must not
// run under the new context.
void detached_context_is_never_rebound() {
  std::printf("-- a detached context is not replaced by the context that "
              "took its address\n");
  CUcontext p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, other = nullptr;
  CUresult detached = CUDA_ERROR_UNKNOWN;
  CUresult after = CUDA_SUCCESS, device_query = CUDA_SUCCESS;
  int landed_on = -1;
  CUdevice device = -1;
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    turns.advance(1);
    turns.await(2);
    CUdeviceptr d = 0;
    after = cuMemAlloc(&d, 64);
    if (after == CUDA_SUCCESS) {
      landed_on = ordinal_of(d);
      cuMemFree(d);
    }
    device_query = cuCtxGetDevice(&device);
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(made));
    detached = cuCtxDetach(made);
    CHECK(cuCtxSetCurrent(p1));
    CHECK(cuCtxCreate(&other, 0, 1));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxDestroy(other));
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  EXPECT(detached == CUDA_SUCCESS, "cuCtxDetach of a created context failed");
  EXPECT(other == made,
         "the fake did not reuse the detached context's handle (is "
         "RGPU_FAKE_REUSE_CONTEXTS=1 set on the server?)");
  if (after != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "a call after another thread detached this thread's context "
                  "returned %d (on device %d), not "
                  "CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)after, landed_on);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(device_query != CUDA_SUCCESS,
         "a thread whose context was detached was bound to the context that "
         "took its address");
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// Calls the server refuses because they would put a context beyond what it
// tracks: cuCtxAttach, which would make cuCtxDetach something other than a
// destroy, and the green-context family, whose contexts carry no record and
// whose destroy releases a primary context unseen. Refused with
// CUDA_ERROR_NOT_SUPPORTED. The fake has none of them either, so run_smoke.sh
// checks the server's log for the refusals too.
void refused_context_calls() {
  std::printf("-- cuCtxAttach and green contexts are refused\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuCtxSetCurrent(p0));
  CUcontext attached = nullptr;
  EXPECT(cuCtxAttach(&attached, 0) == CUDA_ERROR_NOT_SUPPORTED,
         "cuCtxAttach was not refused with CUDA_ERROR_NOT_SUPPORTED");
  CUgreenCtx green = nullptr;
  EXPECT(cuGreenCtxCreate(&green, nullptr, 0, 0) == CUDA_ERROR_NOT_SUPPORTED,
         "cuGreenCtxCreate was not refused");
  CUcontext from_green = nullptr;
  EXPECT(cuCtxFromGreenCtx(&from_green, green) == CUDA_ERROR_NOT_SUPPORTED,
         "cuCtxFromGreenCtx was not refused");
  CUstream stream = nullptr;
  EXPECT(cuGreenCtxStreamCreate(&stream, green, 0, 0) ==
             CUDA_ERROR_NOT_SUPPORTED,
         "cuGreenCtxStreamCreate was not refused");
  EXPECT(cuGreenCtxRecordEvent(green, nullptr) == CUDA_ERROR_NOT_SUPPORTED,
         "cuGreenCtxRecordEvent was not refused");
  EXPECT(cuGreenCtxWaitEvent(green, nullptr) == CUDA_ERROR_NOT_SUPPORTED,
         "cuGreenCtxWaitEvent was not refused");
  EXPECT(cuGreenCtxDestroy(green) == CUDA_ERROR_NOT_SUPPORTED,
         "cuGreenCtxDestroy was not refused");
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// Issue #2's regression in C1, exactly: a thread that creates a context and
// pushes another on top of it, with a background thread's context-free call in
// between, pops back to the context it created. The background thread has no
// context of its own, and making that true on the one server thread must not
// throw away the first thread's stack.
void push_pop_across_a_background_call() {
  std::printf("-- a thread's context stack survives another thread's "
              "context-free call\n");
  CUcontext p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, popped = nullptr;
  CUcontext after_pop = reinterpret_cast<CUcontext>(0xbadull);
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    CHECK(cuCtxPushCurrent(p1));
    turns.advance(1);
    turns.await(2);
    CHECK(cuCtxPopCurrent(&popped));
    CHECK(cuCtxGetCurrent(&after_pop));
    CHECK(cuCtxDestroy(made));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    int count = 0;
    CHECK(cuDeviceGetCount(&count));
    turns.advance(2);
    turns.await(3);
  });
  a.join();
  b.join();
  EXPECT(popped == p1, "the pop did not return the pushed context");
  EXPECT(after_pop == made,
         "popping did not make the context underneath current again: another "
         "thread's call threw the stack away");
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// cuCtxCreate pushes the context it makes. A thread that creates a dozen
// contexts on top of its own has a stack a dozen deep, and a thread with no
// context must still be served in between - with no context, at the cost of
// no more than the serving thread's one entry - after which the first thread
// destroys its contexts one by one and finds each one underneath current.
void deep_stack_of_created_contexts() {
  std::printf("-- a stack of created contexts survives a thread with none\n");
  constexpr int kDepth = 12;
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));

  Turns turns;
  CUcontext made[kDepth] = {};
  int wrong_top = 0;
  CUcontext at_end = reinterpret_cast<CUcontext>(0xbadull);
  CUresult b_call = CUDA_ERROR_UNKNOWN;
  CUcontext b_sees = reinterpret_cast<CUcontext>(0xbadull);
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    for (int i = 0; i < kDepth; i++) CHECK(cuCtxCreate(&made[i], 0, i % 2));
    turns.advance(1);
    turns.await(2);
    for (int i = kDepth - 1; i >= 0; i--) {
      CUcontext top = nullptr;
      CHECK(cuCtxGetCurrent(&top));
      if (top != made[i]) wrong_top++;
      CHECK(cuCtxDestroy(made[i]));
    }
    CHECK(cuCtxGetCurrent(&at_end));
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    int count = 0;
    b_call = cuDeviceGetCount(&count);
    CHECK(cuCtxGetCurrent(&b_sees));
    turns.advance(2);
    turns.await(3);
  });
  a.join();
  b.join();
  if (b_call != CUDA_SUCCESS) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "a thread with no context was refused (%d) while another "
                  "thread had a deep stack",
                  (int)b_call);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(b_sees == nullptr,
         "a thread with no context saw another thread's created context");
  EXPECT(wrong_top == 0,
         "destroying a created context did not uncover the one below it");
  EXPECT(at_end == p0,
         "destroying every created context did not return to the thread's "
         "own context");
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// Test 3 of the design: context stacks are per thread. A pushes one context
// over its own; B, with a different context of its own, pushes and pops; A
// pops and is back where it started, and so is B.
void stacks_are_per_thread() {
  std::printf("-- each thread pops back to its own context\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext a_popped = nullptr, b_popped = nullptr;
  CUcontext a_after = reinterpret_cast<CUcontext>(0xbadull);
  CUcontext b_after = reinterpret_cast<CUcontext>(0xbadull);
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    CHECK(cuCtxPushCurrent(p1));
    turns.advance(1);
    turns.await(2);
    CHECK(cuCtxPopCurrent(&a_popped));
    CHECK(cuCtxGetCurrent(&a_after));
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    CHECK(cuCtxPushCurrent(p0));
    CHECK(cuCtxPopCurrent(&b_popped));
    CHECK(cuCtxGetCurrent(&b_after));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  EXPECT(b_popped == p0, "thread B's pop did not return what it pushed");
  EXPECT(b_after == p1, "thread B did not pop back to its own context");
  EXPECT(a_popped == p1, "thread A's pop did not return what it pushed");
  EXPECT(a_after == p0,
         "thread A did not pop back to its own context after thread B pushed "
         "and popped");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// A device reset destroys everything in the primary context but not the
// context: "Resetting the primary context does not release it", and its handle
// survives. So every thread that had it current still has it, the one that
// reset the device and the others alike, and each goes on allocating in it
// without selecting again.
//
// Its own server and nothing else connected to it, because a reset is refused
// while another session is live.
void reset_by_another_thread() {
  std::printf("-- a device reset by one thread leaves every thread its "
              "context\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));

  Turns turns;
  CUresult b_reset = CUDA_ERROR_UNKNOWN, b_alloc = CUDA_ERROR_UNKNOWN;
  CUresult a_alloc = CUDA_ERROR_UNKNOWN;
  int b_on = -1, a_on = -1;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    CUdeviceptr before = 0;
    CHECK(cuMemAlloc(&before, 64));  // destroyed by the reset, never freed
    turns.advance(1);
    turns.await(2);
    CUdeviceptr d = 0;
    a_alloc = cuMemAlloc(&d, 64);
    if (a_alloc == CUDA_SUCCESS) {
      a_on = ordinal_of(d);
      CHECK(cuMemFree(d));
    }
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p0));
    b_reset = cuDevicePrimaryCtxReset(0);
    CUdeviceptr d = 0;
    b_alloc = cuMemAlloc(&d, 64);
    if (b_alloc == CUDA_SUCCESS) {
      b_on = ordinal_of(d);
      CHECK(cuMemFree(d));
    }
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  if (b_reset != CUDA_SUCCESS) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "the reset was refused (%d); is anything else connected to "
                  "this server?",
                  (int)b_reset);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(b_alloc == CUDA_SUCCESS && b_on == 0,
         "the thread that reset the device lost its context");
  if (a_alloc != CUDA_SUCCESS || a_on != 0) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "another thread's allocation after the reset returned %d "
                  "(on device %d), not success on device 0",
                  (int)a_alloc, a_on);
    fail_at(__FILE__, __LINE__, msg);
  }
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// --- the slot table, from a connection of our own ---------------------------
//
// A real client mints thread ids and announces their end itself, and cannot be
// made to send a call after its thread was announced gone, or to have more
// threads than a server allows, on cue. These cases speak the protocol
// directly, as their own session, and choose the thread ids.

class RawSession {
 public:
  ~RawSession() {
    if (fd_ >= 0) ::close(fd_);
  }

  bool open() {
    const char* env = std::getenv("RGPU_SERVER");
    std::string spec = env ? env : "127.0.0.1:9713";
    const size_t c = spec.rfind(':');
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(spec.substr(0, c).c_str(), spec.substr(c + 1).c_str(),
                      &hints, &res) != 0) {
      return false;
    }
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
      fd_ = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd_ < 0) continue;
      if (::connect(fd_, ai->ai_addr, ai->ai_addrlen) == 0) break;
      ::close(fd_);
      fd_ = -1;
    }
    ::freeaddrinfo(res);
    if (fd_ < 0) return false;
    timeval tv{5, 0};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    static std::atomic<uint64_t> n{0};
    rgpu::Handshake hello{};
    hello.magic = rgpu::kMagicHello;
    hello.version = rgpu::kProtocolVersion;
    if (session_lo_ == 0) {
      session_hi_ = 0x7c7c000000000000ull ^ static_cast<uint64_t>(::getpid());
      session_lo_ = ++n;
    }
    hello.session_hi = session_hi_;
    hello.session_lo = session_lo_;
    // Every request so far has been answered and read.
    hello.last_req_id = next_ - 1;
    const bool resuming = next_ > 1;
    rgpu::HandshakeReply reply{};
    return rgpu::write_exact(fd_, &hello, sizeof(hello)) &&
           rgpu::read_exact(fd_, &reply, sizeof(reply)) &&
           reply.version == rgpu::kProtocolVersion &&
           reply.resumed == (resuming ? 1u : 0u);
  }

  // Drops the connection and comes back to the same session, which the
  // server has kept waiting. The server is given a moment to see the old
  // connection close first, so that it is not still serving it when the new
  // one is handed over.
  bool reconnect() {
    ::close(fd_);
    fd_ = -1;
    ::usleep(300 * 1000);
    return open();
  }

  bool send(uint32_t api, uint32_t thread, const rgpu::Buffer& payload,
            bool no_reply) {
    rgpu::ReqHeader h{};
    h.magic = rgpu::kMagicReq;
    h.api_id = api;
    h.req_id = next_++;
    h.flags = no_reply ? static_cast<uint32_t>(rgpu::kFlagNoReply) : 0u;
    h.thread_id = thread;
    h.payload_len = static_cast<uint32_t>(payload.size());
    return rgpu::send_frame(fd_, h, payload);
  }

  // A call that replies. CUDA_ERROR_UNKNOWN if the connection failed.
  CUresult call(uint32_t api, uint32_t thread, const rgpu::Buffer& payload,
                rgpu::Buffer* rsp) {
    if (!send(api, thread, payload, false)) return CUDA_ERROR_UNKNOWN;
    rgpu::RspHeader rh{};
    std::vector<uint8_t> body;
    if (!rgpu::recv_frame(fd_, rgpu::kMagicRsp, &rh, &body)) {
      return CUDA_ERROR_UNKNOWN;
    }
    *rsp = rgpu::Buffer(std::move(body));
    return static_cast<CUresult>(rh.result);
  }

  CUresult set_current(uint32_t thread, CUcontext ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(ctx));
    return call(rgpu::API_cuCtxSetCurrent, thread, req, &rsp);
  }

  // The thread's current context, or a marker value if the call failed.
  CUcontext get_current(uint32_t thread, CUresult* r = nullptr) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    const CUresult got = call(rgpu::API_cuCtxGetCurrent, thread, req, &rsp);
    if (r) *r = got;
    uint64_t h = 0;
    if (got != CUDA_SUCCESS || !rsp.get(&h)) {
      return reinterpret_cast<CUcontext>(0xbadull);
    }
    return reinterpret_cast<CUcontext>(h);
  }

  CUresult retain(uint32_t thread, int dev, CUcontext* ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    req.put<int>(dev);
    const CUresult r = call(rgpu::API_cuDevicePrimaryCtxRetain, thread, req, &rsp);
    uint64_t h = 0;
    if (r == CUDA_SUCCESS && rsp.get(&h)) *ctx = reinterpret_cast<CUcontext>(h);
    return r;
  }

  CUresult release(uint32_t thread, int dev) {
    rgpu::Buffer req, rsp;
    req.put<int>(dev);
    return call(rgpu::API_cuDevicePrimaryCtxRelease_v2, thread, req, &rsp);
  }

  CUresult create(uint32_t thread, int dev, CUcontext* ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    req.put<unsigned int>(0);
    req.put<int>(dev);
    const CUresult r = call(rgpu::API_cuCtxCreate_v2, thread, req, &rsp);
    uint64_t h = 0;
    if (r == CUDA_SUCCESS && rsp.get(&h)) *ctx = reinterpret_cast<CUcontext>(h);
    return r;
  }

  CUresult destroy(uint32_t thread, CUcontext ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(ctx));
    return call(rgpu::API_cuCtxDestroy_v2, thread, req, &rsp);
  }

  CUresult alloc(uint32_t thread, size_t bytes, CUdeviceptr* ptr) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    req.put<size_t>(bytes);
    const CUresult r = call(rgpu::API_cuMemAlloc_v2, thread, req, &rsp);
    if (r == CUDA_SUCCESS && !rsp.get(ptr)) return CUDA_ERROR_UNKNOWN;
    return r;
  }

  CUresult free(uint32_t thread, CUdeviceptr ptr) {
    rgpu::Buffer req, rsp;
    req.put<CUdeviceptr>(ptr);
    return call(rgpu::API_cuMemFree_v2, thread, req, &rsp);
  }

  CUresult push(uint32_t thread, CUcontext ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(ctx));
    return call(rgpu::API_cuCtxPushCurrent_v2, thread, req, &rsp);
  }

  CUresult pop(uint32_t thread) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(0);
    return call(rgpu::API_cuCtxPopCurrent_v2, thread, req, &rsp);
  }

  // Announces threads gone, sent by `sender` without a reply, as the client
  // sends it.
  bool gone(uint32_t sender, const std::vector<uint32_t>& ids) {
    rgpu::Buffer req;
    req.put<uint32_t>(static_cast<uint32_t>(ids.size()));
    for (uint32_t id : ids) req.put<uint32_t>(id);
    return send(rgpu::API_rgpu_thread_gone, sender, req, true);
  }

 private:
  int fd_ = -1;
  uint32_t next_ = 1;
  uint64_t session_hi_ = 0, session_lo_ = 0;
};

// A primary context one session's thread has current, ended by another
// session's last release in the process and retained again by the first. The
// header says the last release "automatically reset[s]" the primary context:
// the context is emptied, not replaced, and its handle survives. So the thread
// keeps it current, as it would in CUDA, and goes on allocating without
// selecting again - even with another thread's call in between, so that the
// server has to put it back.
//
// Runs before anything in this process retains device 0, so that the second
// session's release really is the last one.
void primary_survives_another_sessions_release() {
  std::printf("-- a thread keeps its primary context current across another "
              "session's last release\n");
  constexpr uint32_t kHolder = 1, kOther = 2;
  RawSession s1, s2;
  if (!s1.open() || !s2.open()) {
    fail_at(__FILE__, __LINE__, "could not open sessions of our own");
    return;
  }
  CUcontext p0 = nullptr, s2_p0 = nullptr, again = nullptr;
  CHECK(s1.retain(kHolder, 0, &p0));
  CHECK(s1.set_current(kHolder, p0));
  CHECK(s2.retain(kHolder, 0, &s2_p0));
  // Not the last release: the second session still holds one.
  CHECK(s1.release(kHolder, 0));
  // Another thread of the first session calls, so the holder's context is no
  // longer the one its session's server thread has current.
  CUresult r = CUDA_ERROR_UNKNOWN;
  s1.get_current(kOther, &r);
  CHECK(r);
  // The last release in the process: device 0's primary context is reset.
  CHECK(s2.release(kHolder, 0));
  CHECK(s1.retain(kHolder, 0, &again));
  EXPECT(again == p0, "retaining again gave a different primary context");
  s1.get_current(kOther, &r);
  CHECK(r);

  CUdeviceptr d = 0;
  const CUresult alloc = s1.alloc(kHolder, 64, &d);
  if (alloc == CUDA_SUCCESS) {
    CHECK(s1.free(kHolder, d));
  } else {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "a thread's allocation in its primary context, after another "
                  "session's last release, returned %d",
                  (int)alloc);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(s1.get_current(kHolder, &r) == p0,
         "a thread lost its primary context to another session's release");
  CHECK(r);
  CHECK(s1.set_current(kHolder, nullptr));
  CHECK(s1.release(kHolder, 0));
}

// Late thread-local destructors on a client thread can call after the client
// has announced the thread gone. Such a call finds the context its thread had,
// not an empty slot and not another thread's context.
void late_call_after_notice(CUcontext p0, CUcontext p1) {
  std::printf("-- a call after its thread was announced gone keeps its "
              "context\n");
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  CHECK(s.set_current(7, p0));
  CHECK(s.set_current(8, p1));
  EXPECT(s.gone(8, {7}), "could not send the notice");
  CUresult r = CUDA_SUCCESS;
  EXPECT(s.get_current(7, &r) == p0,
         "a late call from a thread announced gone lost its context");
  CHECK(r);
  EXPECT(s.gone(8, {7}), "could not send the notice again");
  EXPECT(s.get_current(8) == p1, "the other thread's context changed");
}

// What the serving thread has current belongs to the thread, and the thread
// outlives every connection its session has, so it is kept with the session.
// Kept with the connection instead, it would start out as "nothing" after a
// reconnect while the driver still had the last thread's context current, and
// the next thread with no context of its own would run under that one - and
// then be recorded as having selected it.
void applied_context_outlives_a_connection(CUcontext p0) {
  std::printf("-- a thread with no context has none after a reconnect\n");
  constexpr uint32_t kSelector = 7, kEmpty = 8;
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  CHECK(s.set_current(kSelector, p0));
  if (!s.reconnect()) {
    fail_at(__FILE__, __LINE__, "could not resume the session");
    return;
  }
  CUresult r = CUDA_ERROR_UNKNOWN;
  EXPECT(s.get_current(kEmpty, &r) == nullptr,
         "a thread that selected nothing saw a context after a reconnect");
  CHECK(r);
  CUdeviceptr d = 0;
  const CUresult alloc = s.alloc(kEmpty, 64, &d);
  if (alloc != CUDA_ERROR_INVALID_CONTEXT) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "an allocation by a thread with no context, after a "
                  "reconnect, returned %d, not CUDA_ERROR_INVALID_CONTEXT",
                  (int)alloc);
    fail_at(__FILE__, __LINE__, msg);
    if (alloc == CUDA_SUCCESS) s.free(kEmpty, d);
  }
  EXPECT(s.get_current(kEmpty, &r) == nullptr,
         "a thread with no context was recorded as having the context "
         "another thread selected before the reconnect");
  EXPECT(s.get_current(kSelector, &r) == p0,
         "a thread lost its context across a reconnect");
  CHECK(s.set_current(kSelector, nullptr));
}

// The retired slots are bounded. A late call from a thread retired longer ago
// than the most recent 64 finds an empty slot - no context - and never another
// thread's. The bound is kRetiredSlots in server/main.cpp.
void retired_slots_are_bounded(CUcontext p0, CUcontext p1) {
  std::printf("-- only the most recently retired threads keep their "
              "context\n");
  constexpr uint32_t kRetiredSlots = 64;
  constexpr uint32_t kFirst = 100;
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  // One more than the bound, each retired straight after it selects, so the
  // live table never holds more than two and the cap is not what is tested.
  for (uint32_t id = kFirst; id <= kFirst + kRetiredSlots; id++) {
    CHECK(s.set_current(id, p0));
    EXPECT(s.gone(id, {id}), "could not send a notice");
  }
  // Each late call below brings its thread back to the live table, which
  // leaves the retired set as it was, and no notice follows.
  CHECK(s.set_current(99, p1));
  EXPECT(s.get_current(kFirst + 1) == p0,
         "the oldest thread still within the bound lost its context");
  EXPECT(s.get_current(kFirst + kRetiredSlots) == p0,
         "the most recently retired thread lost its context");
  CUresult r = CUDA_SUCCESS;
  EXPECT(s.get_current(kFirst, &r) == nullptr,
         "a late call from a thread retired past the bound did not get an "
         "empty slot");
  CHECK(r);
}

// A client mints thread ids and nothing authenticates it, so the live slots a
// session may hold are capped (RGPU_MAX_CLIENT_THREADS). A thread past the cap
// is refused, and a notice still gets through and makes room.
void live_slots_are_capped() {
  std::printf("-- a session's live client threads are capped\n");
  const char* env = std::getenv("RGPU_MAX_CLIENT_THREADS");
  const uint32_t cap = env ? static_cast<uint32_t>(std::atol(env)) : 0;
  if (cap == 0 || cap > 1024) {
    fail_at(__FILE__, __LINE__,
            "RGPU_MAX_CLIENT_THREADS must name the server's cap, and a small "
            "one");
    return;
  }
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  for (uint32_t id = 1; id <= cap; id++) {
    CUresult r = CUDA_ERROR_UNKNOWN;
    s.get_current(id, &r);
    if (r != CUDA_SUCCESS) {
      fail_at(__FILE__, __LINE__, "a thread within the cap was refused");
      return;
    }
  }
  CUresult r = CUDA_SUCCESS;
  s.get_current(cap + 1, &r);
  EXPECT(r == CUDA_ERROR_INVALID_VALUE,
         "a thread past the cap was not refused with CUDA_ERROR_INVALID_VALUE");
  // Refused too, and with no reply to say so. It is still that thread's
  // failure once it is let in.
  rgpu::Buffer query;
  query.put<uint8_t>(1);
  EXPECT(s.send(rgpu::API_cuCtxGetCurrent, cap + 1, query, true),
         "could not send a call without a reply");
  // Sent by the refused thread itself, which is how a client whose new thread
  // has just announced an old one would send it.
  EXPECT(s.gone(cap + 1, {1}), "could not send a notice");
  s.get_current(2, &r);
  EXPECT(r == CUDA_SUCCESS,
         "another thread was handed the failure of a call refused past the "
         "cap");
  s.get_current(cap + 1, &r);
  EXPECT(r == CUDA_ERROR_INVALID_VALUE,
         "a thread let in after a call of its own was refused past the cap "
         "was not told of it");
  s.get_current(cap + 1, &r);
  EXPECT(r == CUDA_SUCCESS,
         "a notice from a thread past the cap did not make room for it");
}

// The server's own refusals of a call sent without a reply are deferred like
// any failure, to the thread that sent the call: past its stack's cap, after
// its context was destroyed, and once it has been announced gone. A frame
// naming no thread has no thread to report to, and is reported to none.
void refusals_are_deferred_to_their_own_thread(CUcontext p0) {
  std::printf("-- a refused call without a reply reports to its own thread\n");
  const char* env = std::getenv("RGPU_MAX_CONTEXT_STACK");
  const long cap = env ? std::atol(env) : 0;
  if (cap < 2 || cap > 1024) {
    fail_at(__FILE__, __LINE__,
            "RGPU_MAX_CONTEXT_STACK must name the server's cap, and a small "
            "one");
    return;
  }
  constexpr uint32_t kPusher = 21, kMaker = 22, kGone = 23, kOther = 24;
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  // What each refused frame is followed by: a call from another thread, which
  // must not carry it, then two from the thread that sent it, the first of
  // which must carry it and the second not.
  auto expect_deferred = [&](uint32_t sender, CUresult want, const char* what) {
    CUresult r = CUDA_ERROR_UNKNOWN;
    s.get_current(kOther, &r);
    if (r != CUDA_SUCCESS) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "%s: another thread's call returned %d, carrying it",
                    what, (int)r);
      fail_at(__FILE__, __LINE__, msg);
    }
    s.get_current(sender, &r);
    if (r != want) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "%s: the sending thread's next call returned %d, not %d",
                    what, (int)r, (int)want);
      fail_at(__FILE__, __LINE__, msg);
    }
    s.get_current(sender, &r);
    if (r != CUDA_SUCCESS) {
      char msg[200];
      std::snprintf(msg, sizeof(msg), "%s: reported twice (%d)", what, (int)r);
      fail_at(__FILE__, __LINE__, msg);
    }
  };

  rgpu::Buffer query;
  query.put<uint8_t>(1);
  EXPECT(s.send(rgpu::API_cuCtxGetCurrent, 0, query, true),
         "could not send a frame naming no thread");
  CUresult r = CUDA_ERROR_UNKNOWN;
  s.get_current(kOther, &r);
  EXPECT(r == CUDA_SUCCESS,
         "a refused frame naming no thread had its failure reported on a "
         "thread");

  CHECK(s.set_current(kPusher, p0));
  for (long depth = 1; depth < cap; depth++) CHECK(s.push(kPusher, p0));
  rgpu::Buffer push;
  push.put<uint64_t>(reinterpret_cast<uint64_t>(p0));
  EXPECT(s.send(rgpu::API_cuCtxPushCurrent_v2, kPusher, push, true),
         "could not send a push");
  expect_deferred(kPusher, CUDA_ERROR_INVALID_VALUE, "a push past the cap");
  while (s.pop(kPusher) == CUDA_SUCCESS) {
  }

  CUcontext made = nullptr;
  CHECK(s.create(kMaker, 0, &made));
  CHECK(s.set_current(kOther, p0));
  CHECK(s.destroy(kOther, made));
  rgpu::Buffer alloc;
  alloc.put<uint8_t>(1);
  alloc.put<size_t>(64);
  EXPECT(s.send(rgpu::API_cuMemAlloc_v2, kMaker, alloc, true),
         "could not send an allocation");
  expect_deferred(kMaker, CUDA_ERROR_CONTEXT_IS_DESTROYED,
                  "an allocation after the thread's context was destroyed");
  CHECK(s.set_current(kMaker, nullptr));

  // A push of no context, which the server refuses itself, then the notice
  // that the thread is gone: its late call still carries the failure.
  rgpu::Buffer push_null;
  push_null.put<uint64_t>(0);
  EXPECT(s.send(rgpu::API_cuCtxPushCurrent_v2, kGone, push_null, true),
         "could not send a push");
  EXPECT(s.gone(kOther, {kGone}), "could not send the notice");
  expect_deferred(kGone, CUDA_ERROR_INVALID_CONTEXT,
                  "a push of no context from a thread since announced gone");
  CHECK(s.set_current(kOther, nullptr));
}

// A call that mutates the thread's context stack - set-current, push, pop -
// returns success once it has applied the change, and the client reads that
// success to keep its own idea of the stack in step with the server's. A
// deferred error held for the thread from an earlier no-reply call must not be
// folded into that success: doing so would tell the client the stack op failed
// while the server had already applied it, and the two would disagree about
// the stack from then on. The held error waits for the thread's next call whose
// reply the client reads as a plain result instead, and surfaces there once.
void observed_state_replies_never_carry_a_deferred_error(CUcontext p0) {
  std::printf("-- a held deferred error is not folded into a context-stack "
              "call, whose reply the client reads as state\n");
  constexpr uint32_t kActor = 41;
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  CHECK(s.set_current(kActor, p0));  // a stack to act on, no error held yet

  // Arms a deferred error on kActor: a push of no context, which the server
  // refuses itself (INVALID_CONTEXT), sent without a reply so the failure is
  // held for the thread's next reply-bearing call.
  auto arm = [&] {
    rgpu::Buffer push_null;
    push_null.put<uint64_t>(0);
    EXPECT(s.send(rgpu::API_cuCtxPushCurrent_v2, kActor, push_null, true),
           "could not arm a deferred error");
  };
  // The held error surfaces on the next plain call, once and only once.
  auto surfaces_once = [&](const char* what) {
    CUresult r = CUDA_ERROR_UNKNOWN;
    s.get_current(kActor, &r);
    if (r != CUDA_ERROR_INVALID_CONTEXT) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "%s: the held error did not surface on the next plain "
                    "call (got %d)",
                    what, (int)r);
      fail_at(__FILE__, __LINE__, msg);
    }
    s.get_current(kActor, &r);
    EXPECT(r == CUDA_SUCCESS, "the held error surfaced more than once");
  };
  auto not_folded = [&](CUresult got, const char* what) {
    if (got != CUDA_SUCCESS) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "%s returned %d: a deferred error was folded into a reply "
                    "the client reads as stack state, so the client would "
                    "think it failed while the server applied it",
                    what, (int)got);
      fail_at(__FILE__, __LINE__, msg);
    }
  };

  arm();
  not_folded(s.push(kActor, p0), "a push");
  surfaces_once("push");

  arm();
  not_folded(s.set_current(kActor, p0), "a set-current");
  surfaces_once("set-current");

  arm();
  not_folded(s.pop(kActor), "a pop");
  surfaces_once("pop");

  // The sequence left the stack coherent: one push above what set-current
  // seeded, so a last pop empties it and leaves the thread with no context.
  CHECK(s.pop(kActor));
  CHECK(s.set_current(kActor, nullptr));
}

// A thread's context stack lives in server memory, and the driver's own stack
// on the serving thread stays one deep, so nothing but a cap stops a client
// that pushes in a loop (RGPU_MAX_CONTEXT_STACK). A push or a create past it
// is refused with CUDA_ERROR_INVALID_VALUE, one of the codes cuCtxPushCurrent
// documents, and leaves the stack as it was; a pop makes room again.
void context_stacks_are_capped(CUcontext p0, long cap) {
  std::printf("-- a client thread's context stack is capped at %ld\n", cap);
  if (cap < 2 || cap > 1024) {
    fail_at(__FILE__, __LINE__,
            "RGPU_MAX_CONTEXT_STACK must name the server's cap, and a small "
            "one");
    return;
  }
  constexpr uint32_t kPusher = 5;
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  CHECK(s.set_current(kPusher, p0));
  for (long depth = 1; depth < cap; depth++) {
    if (s.push(kPusher, p0) != CUDA_SUCCESS) {
      fail_at(__FILE__, __LINE__, "a push within the cap was refused");
      return;
    }
  }
  const long contexts = fake_counter("contexts");
  const CUresult over = s.push(kPusher, p0);
  CUcontext made = nullptr;
  const CUresult created = s.create(kPusher, 0, &made);
  if (created == CUDA_SUCCESS) s.destroy(kPusher, made);
  EXPECT(over == CUDA_ERROR_INVALID_VALUE,
         "a push past the cap was not refused with CUDA_ERROR_INVALID_VALUE");
  EXPECT(created == CUDA_ERROR_INVALID_VALUE,
         "a create past the cap was not refused with CUDA_ERROR_INVALID_VALUE");
  EXPECT(fake_counter("contexts") == contexts,
         "a create past the cap made a context");
  CUresult r = CUDA_ERROR_UNKNOWN;
  EXPECT(s.get_current(kPusher, &r) == p0,
         "a refused push changed the thread's current context");
  CHECK(r);
  CHECK(s.pop(kPusher));
  EXPECT(s.push(kPusher, p0) == CUDA_SUCCESS,
         "a pop did not make room for another push");
  long popped = 0;
  while (popped <= cap && s.pop(kPusher) == CUDA_SUCCESS) popped++;
  EXPECT(popped == cap, "the stack did not hold exactly the cap's entries");
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(cuInit(0));
  if (argc > 1 && std::strcmp(argv[1], "reset") == 0) {
    reset_by_another_thread();
    // This server sets no cap of its own, so the default is what applies:
    // kept small, because every destroy sweeps every entry of every stack.
    // The default is max_stack_depth() in server/client_threads.cpp.
    constexpr long kDefaultStackCap = 64;
    CUcontext p0 = nullptr;
    CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
    context_stacks_are_capped(p0, kDefaultStackCap);
    CHECK(cuDevicePrimaryCtxRelease(0));
    if (g_failures) {
      std::printf("\nFAILED: %d check(s)\n", g_failures);
      return 1;
    }
    std::printf("\nPASS: a reset leaves every thread its context, and "
                "stacks are capped by default\n");
    return 0;
  }
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    std::fprintf(stderr, "FAIL: needs a server with two devices "
                         "(RGPU_FAKE_DEVICES=2)\n");
    return 1;
  }

  primary_survives_another_sessions_release();
  currency_is_read_back();
  issue_scenario_by_placement();
  batch_flushed_by_another_thread();
  deferred_error_stays_with_its_thread();
  capture_modes_are_per_thread();
  one_thread_costs_no_switches();
  context_destroyed_by_another_thread();
  recovery_after_destroy();
  detached_context_is_never_rebound();
  refused_context_calls();
  push_pop_across_a_background_call();
  stacks_are_per_thread();
  deep_stack_of_created_contexts();

  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  applied_context_outlives_a_connection(p0);
  late_call_after_notice(p0, p1);
  retired_slots_are_bounded(p0, p1);
  live_slots_are_capped();
  const char* stack_cap = std::getenv("RGPU_MAX_CONTEXT_STACK");
  context_stacks_are_capped(p0, stack_cap ? std::atol(stack_cap) : 0);
  refusals_are_deferred_to_their_own_thread(p0);
  observed_state_replies_never_carry_a_deferred_error(p0);
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: each client thread keeps its own context\n");
  return 0;
}
