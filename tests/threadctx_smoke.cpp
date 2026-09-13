// Checks that each client thread keeps its own CUDA context across the wire,
// without a GPU.
//
// CUDA's current context belongs to the calling thread, but one server thread
// serves every thread of a client process. So the server keeps what each
// client thread last made current, puts it back before running that thread's
// request, and reads it back afterwards. Without that, the last thread to
// select a context wins for everybody, silently: issue #2.
//
// Every case asserts what the driver documents - what cuCtxGetCurrent returns,
// which device an allocation is on - and never that a use under the wrong
// context fails, because on hardware with unified addressing it most likely
// does not.
//
// Runs against its own rgpu-server-fake with two devices and batching on:
//
//   RGPU_FAKE_DEVICES=2 RGPU_FAKE_STATS=/tmp/s RGPU_MAX_CLIENT_THREADS=16
//     rgpu-server-fake 9723 &
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9723 RGPU_BATCH=1
//     RGPU_FAKE_STATS=/tmp/s RGPU_MAX_CLIENT_THREADS=16 ./threadctx_smoke

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <cuda.h>
#include <cuda_runtime_api.h>

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
    for (int i = 1; i <= kQueued; i++) {
      CHECK(cuMemsetD8Async(mem, static_cast<unsigned char>(i), kBytes,
                            nullptr));
    }
    // Nothing from this thread until the other one has flushed the batch.
    turns.advance(3);
    turns.await(4);
    CHECK(cuMemcpyDtoH(&last, mem, 1));
    CHECK(cuMemFree(mem));
    CHECK(cuCtxSetCurrent(nullptr));
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    turns.advance(2);
    turns.await(3);
    flushed_by = cuCtxGetCurrent(&b_sees);
    after = fake_counter("crossctx");
    turns.advance(4);
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();

  EXPECT(before >= 0 && after >= 0,
         "could not read the fake's counters (is RGPU_FAKE_STATS set?)");
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

// The price of all this for a client with one thread: nothing. The server only
// makes a thread's context current when it differs from the one it last made
// current, so a thread calling on its own pays no cuCtxSetCurrent beyond the
// ones it asks for itself.
void one_thread_costs_no_switches() {
  std::printf("-- one thread's calls cost no context switches\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuCtxSetCurrent(p0));
  const long before = fake_counter("ctxsets");
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
  EXPECT(before >= 0 && after >= 0,
         "could not read the fake's counters (is RGPU_FAKE_STATS set?)");
  if (after != before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "a single thread's 120 calls cost %ld context switches",
                  after - before);
    fail_at(__FILE__, __LINE__, msg);
  }
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
}

}  // namespace

int main() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    std::fprintf(stderr, "FAIL: needs a server with two devices "
                         "(RGPU_FAKE_DEVICES=2)\n");
    return 1;
  }

  currency_is_read_back();
  issue_scenario_by_placement();
  batch_flushed_by_another_thread();
  one_thread_costs_no_switches();

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: each client thread keeps its own context\n");
  return 0;
}
