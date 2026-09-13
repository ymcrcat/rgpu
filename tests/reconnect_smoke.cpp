// Checks that a dropped connection does not lose the GPU, without a GPU.
//
// The server is told to break the connection partway through, by frame count,
// so the break lands in the middle of a conversation rather than between two
// tidy operations. Two things have to survive that break. The allocation made
// before it must still hold what was written to it, because the session on the
// other side never went away. And a failure from a call that was sent without
// a reply must still be waiting to be reported, because that error belongs to
// the session too: a lost error turns a failed launch into an apparent
// success, which is worse than the drop it came from.
//
//   RGPU_DROP_AFTER=n rgpu-server-fake &
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./reconnect_smoke
//
// `reconnect_smoke threads` breaks the connection in the middle of a batch
// queued by two client threads with different contexts, and needs a server of
// its own with two devices and a stats file (see run_smoke.sh).

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

static int g_failures = 0;
static std::mutex g_fail_mu;

static void fail(const char* what) {
  std::lock_guard<std::mutex> lk(g_fail_mu);
  std::fprintf(stderr, "FAIL: %s\n", what);
  g_failures++;
}

#define CHECK(call)                                                       \
  do {                                                                    \
    CUresult r_ = (call);                                                 \
    if (r_ != CUDA_SUCCESS) {                                             \
      char m_[256];                                                       \
      std::snprintf(m_, sizeof(m_), "%s:%d: %s -> %d", __FILE__, __LINE__, \
                    #call, (int)r_);                                      \
      fail(m_);                                                           \
    }                                                                     \
  } while (0)

namespace {

// One of the fake driver's counters, as it last published them to
// RGPU_FAKE_STATS, or -1.
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

// Two client threads, each with its own device's primary context current,
// each queue a run of memsets without replies; the second thread's next call
// flushes both runs as one write, and the server breaks the connection partway
// through the first run (run_smoke.sh counts the frames). The client resumes
// the session and sends again every frame the server had not run, each with
// the thread id it was stamped with.
//
// Afterwards each thread's context must still be its own. The replayed
// memsets must have run under the context of the thread that queued them -
// the fake counts any that ran under another - and a table of client threads
// kept with the connection rather than the session would have lost both
// threads' contexts at the break: the replayed memsets would fail, with
// nothing current, and the flushing call would carry that failure.
constexpr int kQueued = 16;
constexpr size_t kBytes = 4096;

int threads_across_a_break() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    fail("threads needs a server with two devices (RGPU_FAKE_DEVICES=2)");
    return 1;
  }
  long cross_before = -1;

  Turns turns;
  CUcontext p0 = nullptr, p1 = nullptr;
  CUdeviceptr mem_a = 0, mem_b = 0;
  CUcontext a_sees = nullptr, b_sees = nullptr;
  CUresult flushed_by = CUDA_ERROR_UNKNOWN;
  unsigned char a_last = 0, b_last = 0;
  int a_on = -1, b_on = -1;
  std::thread a([&] {
    CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
    CHECK(cuCtxSetCurrent(p0));
    CHECK(cuMemAlloc(&mem_a, kBytes));
    turns.advance(1);
    turns.await(2);
    // Every reply so far has been read, so nothing is still running.
    cross_before = fake_counter("crossctx");
    for (int i = 1; i <= kQueued; i++) {
      CHECK(cuMemsetD8Async(mem_a, static_cast<unsigned char>(i), kBytes,
                            nullptr));
    }
    turns.advance(3);
    turns.await(4);
    CHECK(cuCtxGetCurrent(&a_sees));
    CHECK(cuMemcpyDtoH(&a_last, mem_a, 1));
    CUdeviceptr more = 0;
    CHECK(cuMemAlloc(&more, kBytes));
    CHECK(cuPointerGetAttribute(&a_on, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                more));
    CHECK(cuMemFree(more));
    CHECK(cuMemFree(mem_a));
    CHECK(cuCtxSetCurrent(nullptr));
    CHECK(cuDevicePrimaryCtxRelease(0));
    turns.advance(5);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
    CHECK(cuCtxSetCurrent(p1));
    CHECK(cuMemAlloc(&mem_b, kBytes));
    turns.advance(2);
    turns.await(3);
    for (int i = 1; i <= kQueued; i++) {
      CHECK(cuMemsetD8Async(mem_b, static_cast<unsigned char>(100 + i), kBytes,
                            nullptr));
    }
    // Flushes both runs, and is what the break interrupts.
    flushed_by = cuCtxGetCurrent(&b_sees);
    CHECK(cuMemcpyDtoH(&b_last, mem_b, 1));
    CUdeviceptr more = 0;
    CHECK(cuMemAlloc(&more, kBytes));
    CHECK(cuPointerGetAttribute(&b_on, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                more));
    CHECK(cuMemFree(more));
    turns.advance(4);
    turns.await(5);
    CHECK(cuMemFree(mem_b));
    CHECK(cuCtxSetCurrent(nullptr));
    CHECK(cuDevicePrimaryCtxRelease(1));
  });
  a.join();
  b.join();

  const long cross_after = fake_counter("crossctx");
  if (cross_before < 0 || cross_after < 0) {
    fail("could not read the fake's counters (is RGPU_FAKE_STATS set?)");
  } else if (cross_after != cross_before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "%ld memsets ran under another thread's context after the "
                  "break",
                  cross_after - cross_before);
    fail(msg);
  }
  if (flushed_by != CUDA_SUCCESS) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "the call that flushed the batch across the break returned "
                  "%d",
                  (int)flushed_by);
    fail(msg);
  }
  if (b_sees != p1) fail("thread B's context was not its own after the break");
  if (a_sees != p0) fail("thread A's context was not its own after the break");
  if (a_last != kQueued) fail("thread A's queued memsets did not all run");
  if (b_last != 100 + kQueued) {
    fail("thread B's queued memsets did not all run");
  }
  if (a_on != 0) fail("thread A's allocation after the break was not on device 0");
  if (b_on != 1) fail("thread B's allocation after the break was not on device 1");

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: two threads kept their own contexts across a break in "
              "the middle of their batch\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "threads") == 0) {
    return threads_across_a_break();
  }
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));

  // Allocate and fill before the break.
  const size_t n = 4096;
  std::vector<unsigned char> written(n);
  for (size_t i = 0; i < n; i++) written[i] = static_cast<unsigned char>(i * 7);
  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, n));
  CHECK(cuMemcpyHtoD(d, written.data(), n));

  // One launch that cannot work: the function handle is null, so the driver on
  // the far side rejects it. It is sent without a reply, so it cannot report
  // that here; the failure waits on the server for the next call that does
  // reply. Nothing else after this point fails, so if the error is reported
  // later it can only be this one, carried across the break.
  //
  // That it returns success here is the premise of the rest, not an aside: a
  // client that started rejecting a null handle locally would make everything
  // below meaningless, and it should say so here rather than at the sync.
  CUdeviceptr scratch = 0;
  CHECK(cuMemAlloc(&scratch, n));
  unsigned char packed[28] = {0};
  size_t packed_size = sizeof(packed);
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                   CU_LAUNCH_PARAM_END};
  CHECK(cuLaunchKernel(nullptr, 8, 1, 1, 256, 1, 1, 0, nullptr, nullptr, extra));

  // Then a stretch of asynchronous work that does succeed, long enough that
  // the server's drop point lands inside it. That is what makes this the
  // interesting case: the failure is recorded on one connection, and between
  // it and the break there is no call that asks for a reply, so the error has
  // nowhere to go except into the session.
  for (int i = 0; i < 64; i++) {
    CHECK(cuMemsetD8Async(scratch, static_cast<unsigned char>(i), n, nullptr));
  }

  // The first call that can carry an error back. It also flushes everything
  // queued above, so the break falls while the server is working through it
  // and this request is the one replayed afterwards. The code has to be the
  // launch's own, not merely non-zero: a connection that failed to come back
  // would fail here too, and that would prove nothing.
  const CUresult deferred = cuCtxSynchronize();
  if (deferred != CUDA_ERROR_INVALID_HANDLE) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "a launch failed with %d before the break; the first call "
                  "that could report it returned %d",
                  CUDA_ERROR_INVALID_HANDLE, deferred);
    fail(msg);
  }

  // Enough traffic that the session is exercised after it comes back.
  for (int i = 0; i < 40; i++) {
    int value = 0;
    CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev));
    if (value <= 0) {
      std::fprintf(stderr, "FAIL: warp size came back as %d\n", value);
      g_failures++;
      break;
    }
  }

  // The allocation from before the break has to still be there, with its
  // contents. A reconnect that silently started a new session would give a
  // valid-looking pointer to memory that had never been written.
  std::vector<unsigned char> read_back(n, 0);
  CHECK(cuMemcpyDtoH(read_back.data(), d, n));
  if (std::memcmp(read_back.data(), written.data(), n) != 0) {
    std::fprintf(stderr, "FAIL: memory from before the break did not survive\n");
    g_failures++;
  }

  CHECK(cuMemFree(scratch));
  CHECK(cuMemFree(d));
  CHECK(cuCtxDestroy(ctx));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: memory and a deferred error both survived a dropped "
              "connection\n");
  return 0;
}
