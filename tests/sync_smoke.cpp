// Checks when cuStreamSynchronize waits for the server, in each of its three
// modes, and that the mode never changes a result. Without a GPU.
//
// A stream synchronize is a round trip. A program that synchronizes after
// every small upload - llama.cpp does, about twenty times per generated token
// - pays the link's latency each time, for nothing: an upload's bytes are
// taken when it is issued, so there is nothing left for the caller to wait
// for. So by default a synchronize that follows only uploads is queued and
// returns at once, and one that follows any other GPU work - here a memset -
// still waits. RGPU_LAZY_SYNC=0 makes every one wait; RGPU_LAZY_SYNC=1 queues
// every one.
//
// What a queued synchronize cannot do is report a failure, having no reply to
// carry it: the next call that waits reports it instead. By default that can
// only be an upload's own failure. cuCtxSynchronize always waits.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./sync_smoke uploads
//   ... RGPU_LAZY_SYNC=0 ./sync_smoke wait
//   ... RGPU_LAZY_SYNC=1 ./sync_smoke lazy
//
// tests/run_smoke.sh runs all three and counts the round trips each made:
// 24 waiting, 12 by default, none with everything queued. It also runs
// `sync_smoke threads` (two threads, two streams) and `sync_smoke exit` (work
// still queued when the process ends).

#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda.h>

static int g_failures = 0;

#define CHECK(call)                                                       \
  do {                                                                    \
    CUresult r_ = (call);                                                 \
    if (r_ != CUDA_SUCCESS) {                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,  \
                   #call, r_);                                            \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

#define EXPECT(cond, msg)                                                 \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);  \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

enum Mode { kWait, kUploads, kLazy };

// A synchronize after work that failed. If it waited it reports the failure;
// if it was queued it cannot, and the next call that waits does, once.
static void expect_failure(CUstream stream, bool reported_by_sync,
                           const char* what) {
  const CUresult synced = cuStreamSynchronize(stream);
  if (reported_by_sync) {
    if (synced == CUDA_SUCCESS) {
      std::fprintf(stderr, "FAIL: the synchronize after %s waited and should "
                           "have reported the failure\n", what);
      g_failures++;
    }
  } else {
    if (synced != CUDA_SUCCESS) {
      std::fprintf(stderr, "FAIL: the synchronize after %s was to be queued, "
                           "which has no reply, yet it returned %d\n", what, synced);
      g_failures++;
    }
    if (cuCtxSynchronize() == CUDA_SUCCESS) {
      std::fprintf(stderr, "FAIL: the failure of %s should have surfaced at "
                           "the next call that waits\n", what);
      g_failures++;
    }
  }
  CHECK(cuCtxSynchronize());
}

// Two threads, a stream each: one only uploads, the other only memsets, and
// each synchronizes its own stream after every call. The record of
// outstanding work is kept per stream, so the uploader's synchronizes are all
// queued whatever the other thread has outstanding, and exactly the other
// thread's wait. A record shared between the streams would make the uploader
// wait too.
static int two_threads() {
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));
  const size_t n = 4096;
  const int rounds = 200;
  CUstream up = nullptr, set = nullptr;
  CHECK(cuStreamCreate(&up, 0));
  CHECK(cuStreamCreate(&set, 0));
  CUdeviceptr uploaded = 0, filled = 0;
  CHECK(cuMemAlloc(&uploaded, n));
  CHECK(cuMemAlloc(&filled, n));
  if (g_failures) return 1;

  // In lockstep, so that every one of the uploader's synchronizes falls while
  // the other stream has a memset outstanding: left to run freely, the
  // uploader - which never waits - finishes before the other thread starts,
  // and the two streams are never busy at once.
  //   setter:   memset on its stream        (that stream is now busy)
  //   uploader: upload, then synchronize    (must be queued all the same)
  //   setter:   synchronize                 (waits)
  std::mutex mu;
  std::condition_variable cv;
  enum Turn { kMemset, kUpload, kSync } turn = kMemset;
  auto await = [&](Turn mine) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return turn == mine; });
  };
  auto pass = [&](Turn next) {
    { std::lock_guard<std::mutex> lk(mu); turn = next; }
    cv.notify_all();
  };

  int upload_failures = 0, memset_failures = 0;
  std::thread uploader([&] {
    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) upload_failures++;
    std::vector<unsigned char> host(n);
    for (int i = 1; i <= rounds; i++) {
      await(kUpload);
      std::memset(host.data(), i % 251, n);
      if (cuMemcpyHtoDAsync(uploaded, host.data(), n, up) != CUDA_SUCCESS ||
          cuStreamSynchronize(up) != CUDA_SUCCESS) {
        upload_failures++;
      }
      pass(kSync);
    }
  });
  std::thread setter([&] {
    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) memset_failures++;
    for (int i = 1; i <= rounds; i++) {
      await(kMemset);
      if (cuMemsetD8Async(filled, static_cast<unsigned char>(i % 251), n, set) !=
          CUDA_SUCCESS) {
        memset_failures++;
      }
      pass(kUpload);
      await(kSync);
      if (cuStreamSynchronize(set) != CUDA_SUCCESS) memset_failures++;
      pass(kMemset);
    }
  });
  uploader.join();
  setter.join();
  EXPECT(upload_failures == 0, "the uploading thread saw a failure");
  EXPECT(memset_failures == 0, "the memset thread saw a failure");

  std::vector<unsigned char> back(n, 0);
  size_t wrong = 0;
  CHECK(cuMemcpyDtoH(back.data(), uploaded, n));
  for (unsigned char b : back) wrong += b != rounds % 251;
  CHECK(cuMemcpyDtoH(back.data(), filled, n));
  for (unsigned char b : back) wrong += b != rounds % 251;
  EXPECT(wrong == 0, "each buffer should hold its thread's last write");

  CHECK(cuMemFree(uploaded));
  CHECK(cuMemFree(filled));
  CHECK(cuStreamDestroy(up));
  CHECK(cuStreamDestroy(set));
  CHECK(cuCtxDestroy(ctx));
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: stream synchronize, two threads on two streams\n");
  return 0;
}

// Queues work and leaves, without one call that waits. Nothing has been sent
// when main returns: five small memsets are far below the size at which the
// queue is flushed on its own. They have to reach the server anyway, which is
// what the flush at exit is for. tests/run_smoke.sh reads the fake driver's
// count of memsets afterwards.
static int queue_and_exit() {
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, 0));
  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, 4096));
  for (int i = 0; i < 5; i++) CHECK(cuMemsetD8Async(d, 0x5A, 4096, stream));
  std::printf("sync_smoke: exiting with five memsets queued\n");
  return g_failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && !std::strcmp(argv[1], "threads")) return two_threads();
  if (argc > 1 && !std::strcmp(argv[1], "exit")) return queue_and_exit();
  Mode mode;
  if (argc > 1 && !std::strcmp(argv[1], "wait")) mode = kWait;
  else if (argc > 1 && !std::strcmp(argv[1], "uploads")) mode = kUploads;
  else if (argc > 1 && !std::strcmp(argv[1], "lazy")) mode = kLazy;
  else {
    std::fprintf(stderr, "usage: sync_smoke wait|uploads|lazy|threads|exit\n");
    return 2;
  }

  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));
  CUstream stream = nullptr, other = nullptr;
  CHECK(cuStreamCreate(&stream, 0));
  CHECK(cuStreamCreate(&other, 0));
  const size_t n = 4096;
  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, n));
  std::vector<unsigned char> host(n), back(n, 0);

  // 1. Ten uploads, a synchronize after each. The buffer is rewritten between
  //    them, so each upload has to have taken its bytes when it was issued.
  for (int i = 1; i <= 10; i++) {
    std::memset(host.data(), i, n);
    CHECK(cuMemcpyHtoDAsync(d, host.data(), n, stream));
    CHECK(cuStreamSynchronize(stream));
  }
  std::memset(host.data(), 0xEE, n);
  CHECK(cuMemcpyDtoH(back.data(), d, n));
  size_t wrong = 0;
  for (unsigned char b : back) wrong += b != 10;
  EXPECT(wrong == 0, "memory should hold the last of ten uploads");

  // 2. Ten memsets, a synchronize after each: GPU work that is not an upload.
  for (int i = 1; i <= 10; i++) {
    CHECK(cuMemsetD8Async(d, static_cast<unsigned char>(100 + i), n, stream));
    CHECK(cuStreamSynchronize(stream));
  }
  CHECK(cuMemcpyDtoH(back.data(), d, n));
  wrong = 0;
  for (unsigned char b : back) wrong += b != 110;
  EXPECT(wrong == 0, "memory should hold the last of ten memsets");

  // 3. Work on one stream is no reason to wait for another.
  CHECK(cuMemsetD8Async(d, 7, n, stream));
  CHECK(cuStreamSynchronize(other));
  CHECK(cuStreamSynchronize(stream));

  // 4. A failing memset: a write to memory nobody allocated. Only with every
  //    synchronize queued does the one after it not wait.
  CHECK(cuMemsetD8Async(static_cast<CUdeviceptr>(0x1000), 1, n, stream));
  expect_failure(stream, mode != kLazy, "a failing memset");

  // 5. A failing upload. By default the synchronize after an upload is
  //    queued, so this is the one failure the default reports late.
  CHECK(cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(0x1000), host.data(), n, stream));
  expect_failure(stream, mode == kWait, "a failing upload");

  CHECK(cuMemFree(d));
  CHECK(cuStreamDestroy(other));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuCtxDestroy(ctx));
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  static const char* const kNames[] = {
      "every one waits (RGPU_LAZY_SYNC=0)",
      "queued after uploads only (the default)",
      "every one queued (RGPU_LAZY_SYNC=1)"};
  std::printf("PASS: stream synchronize, %s\n", kNames[mode]);
  return 0;
}
