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
// 24 waiting, 12 by default, none with everything queued.

#include <cstdio>
#include <cstring>
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

int main(int argc, char** argv) {
  Mode mode;
  if (argc > 1 && !std::strcmp(argv[1], "wait")) mode = kWait;
  else if (argc > 1 && !std::strcmp(argv[1], "uploads")) mode = kUploads;
  else if (argc > 1 && !std::strcmp(argv[1], "lazy")) mode = kLazy;
  else {
    std::fprintf(stderr, "usage: sync_smoke wait|uploads|lazy\n");
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
