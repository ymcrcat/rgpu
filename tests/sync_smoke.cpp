// Checks what RGPU_LAZY_SYNC=1 changes about cuStreamSynchronize, and what it
// must not, without a GPU.
//
// By default a stream synchronize is a round trip: everything queued is sent
// and the client waits for the server to finish it. A program that
// synchronizes after every small upload - llama.cpp does, some twenty times
// per generated token - pays the link's latency each time. With
// RGPU_LAZY_SYNC=1 the synchronize is queued like the work it follows and
// returns at once. What that must leave alone is the result: the server still
// runs everything in order, and reading memory back is still a round trip that
// sees all of it. What it does change is when a failure is reported: not by
// the synchronize, which has no reply to carry it, but by the next call that
// waits. cuCtxSynchronize always waits, so a program always has a way to ask.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./sync_smoke wait
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 RGPU_LAZY_SYNC=1 \
//     ./sync_smoke lazy
//
// tests/run_smoke.sh runs both and counts the round trips each made.

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

int main(int argc, char** argv) {
  if (argc < 2 || (std::strcmp(argv[1], "lazy") && std::strcmp(argv[1], "wait"))) {
    std::fprintf(stderr, "usage: sync_smoke lazy|wait\n");
    return 2;
  }
  const bool lazy = std::strcmp(argv[1], "lazy") == 0;

  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, 0));
  const size_t n = 4096;
  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, n));

  // Work, then a synchronize, ten times over. Whichever way the synchronize
  // travels, the memory afterwards holds the last thing written.
  for (int i = 1; i <= 10; i++) {
    CHECK(cuMemsetD8Async(d, static_cast<unsigned char>(i), n, stream));
    CHECK(cuStreamSynchronize(stream));
  }
  std::vector<unsigned char> back(n, 0);
  CHECK(cuMemcpyDtoH(back.data(), d, n));
  size_t wrong = 0;
  for (unsigned char b : back) wrong += b != 10;
  EXPECT(wrong == 0, "memory should hold the last of ten queued writes");

  // A failure in queued work: a write to memory nobody allocated. The
  // synchronize after it reports the failure if it waits, and cannot if it
  // was queued - then the next call that waits does, once.
  CHECK(cuMemsetD8Async(static_cast<CUdeviceptr>(0x1000), 1, n, stream));
  const CUresult synced = cuStreamSynchronize(stream);
  if (lazy) {
    EXPECT(synced == CUDA_SUCCESS,
           "a queued synchronize has no reply and should return success");
    EXPECT(cuCtxSynchronize() != CUDA_SUCCESS,
           "the failure should surface at the next call that waits");
  } else {
    EXPECT(synced != CUDA_SUCCESS,
           "a waiting synchronize should report the failure before it");
  }
  CHECK(cuCtxSynchronize());

  CHECK(cuMemFree(d));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuCtxDestroy(ctx));
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: stream synchronize, %s\n",
              lazy ? "queued (RGPU_LAZY_SYNC=1)" : "waiting (the default)");
  return 0;
}
