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

int main() {
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
  CUdeviceptr scratch = 0;
  CHECK(cuMemAlloc(&scratch, n));
  unsigned char packed[28] = {0};
  size_t packed_size = sizeof(packed);
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                   CU_LAUNCH_PARAM_END};
  cuLaunchKernel(nullptr, 8, 1, 1, 256, 1, 1, 0, nullptr, nullptr, extra);

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
    std::fprintf(stderr,
                 "FAIL: a launch failed with %d before the break; the first "
                 "call that could report it returned %d\n",
                 CUDA_ERROR_INVALID_HANDLE, deferred);
    g_failures++;
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
