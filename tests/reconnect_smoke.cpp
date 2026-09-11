// Checks that a dropped connection does not lose the GPU, without a GPU.
//
// The server is told to break the connection partway through, by frame count,
// so the break lands in the middle of a conversation rather than between two
// tidy operations. What matters afterwards is not that calls still succeed but
// that they still refer to the same things: the allocation made before the
// break must still hold what was written to it, because the session on the
// other side never went away.
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

  // Enough traffic that the server's drop point falls somewhere in here.
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

  CHECK(cuMemFree(d));
  CHECK(cuCtxDestroy(ctx));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: the session survived a dropped connection\n");
  return 0;
}
