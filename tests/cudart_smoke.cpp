// The phase-2 test: a program written against the CUDA runtime API rather
// than the driver API. PyTorch sits on exactly this surface.
//
// Run it against our libcudart to test the whole stack:
//   LD_LIBRARY_PATH=build ./cudart_smoke
//
// Run it against the stock libcudart to demonstrate why we need our own. That
// one calls cuGetExportTable right after cuInit and refuses to initialise
// without a table of undocumented internal driver pointers, which cannot cross
// a process boundary:
//   LD_LIBRARY_PATH=third_party/cudart:build ./cudart_smoke

#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime_api.h>

static int g_failures = 0;

#define CHECK(call)                                                        \
  do {                                                                     \
    cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__,   \
                   #call, cudaGetErrorString(e_));                         \
      g_failures++;                                                        \
    }                                                                      \
  } while (0)

#define EXPECT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);   \
      g_failures++;                                                        \
    }                                                                      \
  } while (0)

int main() {
  int runtime_version = 0, driver_version = 0;
  CHECK(cudaRuntimeGetVersion(&runtime_version));
  CHECK(cudaDriverGetVersion(&driver_version));
  std::printf("runtime %d, driver %d\n", runtime_version, driver_version);

  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  EXPECT(count > 0, "expected at least one device");
  if (count <= 0 || g_failures) {
    std::printf("\nFAILED early: the runtime could not reach a device\n");
    return 1;
  }

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device 0: %s, sm_%d%d, %.1f GiB\n", prop.name, prop.major,
              prop.minor, double(prop.totalGlobalMem) / (1 << 30));

  CHECK(cudaSetDevice(0));

  // A data round trip through the runtime's own memory calls.
  const size_t n = 256 * 1024;
  std::vector<unsigned char> src(n), dst(n, 0);
  for (size_t i = 0; i < n; i++) src[i] = static_cast<unsigned char>(i * 17 + 3);

  void* d = nullptr;
  CHECK(cudaMalloc(&d, n));
  EXPECT(d != nullptr, "cudaMalloc returned null");
  if (d) {
    CHECK(cudaMemcpy(d, src.data(), n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dst.data(), d, n, cudaMemcpyDeviceToHost));
    EXPECT(std::memcmp(src.data(), dst.data(), n) == 0,
           "round trip through cudaMemcpy corrupted data");
    CHECK(cudaFree(d));
  }

  size_t free_bytes = 0, total_bytes = 0;
  CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("memory: %.1f GiB free of %.1f GiB\n",
              double(free_bytes) / (1 << 30), double(total_bytes) / (1 << 30));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: the CUDA runtime API works over the shim\n");
  return 0;
}
