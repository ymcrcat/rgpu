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

// The hidden entry points nvcc emits calls to, which are how a host function
// comes to stand for a kernel. Declared here because no header does.
extern "C" {
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaRegisterFatBinaryEnd(void** fatCubinHandle);
void __cudaUnregisterFatBinary(void** fatCubinHandle);
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
                            char* deviceFun, const char* deviceName,
                            int thread_limit, uint3* tid, uint3* bid,
                            dim3* bDim, dim3* gDim, int* wSize);
}

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

  // Every field the driver can answer, and not just the ones PyTorch reads.
  // llama.cpp picks how its quantized matrix multiplies run from
  // sharedMemPerBlockOptin: reported as zero it took a different path and
  // produced different numbers from the same GPU. The values are the fake's.
  EXPECT(prop.sharedMemPerBlockOptin == 101376, "sharedMemPerBlockOptin");
  EXPECT(prop.reservedSharedMemPerBlock == 1024, "reservedSharedMemPerBlock");
  EXPECT(prop.regsPerMultiprocessor == 65536, "regsPerMultiprocessor");
  EXPECT(prop.memPitch == 2147483647, "memPitch");
  EXPECT(prop.textureAlignment == 512, "textureAlignment");
  EXPECT(prop.asyncEngineCount == 2, "asyncEngineCount");
  EXPECT(prop.maxTexture2D[0] == 131072 && prop.maxTexture2D[1] == 65536,
         "maxTexture2D");
  EXPECT(prop.maxTexture3D[2] == 16384, "maxTexture3D depth");
  EXPECT(prop.persistingL2CacheMaxSize == 51904512, "persistingL2CacheMaxSize");
  // Deliberately not passed on: the GPU can use a host pointer for registered
  // memory, but that host is the server, and the client's memory is not there.
  EXPECT(prop.canUseHostPointerForRegisteredMem == 0,
         "a host-memory capability should not be advertised across the wire");
  // The properties of a device do not change, so asking again must cost
  // nothing. tests/run_smoke.sh counts the round trips this loop made.
  for (int i = 0; i < 8; i++) {
    cudaDeviceProp again{};
    CHECK(cudaGetDeviceProperties(&again, 0));
    EXPECT(std::memcmp(&again, &prop, sizeof(prop)) == 0,
           "the properties changed between two calls");
  }

  CHECK(cudaSetDevice(0));

  // The occupancy of a kernel at a launch shape, which llama.cpp asks for
  // before nearly every launch: 24 round trips per generated token. The answer
  // depends only on the kernel and the shape, so each distinct question should
  // reach the server once. A kernel is registered by hand, as nvcc would.
  {
    static unsigned char image[16 + 256];
    const unsigned int magic = 0xBA55ED50u;
    const unsigned short version = 1, header_size = 16;
    const unsigned long long body = 256;
    std::memset(image, 0xAB, sizeof(image));
    std::memcpy(image + 0, &magic, sizeof(magic));
    std::memcpy(image + 4, &version, sizeof(version));
    std::memcpy(image + 6, &header_size, sizeof(header_size));
    std::memcpy(image + 8, &body, sizeof(body));
    static const char kernel = 0;  // its address is the host function
    void** module = __cudaRegisterFatBinary(image);
    __cudaRegisterFunction(module, &kernel, nullptr, "rgpu_check_args", -1,
                           nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFatBinaryEnd(module);
    for (int i = 0; i < 8; i++) {
      int blocks = 0;
      CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, &kernel, 256, 0));
      EXPECT(blocks == 16, "occupancy at 256 threads should be the fake's 16");
      CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, &kernel, 128, 64));
      EXPECT(blocks == 31, "occupancy at 128 threads with shared memory should be 31");
    }
    __cudaUnregisterFatBinary(module);
  }

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
