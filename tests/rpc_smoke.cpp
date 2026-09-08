// End-to-end check of the remoting path, with or without a GPU.
//
// Exercises every parameter marshalling mode the generator produces except
// kernel launch: scalars, output scalars, output handles, input buffers,
// output buffers and strings. A byte-for-byte memory round trip is the real
// assertion: if serialization is wrong anywhere, the data comes back wrong.
//
//   RGPU_SERVER=127.0.0.1:9713 LD_LIBRARY_PATH=build ./rpc_smoke

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

#define EXPECT(cond, msg)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__,         \
                   __LINE__, msg);                               \
      g_failures++;                                              \
    }                                                            \
  } while (0)

int main() {
  CHECK(cuInit(0));

  int version = 0;
  CHECK(cuDriverGetVersion(&version));
  EXPECT(version > 0, "driver version should be positive");
  std::printf("driver version: %d\n", version);

  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  EXPECT(count > 0, "expected at least one device");
  if (count <= 0) return 1;

  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));

  // out_buffer: the server fills a caller-sized buffer.
  char name[256] = {0};
  CHECK(cuDeviceGetName(name, sizeof(name), dev));
  EXPECT(name[0] != '\0', "device name should not be empty");
  std::printf("device 0: %s\n", name);

  // out_scalar.
  int major = 0, minor = 0;
  CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  EXPECT(major > 0, "compute capability major should be positive");
  std::printf("compute capability: %d.%d\n", major, minor);

  size_t total = 0;
  CHECK(cuDeviceTotalMem(&total, dev));
  EXPECT(total > 0, "total memory should be positive");
  std::printf("total memory: %.1f GiB\n", double(total) / (1 << 30));

  // out_handle.
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));
  EXPECT(ctx != nullptr, "context handle should not be null");

  // The real test: a data round trip through both marshalling directions.
  const size_t n = 64 * 1024;
  std::vector<unsigned char> src(n), dst(n, 0);
  for (size_t i = 0; i < n; i++) src[i] = static_cast<unsigned char>(i * 31 + 7);

  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, n));
  EXPECT(d != 0, "device pointer should not be null");
  CHECK(cuMemcpyHtoD(d, src.data(), n));       // in_buffer
  CHECK(cuMemcpyDtoH(dst.data(), d, n));       // out_buffer

  size_t bad = 0;
  for (size_t i = 0; i < n; i++) {
    if (dst[i] != src[i]) bad++;
  }
  EXPECT(bad == 0, "memory round trip corrupted data");
  if (bad) std::fprintf(stderr, "  %zu of %zu bytes differ\n", bad, n);

  // A partial copy at an offset, to catch off-by-one handling of sizes.
  const size_t half = n / 2;
  std::vector<unsigned char> partial(half, 0);
  CHECK(cuMemcpyDtoH(partial.data(), d + half, half));
  EXPECT(std::memcmp(partial.data(), src.data() + half, half) == 0,
         "offset copy returned the wrong bytes");

  CHECK(cuMemFree(d));
  CHECK(cuCtxDestroy(ctx));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: remoting round trip correct\n");
  return 0;
}
