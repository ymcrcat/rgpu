// Driver-API vector add. Uses no CUDA runtime, so it exercises exactly the
// surface the shim implements.
//
// Runs against the real driver on the GPU host, or against the shim from a
// machine with no GPU:
//   LD_LIBRARY_PATH=build RGPU_SERVER=host:9713 ./vecadd vecadd.fatbin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>

#define CHECK(call)                                                     \
  do {                                                                  \
    CUresult r_ = (call);                                               \
    if (r_ != CUDA_SUCCESS) {                                           \
      const char* s_ = nullptr;                                         \
      cuGetErrorName(r_, &s_);                                          \
      std::fprintf(stderr, "%s:%d: %s failed: %s (%d)\n", __FILE__,     \
                   __LINE__, #call, s_ ? s_ : "?", r_);                 \
      return 1;                                                         \
    }                                                                   \
  } while (0)

int main(int argc, char** argv) {
  const char* image_path = argc > 1 ? argv[1] : "vecadd.fatbin";
  const int n = argc > 2 ? std::atoi(argv[2]) : 1 << 20;

  std::FILE* f = std::fopen(image_path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", image_path);
    return 1;
  }
  std::fseek(f, 0, SEEK_END);
  long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> image(static_cast<size_t>(len) + 1, 0);
  if (std::fread(image.data(), 1, static_cast<size_t>(len), f) !=
      static_cast<size_t>(len)) {
    std::fprintf(stderr, "short read on %s\n", image_path);
    std::fclose(f);
    return 1;
  }
  std::fclose(f);

  CHECK(cuInit(0));

  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 1) {
    std::fprintf(stderr, "no CUDA devices\n");
    return 1;
  }
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  char name[256] = {0};
  CHECK(cuDeviceGetName(name, sizeof(name), dev));
  int major = 0, minor = 0;
  CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  std::printf("device 0: %s (sm_%d%d)\n", name, major, minor);

  CUcontext ctx;
  CHECK(cuCtxCreate(&ctx, 0, dev));

  CUmodule mod;
  CHECK(cuModuleLoadData(&mod, image.data()));
  CUfunction fn;
  CHECK(cuModuleGetFunction(&fn, mod, "vecadd"));

  std::vector<float> ha(n), hb(n), hc(n, 0.0f);
  for (int i = 0; i < n; i++) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i);
  }

  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUdeviceptr da, db, dc;
  CHECK(cuMemAlloc(&da, bytes));
  CHECK(cuMemAlloc(&db, bytes));
  CHECK(cuMemAlloc(&dc, bytes));
  CHECK(cuMemcpyHtoD(da, ha.data(), bytes));
  CHECK(cuMemcpyHtoD(db, hb.data(), bytes));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  void* args[] = {&da, &db, &dc, const_cast<int*>(&n)};
  CHECK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args,
                       nullptr));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(hc.data(), dc, bytes));

  int bad = 0;
  for (int i = 0; i < n; i++) {
    const float want = static_cast<float>(i) + static_cast<float>(2 * i);
    if (hc[i] != want) {
      if (bad < 5) {
        std::fprintf(stderr, "mismatch at %d: got %f want %f\n", i, hc[i], want);
      }
      bad++;
    }
  }

  CHECK(cuMemFree(da));
  CHECK(cuMemFree(db));
  CHECK(cuMemFree(dc));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));

  if (bad) {
    std::printf("FAIL: %d of %d elements wrong\n", bad, n);
    return 1;
  }
  std::printf("PASS: %d elements correct\n", n);
  return 0;
}
