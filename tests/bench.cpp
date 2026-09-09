// Measures what the remoting layer costs per call.
//
// Run against rgpu-server-fake and the numbers are almost pure protocol: the
// fake driver returns immediately, so what is left is serialization, the write,
// the wait and the read. That is the cost batching is meant to remove, and it
// can be measured without renting a GPU.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./bench
//
// Against a real server the same numbers include the GPU and the network.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda.h>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void report(const char* what, int calls, double ms) {
  std::printf("  %-34s %7d calls  %8.1f ms  %7.1f us/call  %9.0f calls/s\n",
              what, calls, ms, ms * 1000.0 / calls, calls / (ms / 1000.0));
}

// The same synthetic image and kernel the launch test uses, so the fake driver
// recognises it and the argument check passes.
std::vector<unsigned char> fake_fatbin(size_t body) {
  std::vector<unsigned char> img(16 + body, 0xAB);
  const unsigned int magic = 0xBA55ED50u;
  const unsigned short version = 1, header_size = 16;
  const unsigned long long fat_size = body;
  std::memcpy(img.data() + 0, &magic, sizeof(magic));
  std::memcpy(img.data() + 4, &version, sizeof(version));
  std::memcpy(img.data() + 6, &header_size, sizeof(header_size));
  std::memcpy(img.data() + 8, &fat_size, sizeof(fat_size));
  return img;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
  // A real driver rejects the synthetic image, so on real hardware pass a
  // fatbin built by nvcc and the launch numbers become measurable too. That is
  // the call a workload makes most, so it is the one worth measuring there.
  const char* fatbin_path = argc > 2 ? argv[2] : nullptr;

  if (cuInit(0) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuInit failed; is the server running?\n");
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    std::fprintf(stderr, "could not get a context\n");
    return 1;
  }

  std::printf("\nper-call cost over the wire, %d calls each\n\n", n);

  // A call that carries almost nothing, to isolate the round trip itself.
  {
    auto t0 = Clock::now();
    for (int i = 0; i < n; i++) {
      int count = 0;
      cuDeviceGetCount(&count);
    }
    report("cuDeviceGetCount (empty round trip)", n, ms_since(t0));
  }

  // Allocation, which the client also has to track.
  {
    std::vector<CUdeviceptr> ptrs(n);
    auto t0 = Clock::now();
    for (int i = 0; i < n; i++) cuMemAlloc(&ptrs[i], 1024);
    report("cuMemAlloc", n, ms_since(t0));
    t0 = Clock::now();
    for (int i = 0; i < n; i++) cuMemFree(ptrs[i]);
    report("cuMemFree", n, ms_since(t0));
  }

  // Host to device with a payload, the case where bandwidth starts to matter.
  {
    CUdeviceptr d = 0;
    const size_t bytes = 64 * 1024;
    std::vector<unsigned char> host(bytes, 7);
    if (cuMemAlloc(&d, bytes) == CUDA_SUCCESS) {
      auto t0 = Clock::now();
      for (int i = 0; i < n; i++) cuMemcpyHtoD(d, host.data(), bytes);
      const double ms = ms_since(t0);
      report("cuMemcpyHtoD 64 KiB", n, ms);
      std::printf("  %-34s %7s %27.1f MiB/s\n", "", "",
                  (double(n) * bytes / (1 << 20)) / (ms / 1000.0));
      cuMemFree(d);
    }
  }

  // Kernel launches, the call that dominates any real workload.
  {
    std::vector<unsigned char> image;
    const char* kernel = "rgpu_check_args";
    if (fatbin_path) {
      std::FILE* f = std::fopen(fatbin_path, "rb");
      if (!f) {
        std::fprintf(stderr, "cannot open %s\n", fatbin_path);
        return 1;
      }
      std::fseek(f, 0, SEEK_END);
      const long len = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      image.resize(static_cast<size_t>(len));
      if (std::fread(image.data(), 1, image.size(), f) != image.size()) {
        std::fprintf(stderr, "short read on %s\n", fatbin_path);
        std::fclose(f);
        return 1;
      }
      std::fclose(f);
      kernel = "vecadd";
    } else {
      image = fake_fatbin(512);
    }

    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    // Real device buffers when running a real kernel; the fake driver only
    // checks that the recognisable values arrive.
    CUdeviceptr a = 0x1111111111111111ull, b = 0x2222222222222222ull,
                c = 0x3333333333333333ull;
    int k = 42;
    if (fatbin_path) {
      k = 1024;
      const size_t bytes = size_t(k) * sizeof(float);
      if (cuMemAlloc(&a, bytes) != CUDA_SUCCESS ||
          cuMemAlloc(&b, bytes) != CUDA_SUCCESS ||
          cuMemAlloc(&c, bytes) != CUDA_SUCCESS) {
        std::fprintf(stderr, "could not allocate kernel buffers\n");
        return 1;
      }
    }
    if (cuModuleLoadData(&mod, image.data()) == CUDA_SUCCESS &&
        cuModuleGetFunction(&fn, mod, kernel) == CUDA_SUCCESS) {
      void* args[] = {&a, &b, &c, &k};
      // Warm the parameter layout cache so the first call does not skew it.
      cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);

      // Two numbers, because batching separates them. The first is how fast
      // the client can issue launches, which is what a program sees. The
      // second includes the synchronization that waits for them to actually
      // run, which is the honest end-to-end cost: without it the first number
      // would just be measuring how fast we can defer work.
      auto t0 = Clock::now();
      for (int i = 0; i < n; i++) {
        cuLaunchKernel(fn, 8, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
      }
      const double issue_ms = ms_since(t0);
      cuCtxSynchronize();
      const double total_ms = ms_since(t0);
      report("cuLaunchKernel (issue)", n, issue_ms);
      report("cuLaunchKernel (issue + synchronize)", n, total_ms);
      cuModuleUnload(mod);
    } else {
      std::printf("  (kernel launch skipped: this server has a real driver, "
                  "which rejects the synthetic image)\n");
    }
  }

  std::printf("\n");
  cuCtxDestroy(ctx);
  return 0;
}
