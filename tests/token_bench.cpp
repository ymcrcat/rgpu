// Replays the CUDA calls llama.cpp makes to generate one token, through the
// runtime shim, and says where the client's time goes.
//
// Measured on a real run (qwen2.5:0.5b, 320 tokens), one token is about:
//   20  small uploads, each followed by a stream synchronize
//   450 kernel launches
//   1   download of the model's output, 0.58 MiB
// which is some 490 calls that want no reply and one round trip. Against
// rgpu-server-fake nothing here computes, so what is timed is the remoting
// layer alone: packing, queueing, the writes, and the one wait per token.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./token_bench \\
//     [tokens [launches [download-bytes]]]
//
// Each stage is timed on its own and as CPU time of this process, so that time
// spent computing here can be told from time spent waiting for the server.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_runtime_api.h>

extern "C" {
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaRegisterFatBinaryEnd(void** fatCubinHandle);
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
                            char* deviceFun, const char* deviceName,
                            int thread_limit, uint3* tid, uint3* bid,
                            dim3* bDim, dim3* gDim, int* wSize);
}

namespace {

using Clock = std::chrono::steady_clock;

double wall_ms() {
  return std::chrono::duration<double, std::milli>(
             Clock::now().time_since_epoch()).count();
}
double cpu_ms() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

struct Stage {
  const char* name;
  int calls_per_token;
  double wall = 0, cpu = 0;
  int failed = 0;
};

struct Timer {
  Stage& s;
  double w0 = wall_ms(), c0 = cpu_ms();
  explicit Timer(Stage& stage) : s(stage) {}
  ~Timer() {
    s.wall += wall_ms() - w0;
    s.cpu += cpu_ms() - c0;
  }
};

}  // namespace

int main(int argc, char** argv) {
  const int tokens = argc > 1 ? std::atoi(argv[1]) : 200;
  // The shape of a token can be changed to see what each part costs: fewer
  // launches, or a smaller download.
  const int kUploads = 20;
  const int kLaunches = argc > 2 ? std::atoi(argv[2]) : 450;
  const size_t kUploadBytes = 5 * 1024;
  const size_t kDownloadBytes =
      argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 607744;  // 151936 floats

  static unsigned char image[16 + 512];
  const unsigned int magic = 0xBA55ED50u;
  const unsigned short version = 1, header_size = 16;
  const unsigned long long body = 512;
  std::memset(image, 0xAB, sizeof(image));
  std::memcpy(image + 0, &magic, sizeof(magic));
  std::memcpy(image + 4, &version, sizeof(version));
  std::memcpy(image + 6, &header_size, sizeof(header_size));
  std::memcpy(image + 8, &body, sizeof(body));
  static const char kernel = 0;
  void** module = __cudaRegisterFatBinary(image);
  __cudaRegisterFunction(module, &kernel, nullptr, "rgpu_check_args", -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(module);

  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "no device; is the server running?\n");
    return 1;
  }
  cudaStream_t stream = nullptr;
  void *up = nullptr, *down = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess ||
      cudaMalloc(&up, kUploadBytes) != cudaSuccess ||
      cudaMalloc(&down, kDownloadBytes) != cudaSuccess) {
    std::fprintf(stderr, "setup failed\n");
    return 1;
  }
  std::vector<unsigned char> host_up(kUploadBytes, 7), host_down(kDownloadBytes);

  // The arguments the fake driver's kernel checks for.
  unsigned long long a = 0x1111111111111111ull, b = 0x2222222222222222ull,
                     c = 0x3333333333333333ull;
  int k = 42;
  void* args[] = {&a, &b, &c, &k};
  const dim3 grid(8, 1, 1), block(256, 1, 1);
  // Once, so the first token does not pay for loading the module.
  cudaLaunchKernel(&kernel, grid, block, args, 0, stream);
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();

  Stage uploads{"uploads, each with a synchronize", 2 * kUploads};
  Stage launches{"kernel launches", kLaunches};
  Stage download{"download of the output", 2};
  int failures = 0;
  const double w0 = wall_ms(), c0 = cpu_ms();
  for (int t = 0; t < tokens; t++) {
    {
      Timer timer(uploads);
      for (int i = 0; i < kUploads; i++) {
        uploads.failed += cudaMemcpyAsync(up, host_up.data(), kUploadBytes,
                                    cudaMemcpyHostToDevice, stream) != cudaSuccess;
        uploads.failed += cudaStreamSynchronize(stream) != cudaSuccess;
      }
    }
    {
      Timer timer(launches);
      for (int i = 0; i < kLaunches; i++) {
        launches.failed += cudaLaunchKernel(&kernel, grid, block, args, 0, stream) !=
                    cudaSuccess;
      }
    }
    {
      Timer timer(download);
      download.failed += cudaMemcpyAsync(host_down.data(), down, kDownloadBytes,
                                  cudaMemcpyDeviceToHost, stream) != cudaSuccess;
      download.failed += cudaStreamSynchronize(stream) != cudaSuccess;
    }
  }
  failures = uploads.failed + launches.failed + download.failed;
  const double wall = wall_ms() - w0, cpu = cpu_ms() - c0;

  std::printf("\none token's CUDA calls, replayed %d times: %d launches, a "
              "%zu-byte download%s\n\n", tokens, kLaunches, kDownloadBytes,
              failures ? "  (SOME CALLS FAILED)" : "");
  std::printf("  %-34s %7s %11s %11s %12s\n", "stage", "calls", "wall/token",
              "cpu/token", "wall/call");
  for (const Stage* s : {&uploads, &launches, &download}) {
    std::printf("  %-34s %7d %8.1f us %8.1f us %9.2f us%s\n", s->name,
                s->calls_per_token, s->wall * 1e3 / tokens, s->cpu * 1e3 / tokens,
                s->wall * 1e3 / tokens / s->calls_per_token,
                s->failed ? "  FAILED" : "");
  }
  std::printf("  %-34s %7d %8.1f us %8.1f us\n", "whole token",
              2 * kUploads + kLaunches + 2, wall * 1e3 / tokens,
              cpu * 1e3 / tokens);
  std::printf("\n  %.0f tokens per second if the GPU took no time at all\n\n",
              tokens / (wall / 1e3));
  return failures ? 1 : 0;
}
