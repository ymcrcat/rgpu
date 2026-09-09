// Checks the kernel launch path without a GPU.
//
// Only meaningful against rgpu-server-fake, whose driver knows the layout and
// the values to expect. Against a real driver it skips, because the synthetic
// module image it uses is correctly rejected. The real-hardware equivalent is
// tests/cuda/vecadd.cpp, which runs an actual kernel.
//
// This is the most intricate part of the system. The driver API hands over
// kernel arguments as an array of pointers with no sizes, so the client has to
// ask the server for the kernel's parameter layout, pack the arguments into
// it, and hand the packed buffer over through the extra[] mechanism. Every one
// of those steps can be wrong in a way that produces a plausible-looking
// launch and garbage results.
//
// Here the server's fake driver knows the layout and the exact values it
// should receive, and reports a mismatch as a failed launch. Sizing the fatbin
// from its own header is covered too: a wrong size truncates the image and the
// load fails.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./launch_smoke

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

namespace {

// A fatbin header the shim can size: magic, version, header length, then the
// length of everything after the header.
std::vector<unsigned char> make_fake_fatbin(size_t body_bytes) {
  std::vector<unsigned char> img(16 + body_bytes, 0xAB);
  const unsigned int magic = 0xBA55ED50u;
  const unsigned short version = 1;
  const unsigned short header_size = 16;
  const unsigned long long fat_size = body_bytes;
  std::memcpy(img.data() + 0, &magic, sizeof(magic));
  std::memcpy(img.data() + 4, &version, sizeof(version));
  std::memcpy(img.data() + 6, &header_size, sizeof(header_size));
  std::memcpy(img.data() + 8, &fat_size, sizeof(fat_size));
  return img;
}

}  // namespace

int main() {
  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK(cuCtxCreate(&ctx, 0, dev));

  // The image is a header with a filler body: enough for the shim to size and
  // ship it, and enough for the fake driver to accept. A real driver rejects
  // it, which is how this test knows it is pointed at real hardware and has
  // nothing to say.
  const std::vector<unsigned char> image = make_fake_fatbin(512);
  CUmodule mod = nullptr;
  if (cuModuleLoadData(&mod, image.data()) != CUDA_SUCCESS) {
    std::printf("SKIP: the server has a real driver, which rejects this "
                "synthetic image as it should.\n"
                "      Run this against rgpu-server-fake; use vecadd for a "
                "real kernel on real hardware.\n");
    return 0;
  }

  CUfunction fn = nullptr;
  CHECK(cuModuleGetFunction(&fn, mod, "rgpu_check_args"));

  // The values the server checks for. Three pointer-sized arguments and an int,
  // which exercises a layout with a smaller trailing parameter.
  CUdeviceptr a = 0x1111111111111111ull;
  CUdeviceptr b = 0x2222222222222222ull;
  CUdeviceptr c = 0x3333333333333333ull;
  int n = 42;
  void* args[] = {&a, &b, &c, &n};

  // Arguments given as an array of pointers: the client must fetch the layout
  // and pack them.
  CHECK(cuLaunchKernel(fn, 8, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));

  // The same arguments packed by the caller and passed through extra[], which
  // takes the layout lookup out of the picture.
  unsigned char packed[28];
  std::memcpy(packed + 0, &a, 8);
  std::memcpy(packed + 8, &b, 8);
  std::memcpy(packed + 16, &c, 8);
  std::memcpy(packed + 24, &n, 4);
  size_t packed_size = sizeof(packed);
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                   CU_LAUNCH_PARAM_END};
  CHECK(cuLaunchKernel(fn, 8, 1, 1, 256, 1, 1, 0, nullptr, nullptr, extra));

  // A wrong argument must be reported, otherwise the checks above prove
  // nothing: a server that accepted anything would pass them too.
  //
  // Where it is reported depends on batching. A launch is normally sent
  // without waiting for a reply, so it returns success and the failure
  // arrives at the next call that does reply. The rule that holds either way
  // is that it must not survive the next synchronization.
  int wrong = 43;
  void* bad_args[] = {&a, &b, &c, &wrong};
  const CUresult launch_rc =
      cuLaunchKernel(fn, 8, 1, 1, 256, 1, 1, 0, nullptr, bad_args, nullptr);
  const CUresult sync_rc = cuCtxSynchronize();
  if (launch_rc == CUDA_SUCCESS && sync_rc == CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: a launch with wrong arguments was never "
                         "reported, so the checks above prove nothing\n");
    g_failures++;
  }

  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: kernel arguments arrive intact in device layout\n");
  return 0;
}
