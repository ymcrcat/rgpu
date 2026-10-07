// Checks CUDA's virtual memory management calls across the wire, without a
// GPU, and that a client which dies holding mapped memory takes it with it.
//
// These are the calls llama.cpp builds its device memory pool from: reserve an
// address range, create memory, map one onto the other, grant access. Four of
// them take a pointer to a small struct, which is the part worth testing: the
// fake driver refuses a property it was not sent, so a struct that crossed
// wrong fails here rather than allocating something else.
//
// The memory belongs to no context, so nothing a context teardown does gives
// it back. `vmm_smoke die` maps memory the way llama.cpp does - the handle
// released straight after mapping, so only the mapping holds the memory - and
// exits without unmapping; tests/run_smoke.sh then checks that the session's
// expiry gave it all back.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./vmm_smoke [die]

#include <cstdio>
#include <cstring>
#include <vector>

#include <unistd.h>

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

static CUmemAllocationProp device_memory(int device) {
  CUmemAllocationProp prop;
  std::memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}

// Reserves, creates, maps and grants access, releasing the handle as soon as
// it is mapped. Returns the mapped address, or 0.
static CUdeviceptr map_memory(size_t size) {
  const CUmemAllocationProp prop = device_memory(0);
  CUdeviceptr ptr = 0;
  CUmemGenericAllocationHandle handle = 0;
  CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  CHECK(cuMemCreate(&handle, size, &prop, 0));
  CHECK(cuMemMap(ptr, size, 0, handle, 0));
  CHECK(cuMemRelease(handle));
  CUmemAccessDesc access;
  std::memset(&access, 0, sizeof(access));
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK(cuMemSetAccess(ptr, size, &access, 1));
  return ptr;
}

int main(int argc, char** argv) {
  const bool die = argc > 1 && std::strcmp(argv[1], "die") == 0;
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, dev));

  const CUmemAllocationProp prop = device_memory(0);
  size_t granularity = 0;
  CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  EXPECT(granularity == (size_t(2) << 20), "granularity should be the fake's 2 MiB");
  if (g_failures) return 1;

  if (die) {
    map_memory(granularity);
    map_memory(2 * granularity);
    std::printf("vmm_smoke: exiting with two ranges still mapped\n");
    std::fflush(stdout);
    _exit(g_failures ? 1 : 0);
  }

  // A property the fake refuses has to be refused through the wire too: if the
  // struct did not travel, the server would see whatever was in its buffer.
  CUmemAllocationProp host_memory = prop;
  host_memory.location.type = CU_MEM_LOCATION_TYPE_HOST;
  size_t ignored = 0;
  EXPECT(cuMemGetAllocationGranularity(&ignored, &host_memory,
                                       CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) ==
             CUDA_ERROR_INVALID_VALUE,
         "a host-memory property should be refused by the fake");

  const size_t size = granularity;
  CUdeviceptr ptr = 0;
  CUmemGenericAllocationHandle handle = 0;
  CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  EXPECT(ptr != 0, "reserved address should not be null");
  CHECK(cuMemCreate(&handle, size, &prop, 0));
  EXPECT(handle != 0, "allocation handle should not be null");

  CUmemAllocationProp read_back;
  std::memset(&read_back, 0xAB, sizeof(read_back));
  CHECK(cuMemGetAllocationPropertiesFromHandle(&read_back, handle));
  EXPECT(std::memcmp(&read_back, &prop, sizeof(prop)) == 0,
         "properties read back from the handle differ from those it was made with");

  CHECK(cuMemMap(ptr, size, 0, handle, 0));

  // Mapped but not yet accessible: the fake refuses a copy, as a GPU faults.
  std::vector<unsigned char> src(4096), dst(4096, 0);
  for (size_t i = 0; i < src.size(); i++) src[i] = static_cast<unsigned char>(i * 13 + 5);
  EXPECT(cuMemcpyHtoD(ptr, src.data(), src.size()) != CUDA_SUCCESS,
         "a copy into memory with no access granted should fail");

  CUmemAccessDesc access;
  std::memset(&access, 0, sizeof(access));
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK(cuMemSetAccess(ptr, size, &access, 1));
  unsigned long long flags = 0;
  CHECK(cuMemGetAccess(&flags, &prop.location, ptr));
  EXPECT(flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
         "access read back should be read-write");

  // The mapped memory behaves as device memory, at an offset too.
  CHECK(cuMemcpyHtoD(ptr + 8192, src.data(), src.size()));
  CHECK(cuMemcpyDtoH(dst.data(), ptr + 8192, dst.size()));
  EXPECT(std::memcmp(src.data(), dst.data(), src.size()) == 0,
         "data written through the mapping did not read back");

  CHECK(cuMemUnmap(ptr, size));
  CHECK(cuMemRelease(handle));
  CHECK(cuMemAddressFree(ptr, size));
  CHECK(cuCtxDestroy(ctx));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: virtual memory calls correct across the wire\n");
  return 0;
}
