// Checks that cuDNN backend attributes survive the wire, without a GPU.
//
// The attribute array is the part worth testing. Its element width is not in
// the call: it follows from the type enum, so a wrong width sends the wrong
// number of bytes. That corrupts a graph rather than failing it, which is the
// worst way for this to go wrong. The server's fake library knows every value
// and rejects anything else.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./cudnn_smoke

#include <cstdio>
#include <cstring>

#include <cudnn.h>

static int g_failures = 0;

#define CHECK(call)                                                        \
  do {                                                                     \
    cudnnStatus_t s_ = (call);                                             \
    if (s_ != CUDNN_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,   \
                   #call, s_);                                             \
      g_failures++;                                                        \
    }                                                                      \
  } while (0)

namespace {
// Must match tests/fake_cudnn.cpp.
const int64_t kDims[4] = {1, 3, 224, 224};
void* const kDevicePtr = reinterpret_cast<void*>(0xDEC1CEull);
constexpr int64_t kUid = 0x5150;
const int64_t kGetBack[3] = {11, 22, 33};
}  // namespace

int main() {
  cudnnHandle_t h = nullptr;
  CHECK(cudnnCreate(&h));
  CHECK(cudnnSetStream(h, nullptr));
  if (cudnnGetVersion() == 0) {
    std::fprintf(stderr, "FAIL: cudnnGetVersion returned 0\n");
    g_failures++;
  }

  cudnnBackendDescriptor_t tensor = nullptr;
  CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));

  // Eight-byte elements.
  CHECK(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS,
                                 CUDNN_TYPE_INT64, 4, kDims));

  // Four-byte elements, to cover the other width.
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  CHECK(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_DATA_TYPE,
                                 CUDNN_TYPE_DATA_TYPE, 1, &dtype));

  // A wrong dimension must be rejected, or the checks above prove nothing: a
  // server that accepted anything would pass them too.
  const int64_t wrong[4] = {1, 3, 224, 225};
  if (cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS,
                               CUDNN_TYPE_INT64, 4,
                               wrong) == CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: wrong dimensions were accepted\n");
    g_failures++;
  }

  // Reading back: the count and the values both have to survive.
  int64_t count = 0;
  int64_t got[8] = {0};
  CHECK(cudnnBackendGetAttribute(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS,
                                 CUDNN_TYPE_INT64, 8, &count, got));
  if (count != 3) {
    std::fprintf(stderr, "FAIL: expected 3 elements back, got %lld\n",
                 static_cast<long long>(count));
    g_failures++;
  } else if (std::memcmp(got, kGetBack, sizeof(kGetBack)) != 0) {
    std::fprintf(stderr, "FAIL: values read back do not match\n");
    g_failures++;
  }

  CHECK(cudnnBackendFinalize(tensor));

  // A variant pack carries device pointers, which are values in the server's
  // address space and must arrive unchanged.
  cudnnBackendDescriptor_t pack = nullptr;
  CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                     &pack));
  void* pointers[1] = {kDevicePtr};
  CHECK(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                 CUDNN_TYPE_VOID_PTR, 1, pointers));
  const int64_t uids[1] = {kUid};
  CHECK(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                 CUDNN_TYPE_INT64, 1, uids));

  cudnnBackendDescriptor_t plan = nullptr;
  CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                     &plan));
  CHECK(cudnnBackendExecute(h, plan, pack));

  CHECK(cudnnBackendDestroyDescriptor(plan));
  CHECK(cudnnBackendDestroyDescriptor(pack));
  CHECK(cudnnBackendDestroyDescriptor(tensor));
  CHECK(cudnnDestroy(h));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: cuDNN backend attributes survive the wire\n");
  return 0;
}
