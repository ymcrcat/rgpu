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

#include "client/rpc.h"
#include "common/cudnn_ids.h"

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

  // A string coming back the other way, into a buffer the caller owns.
  char err[64] = {0};
  cudnnGetLastErrorString(err, sizeof(err));
  if (std::strcmp(err, "fake cuDNN has nothing to report") != 0) {
    std::fprintf(stderr, "FAIL: last error string came back as \"%s\"\n", err);
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
  // server that accepted anything would pass them too. Setting an attribute
  // does not wait for a reply, so the rejection arrives at the next call that
  // does. Finalize is that call, which is also why finalize still waits.
  const int64_t wrong[4] = {1, 3, 224, 225};
  cudnnStatus_t set = cudnnBackendSetAttribute(
      tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, wrong);
  cudnnStatus_t later = cudnnBackendFinalize(tensor);
  if (set == CUDNN_STATUS_SUCCESS && later == CUDNN_STATUS_SUCCESS) {
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

  // Descriptor handles are minted on the client, so an attribute whose
  // elements are descriptors has to be translated on the way out and left
  // alone on the way back. The server's fake rejects anything but its own
  // descriptor, and this end checks it gets its own handle back.
  cudnnBackendDescriptor_t graph = nullptr, op = nullptr;
  CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                     &graph));
  CHECK(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));
  CHECK(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  cudnnBackendDescriptor_t ops_back[1] = {op};
  int64_t ops_count = 0;
  CHECK(cudnnBackendGetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ops_count,
                                 ops_back));
  if (ops_count != 1 || ops_back[0] != op) {
    std::fprintf(stderr, "FAIL: descriptor handle did not survive the wire\n");
    g_failures++;
  }
  CHECK(cudnnBackendDestroyDescriptor(op));
  CHECK(cudnnBackendDestroyDescriptor(graph));

  // The legacy path, which is how batch normalisation still reaches cuDNN.
  cudnnTensorDescriptor_t td = nullptr;
  CHECK(cudnnCreateTensorDescriptor(&td));
  int dims[4], strides[4];
  for (int i = 0; i < 4; i++) {
    dims[i] = static_cast<int>(kDims[i]);
    strides[i] = i + 1;
  }
  CHECK(cudnnSetTensorNdDescriptor(td, CUDNN_DATA_FLOAT, 4, dims, strides));

  cudnnDataType_t back = CUDNN_DATA_DOUBLE;
  int nb = 0, gotDims[8] = {0}, gotStrides[8] = {0};
  CHECK(cudnnGetTensorNdDescriptor(td, 8, &back, &nb, gotDims, gotStrides));
  if (back != CUDNN_DATA_FLOAT || nb != 4 || gotDims[3] != dims[3] ||
      gotStrides[3] != strides[3]) {
    std::fprintf(stderr, "FAIL: descriptor read back wrong\n");
    g_failures++;
  }

  cudnnTensorDescriptor_t bnd = nullptr;
  CHECK(cudnnCreateTensorDescriptor(&bnd));
  CHECK(cudnnDeriveBNTensorDescriptor(bnd, td, CUDNN_BATCHNORM_SPATIAL));

  // alpha and beta are floats here, because the tensor is float. Sending them
  // as doubles would arrive as zero, which is the bug this catches.
  const float one = 1.0f, zero = 0.0f;
  CHECK(cudnnBatchNormalizationForwardInference(
      h, CUDNN_BATCHNORM_SPATIAL, &one, &zero, td, kDevicePtr, td, kDevicePtr,
      bnd, kDevicePtr, kDevicePtr, kDevicePtr, kDevicePtr, 1e-5));

  CHECK(cudnnDestroyTensorDescriptor(bnd));
  CHECK(cudnnDestroyTensorDescriptor(td));

  CHECK(cudnnBackendDestroyDescriptor(plan));
  CHECK(cudnnBackendDestroyDescriptor(pack));
  CHECK(cudnnBackendDestroyDescriptor(tensor));
  CHECK(cudnnDestroy(h));

  // Requests whose count disagrees with the bytes sent, built by hand
  // because the library never sends one. cuDNN reads or writes count times
  // the element size, so a count larger than the bytes on the wire made it
  // run past the server's buffer. Attributes the fake does not check itself,
  // so a refusal can only have come from the server.
  cudnnBackendDescriptor_t victim = nullptr;
  CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &victim));
  const int64_t eight_bytes = 0;
  {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(victim));
    req.put<int32_t>(CUDNN_ATTR_TENSOR_STRIDES);
    req.put<int32_t>(CUDNN_TYPE_INT64);
    req.put<int64_t>(1000);             // a thousand elements...
    req.put<uint8_t>(1);
    req.put_sized(&eight_bytes, 8);     // ...in eight bytes
    rgpu::call(rgpu::API_cudnnBackendSetAttribute, req, &rsp);
    int32_t status = -1;
    if (!rsp.get(&status) || status != CUDNN_STATUS_BAD_PARAM) {
      std::fprintf(stderr, "FAIL: an attribute count past the bytes sent was accepted\n");
      g_failures++;
    }
  }
  {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(victim));
    req.put<int32_t>(CUDNN_ATTR_TENSOR_DIMENSIONS);
    req.put<int32_t>(CUDNN_TYPE_INT64);
    req.put<int64_t>(1000);             // room for a thousand, it says...
    req.put<uint8_t>(1);                // wants the count
    req.put<uint8_t>(1);                // and the values
    req.put_sized(&eight_bytes, 8);     // ...in eight bytes
    rgpu::call(rgpu::API_cudnnBackendGetAttribute, req, &rsp);
    int32_t status = -1;
    if (!rsp.get(&status) || status != CUDNN_STATUS_BAD_PARAM) {
      std::fprintf(stderr, "FAIL: a read-back with less room than it claimed was accepted\n");
      g_failures++;
    }
  }
  CHECK(cudnnBackendDestroyDescriptor(victim));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: cuDNN backend attributes survive the wire\n");
  return 0;
}
