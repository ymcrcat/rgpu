// A cuDNN that checks its arguments instead of computing.
//
// Linked into rgpu-server-fake so the backend graph API's marshalling can be
// tested without a GPU. The part worth testing is the attribute array: its
// width comes from the type enum rather than from anything in the call, so a
// wrong width sends the wrong number of bytes and would corrupt a graph rather
// than fail it.

#include <cstdio>
#include <cstring>

#include <cudnn.h>

namespace {

// Must match tests/cudnn_smoke.cpp.
const int64_t kDims[4] = {1, 3, 224, 224};
void* const kDevicePtr = reinterpret_cast<void*>(0xDEC1CEull);
constexpr int64_t kUid = 0x5150;
const int64_t kGetBack[3] = {11, 22, 33};

// What cudnnBackendCreateDescriptor below hands out for an operation, which is
// what an attribute array of descriptors has to contain by the time it gets
// here.
void* const kOpDescriptor = reinterpret_cast<void*>(
    0xDE5C0000ull +
    static_cast<unsigned>(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR));

}  // namespace

extern "C" {

cudnnStatus_t cudnnCreate(cudnnHandle_t* h) {
  if (!h) return CUDNN_STATUS_BAD_PARAM;
  *h = reinterpret_cast<cudnnHandle_t>(0xD00Dull);
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t h) {
  return h ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t h, cudaStream_t) {
  return h ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

size_t cudnnGetVersion(void) { return 90502; }

const char kLastError[] = "fake cuDNN has nothing to report";

void cudnnGetLastErrorString(char* message, size_t max_size) {
  if (!message || max_size == 0) return;
  std::snprintf(message, max_size, "%s", kLastError);
}

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t type,
                                           cudnnBackendDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  // The descriptor value encodes its type, so the test can tell them apart.
  *d = reinterpret_cast<cudnnBackendDescriptor_t>(0xDE5C0000ull +
                                                  static_cast<unsigned>(type));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t d) {
  return d ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t d) {
  return d ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t d,
                                       cudnnBackendAttributeName_t name,
                                       cudnnBackendAttributeType_t type,
                                       int64_t count, const void* array) {
  if (!d || !array) return CUDNN_STATUS_BAD_PARAM;

  // Eight-byte elements: a wrong width shows up immediately as wrong values.
  if (name == CUDNN_ATTR_TENSOR_DIMENSIONS) {
    if (type != CUDNN_TYPE_INT64 || count != 4) return CUDNN_STATUS_BAD_PARAM;
    const auto* v = static_cast<const int64_t*>(array);
    for (int i = 0; i < 4; i++) {
      if (v[i] != kDims[i]) return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
  }

  // A device pointer is a value in this process; it has to arrive unchanged.
  if (name == CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS) {
    if (type != CUDNN_TYPE_VOID_PTR || count != 1) return CUDNN_STATUS_BAD_PARAM;
    void* got = nullptr;
    std::memcpy(&got, array, sizeof(got));
    return got == kDevicePtr ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
  }

  if (name == CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS) {
    if (type != CUDNN_TYPE_INT64 || count != 1) return CUDNN_STATUS_BAD_PARAM;
    int64_t got = 0;
    std::memcpy(&got, array, sizeof(got));
    return got == kUid ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
  }

  // An array whose elements are descriptors. The client mints those handles
  // itself, so what arrives here must be the descriptor this library handed
  // out, not the client's handle for it.
  if (name == CUDNN_ATTR_OPERATIONGRAPH_OPS) {
    if (type != CUDNN_TYPE_BACKEND_DESCRIPTOR || count != 1) {
      return CUDNN_STATUS_BAD_PARAM;
    }
    void* got = nullptr;
    std::memcpy(&got, array, sizeof(got));
    return got == kOpDescriptor ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
  }

  // A four-byte enumeration, to cover the other width.
  if (name == CUDNN_ATTR_TENSOR_DATA_TYPE) {
    if (type != CUDNN_TYPE_DATA_TYPE || count != 1) return CUDNN_STATUS_BAD_PARAM;
    int got = 0;
    std::memcpy(&got, array, sizeof(got));
    return got == CUDNN_DATA_FLOAT ? CUDNN_STATUS_SUCCESS
                                   : CUDNN_STATUS_BAD_PARAM;
  }

  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t d,
                                       cudnnBackendAttributeName_t name,
                                       cudnnBackendAttributeType_t type,
                                       int64_t requested, int64_t* count,
                                       void* array) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;

  // Reading descriptors back: cuDNN fills the caller's own descriptors, so
  // this checks it was given real ones and leaves them alone. The client
  // should still see the handles it minted.
  if (name == CUDNN_ATTR_OPERATIONGRAPH_OPS) {
    if (type != CUDNN_TYPE_BACKEND_DESCRIPTOR || requested < 1 || !array) {
      return CUDNN_STATUS_BAD_PARAM;
    }
    void* got = nullptr;
    std::memcpy(&got, array, sizeof(got));
    if (got != kOpDescriptor) return CUDNN_STATUS_BAD_PARAM;
    if (count) *count = 1;
    return CUDNN_STATUS_SUCCESS;
  }

  if (name != CUDNN_ATTR_TENSOR_DIMENSIONS || type != CUDNN_TYPE_INT64) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const int64_t give = 3;
  if (count) *count = give;
  if (array && requested >= give) {
    std::memcpy(array, kGetBack, sizeof(kGetBack));
  }
  return CUDNN_STATUS_SUCCESS;
}

// The legacy path: a descriptor whose dimensions must arrive intact, and a
// batch norm whose alpha is a float, not a widened double.
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnTensorDescriptor_t>(0x7E4501ull);
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d) {
  return d ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t d,
                                         cudnnDataType_t type, int nbDims,
                                         const int dimA[], const int strideA[]) {
  if (!d || type != CUDNN_DATA_FLOAT || nbDims != 4) return CUDNN_STATUS_BAD_PARAM;
  for (int i = 0; i < 4; i++) {
    if (dimA[i] != static_cast<int>(kDims[i]) || strideA[i] != i + 1) {
      return CUDNN_STATUS_BAD_PARAM;
    }
  }
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t d, int requested,
                                         cudnnDataType_t* type, int* nbDims,
                                         int dimA[], int strideA[]) {
  if (!d || requested < 4) return CUDNN_STATUS_BAD_PARAM;
  if (type) *type = CUDNN_DATA_FLOAT;
  if (nbDims) *nbDims = 4;
  for (int i = 0; i < 4; i++) {
    if (dimA) dimA[i] = static_cast<int>(kDims[i]);
    if (strideA) strideA[i] = i + 1;
  }
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derived,
                                            cudnnTensorDescriptor_t x,
                                            cudnnBatchNormMode_t) {
  return derived && x ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha,
    const void* beta, cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y, cudnnTensorDescriptor_t bnDesc,
    const void* scale, const void* bias, const void* mean, const void* var,
    double epsilon) {
  if (!h || !xDesc || !yDesc || !bnDesc) return CUDNN_STATUS_BAD_PARAM;
  if (mode != CUDNN_BATCHNORM_SPATIAL) return CUDNN_STATUS_BAD_PARAM;
  // A float tensor, so these are floats. Widening on the way would give 0.
  if (*static_cast<const float*>(alpha) != 1.0f) return CUDNN_STATUS_BAD_PARAM;
  if (*static_cast<const float*>(beta) != 0.0f) return CUDNN_STATUS_BAD_PARAM;
  if (x != kDevicePtr || y != kDevicePtr) return CUDNN_STATUS_BAD_PARAM;
  if (!scale || !bias || !mean || !var) return CUDNN_STATUS_BAD_PARAM;
  if (epsilon != 1e-5) return CUDNN_STATUS_BAD_PARAM;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t h, cudnnBackendDescriptor_t plan,
                                  cudnnBackendDescriptor_t pack) {
  if (!h || !plan || !pack) return CUDNN_STATUS_BAD_PARAM;
  // The two descriptors must be distinct, or the client swapped them.
  return plan != pack ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

}  // extern "C"
