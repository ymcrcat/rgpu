// A cuDNN that checks its arguments instead of computing.
//
// Linked into rgpu-server-fake so the backend graph API's marshalling can be
// tested without a GPU. The part worth testing is the attribute array: its
// width comes from the type enum rather than from anything in the call, so a
// wrong width sends the wrong number of bytes and would corrupt a graph rather
// than fail it.

#include <cstring>

#include <cudnn.h>

namespace {

// Must match tests/cudnn_smoke.cpp.
const int64_t kDims[4] = {1, 3, 224, 224};
void* const kDevicePtr = reinterpret_cast<void*>(0xDEC1CEull);
constexpr int64_t kUid = 0x5150;
const int64_t kGetBack[3] = {11, 22, 33};

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

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t h, cudnnBackendDescriptor_t plan,
                                  cudnnBackendDescriptor_t pack) {
  if (!h || !plan || !pack) return CUDNN_STATUS_BAD_PARAM;
  // The two descriptors must be distinct, or the client swapped them.
  return plan != pack ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

}  // extern "C"
