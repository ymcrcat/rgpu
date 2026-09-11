// The width of one element of a cuDNN backend attribute array.
//
// Shared by the client and the server. The width is not in the call - it
// follows from the type - so the server cannot trust the client's element
// count on its own: cuDNN reads or writes count times this many bytes, and the
// server checks that against the bytes that actually arrived.
#pragma once

#include <cstddef>

#include <cudnn.h>

namespace rgpu {

inline size_t cudnn_element_size(cudnnBackendAttributeType_t t) {
  switch (t) {
    case CUDNN_TYPE_INT64:
    case CUDNN_TYPE_DOUBLE:
      return 8;
    // A handle, a device pointer and a descriptor are all pointer-sized values
    // belonging to the server. Copying the bytes is exactly right.
    case CUDNN_TYPE_VOID_PTR:
    case CUDNN_TYPE_HANDLE:
    case CUDNN_TYPE_BACKEND_DESCRIPTOR:
      return sizeof(void*);
    case CUDNN_TYPE_FLOAT:
    case CUDNN_TYPE_INT32:
      return 4;
    case CUDNN_TYPE_BOOLEAN:
      return sizeof(bool);
    case CUDNN_TYPE_CHAR:
      return 1;
    case CUDNN_TYPE_FRACTION:
      return sizeof(cudnnFraction_t);
    default:
      // Every remaining type is an enumeration, which is int-sized.
      return sizeof(int);
  }
}

// Whether a type is one this build of the headers knows about.
inline bool cudnn_type_known(cudnnBackendAttributeType_t t) {
  return t >= CUDNN_TYPE_HANDLE && t <= CUDNN_TYPE_TENSOR_REORDERING_MODE;
}

}  // namespace rgpu
