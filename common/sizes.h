// Sizes the client and the server must agree on.
//
// Shared because the server has to check them too: a size the client sends is
// only a claim, and the server compares it with the size the driver will
// really read or write before letting the driver near the buffer.
#pragma once

#include <cstddef>

#include <cuda.h>

namespace rgpu {

// The size of the value cuPointerGetAttribute writes for an attribute.
inline size_t pointer_attr_size(unsigned int attribute) {
  switch (attribute) {
    // Handles and addresses: one pointer-sized value.
    case CU_POINTER_ATTRIBUTE_CONTEXT:
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
      return sizeof(void*);
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
    case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
      return sizeof(size_t);
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
      return sizeof(unsigned int);
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_MAPPED:
      return sizeof(bool);
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
    case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
      return sizeof(unsigned long long);
    default:
      // P2P tokens and mempool handles among others. Zero makes the call
      // fail rather than move a wrong number of bytes.
      return 0;
  }
}

}  // namespace rgpu
