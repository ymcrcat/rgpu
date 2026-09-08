// Wire ids for forwarded cuBLAS calls.
//
// cuBLAS cannot run on the client: like the stock CUDA runtime it calls
// cuGetExportTable while initialising and fails without it. So the real
// library runs on the GPU host and these calls carry the arguments to it.
//
// Kept in their own range so they can never collide with the driver API ids or
// the internal ones.
#pragma once

#include <cstdint>

namespace rgpu {

enum CublasId : uint32_t {
  kCublasBase = 0x50000000u,

  API_cublasCreate = kCublasBase + 1,
  API_cublasDestroy = kCublasBase + 2,
  API_cublasSetStream = kCublasBase + 3,
  API_cublasGetStream = kCublasBase + 4,
  API_cublasSetPointerMode = kCublasBase + 5,
  API_cublasGetPointerMode = kCublasBase + 6,
  API_cublasSetMathMode = kCublasBase + 7,
  API_cublasGetMathMode = kCublasBase + 8,
  API_cublasSetWorkspace = kCublasBase + 9,
  API_cublasGetVersion = kCublasBase + 10,
  API_cublasGetProperty = kCublasBase + 11,
  API_cublasSetSmCountTarget = kCublasBase + 12,
  API_cublasGetSmCountTarget = kCublasBase + 13,

  API_cublasSgemm = kCublasBase + 20,
  API_cublasDgemm = kCublasBase + 21,
  API_cublasGemmEx = kCublasBase + 22,
  API_cublasGemmStridedBatchedEx = kCublasBase + 23,
  API_cublasSgemmStridedBatched = kCublasBase + 24,
};

}  // namespace rgpu
