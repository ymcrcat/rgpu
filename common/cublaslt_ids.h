// Wire ids for forwarded cuBLASLt calls.
//
// cuBLASLt is worse than cuBLAS on a remoted driver: rather than reporting an
// error it segfaults inside cublasLtMatmulAlgoGetHeuristic, which is where
// PyTorch's addmm lands by default. So the real library runs on the GPU host
// and these calls carry the arguments to it.
#pragma once

#include <cstdint>

namespace rgpu {

enum CublasLtId : uint32_t {
  kCublasLtBase = 0x51000000u,

  API_cublasLtCreate = kCublasLtBase + 1,
  API_cublasLtDestroy = kCublasLtBase + 2,
  API_cublasLtGetVersion = kCublasLtBase + 3,
  API_cublasLtGetCudartVersion = kCublasLtBase + 4,
  API_cublasLtGetProperty = kCublasLtBase + 5,

  API_cublasLtMatmulDescCreate = kCublasLtBase + 10,
  API_cublasLtMatmulDescDestroy = kCublasLtBase + 11,
  API_cublasLtMatmulDescSetAttribute = kCublasLtBase + 12,
  API_cublasLtMatmulDescGetAttribute = kCublasLtBase + 13,

  API_cublasLtMatrixLayoutCreate = kCublasLtBase + 20,
  API_cublasLtMatrixLayoutDestroy = kCublasLtBase + 21,
  API_cublasLtMatrixLayoutSetAttribute = kCublasLtBase + 22,
  API_cublasLtMatrixLayoutGetAttribute = kCublasLtBase + 23,

  API_cublasLtMatmulPreferenceCreate = kCublasLtBase + 30,
  API_cublasLtMatmulPreferenceDestroy = kCublasLtBase + 31,
  API_cublasLtMatmulPreferenceSetAttribute = kCublasLtBase + 32,
  API_cublasLtMatmulPreferenceGetAttribute = kCublasLtBase + 33,

  API_cublasLtMatmulAlgoGetHeuristic = kCublasLtBase + 40,
  API_cublasLtMatmul = kCublasLtBase + 41,
};

}  // namespace rgpu
