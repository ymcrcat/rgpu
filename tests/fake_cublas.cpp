// A cuBLAS and cuBLASLt that check their arguments instead of computing.
//
// Linked into rgpu-server-fake so the marshalling of these libraries can be
// tested without a GPU. Every value the client sends is compared against what
// tests/cublas_smoke.cpp said it would send, and a mismatch comes back as an
// error. That matters more here than for most calls: alpha and beta are host
// values or device pointers depending on a mode set earlier, and getting that
// wrong would produce quietly wrong arithmetic rather than a failure.

#include <cstring>

#include <cublas_v2.h>
#include <cublasLt.h>

namespace {

// The values tests/cublas_smoke.cpp sends. Any difference is a marshalling bug.
constexpr float kAlpha = 2.5f;
constexpr float kBeta = -0.75f;
constexpr int kM = 12, kN = 34, kK = 56;
constexpr int kLda = 78, kLdb = 90, kLdc = 21;
const void* const kA = reinterpret_cast<const void*>(0xA000ull);
const void* const kB = reinterpret_cast<const void*>(0xB000ull);
const void* const kC = reinterpret_cast<const void*>(0xC000ull);
const void* const kD = reinterpret_cast<const void*>(0xD000ull);
const void* const kWorkspace = reinterpret_cast<const void*>(0xE000ull);
constexpr size_t kWorkspaceSize = 4096;

bool near(float a, float b) { return a > b - 1e-6f && a < b + 1e-6f; }

}  // namespace

extern "C" {

// --- cuBLAS ---------------------------------------------------------------

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  *handle = reinterpret_cast<cublasHandle_t>(0xB1A5ull);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  return handle ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t) {
  return handle ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t) {
  return handle ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t) {
  return handle ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version) {
  if (!handle || !version) return CUBLAS_STATUS_INVALID_VALUE;
  *version = 120804;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                     size_t size) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  if (workspace != kWorkspace || size != kWorkspaceSize) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta,
                              float* C, int ldc) {
  if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (transa != CUBLAS_OP_T || transb != CUBLAS_OP_N) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (m != kM || n != kN || k != kK) return CUBLAS_STATUS_INVALID_VALUE;
  if (lda != kLda || ldb != kLdb || ldc != kLdc) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (A != kA || B != kB || C != kC) return CUBLAS_STATUS_INVALID_VALUE;
  // In host pointer mode these are readable values, not device addresses.
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  if (!near(*alpha, kAlpha) || !near(*beta, kBeta)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

// --- cuBLASLt -------------------------------------------------------------

cublasStatus_t cublasLtCreate(cublasLtHandle_t* h) {
  if (!h) return CUBLAS_STATUS_INVALID_VALUE;
  *h = reinterpret_cast<cublasLtHandle_t>(0x17000ull);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t h) {
  return h ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* d,
                                        cublasComputeType_t ct,
                                        cudaDataType_t st) {
  if (!d) return CUBLAS_STATUS_INVALID_VALUE;
  if (ct != CUBLAS_COMPUTE_32F || st != CUDA_R_32F) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  *d = reinterpret_cast<cublasLtMatmulDesc_t>(0x17100ull);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t d) {
  return d ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t attr,
    const void* buf, size_t size) {
  if (!d || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  // The transpose attribute is what the test sets; check it arrived intact.
  if (attr == CUBLASLT_MATMUL_DESC_TRANSA) {
    if (size != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
    int32_t v = 0;
    std::memcpy(&v, buf, sizeof(v));
    if (v != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* l,
                                          cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
  if (!l) return CUBLAS_STATUS_INVALID_VALUE;
  if (type != CUDA_R_32F) return CUBLAS_STATUS_INVALID_VALUE;
  if (rows != static_cast<uint64_t>(kM) || cols != static_cast<uint64_t>(kN) ||
      ld != kLda) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  *l = reinterpret_cast<cublasLtMatrixLayout_t>(0x17200ull);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t l) {
  return l ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* p) {
  if (!p) return CUBLAS_STATUS_INVALID_VALUE;
  *p = reinterpret_cast<cublasLtMatmulPreference_t>(0x17300ull);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t p) {
  return p ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t attr,
    const void* buf, size_t size) {
  if (!p || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  if (attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) {
    if (size != sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
    uint64_t v = 0;
    std::memcpy(&v, buf, sizeof(v));
    if (v != kWorkspaceSize) return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

// Fills the results with a recognisable pattern so the client can prove the
// array came back intact, including the opaque algo inside each entry.
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t h, cublasLtMatmulDesc_t d, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t p, int requested,
    cublasLtMatmulHeuristicResult_t results[], int* returned) {
  if (!h || !d || !p || !results || !returned || requested < 2) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  const int give = 2;
  for (int i = 0; i < give; i++) {
    std::memset(&results[i], 0, sizeof(results[i]));
    results[i].state = CUBLAS_STATUS_SUCCESS;
    results[i].workspaceSize = kWorkspaceSize;
    results[i].wavesCount = 1.0f + i;
    // The algo is opaque data; stamp it so a truncated copy is detectable.
    std::memset(&results[i].algo, 0x40 + i, sizeof(results[i].algo));
  }
  *returned = give;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t d,
                              const void* alpha, const void* A,
                              cublasLtMatrixLayout_t, const void* B,
                              cublasLtMatrixLayout_t, const void* beta,
                              const void* C, cublasLtMatrixLayout_t, void* D,
                              cublasLtMatrixLayout_t,
                              const cublasLtMatmulAlgo_t* algo, void* workspace,
                              size_t workspaceSize, cudaStream_t) {
  if (!h || !d) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (A != kA || B != kB || C != kC || D != kD) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (workspace != kWorkspace || workspaceSize != kWorkspaceSize) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  if (!near(*static_cast<const float*>(alpha), kAlpha) ||
      !near(*static_cast<const float*>(beta), kBeta)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  // The algo must be the first one the heuristic handed out, byte for byte.
  if (!algo) return CUBLAS_STATUS_INVALID_VALUE;
  const auto* bytes = reinterpret_cast<const unsigned char*>(algo);
  for (size_t i = 0; i < sizeof(*algo); i++) {
    if (bytes[i] != 0x40) return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
