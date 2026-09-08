// Checks that cuBLAS and cuBLASLt arguments survive the wire, without a GPU.
//
// The server's fake libraries know every value this sends and reject anything
// else, so a passing run means the marshalling is right rather than merely
// that nothing crashed. That is worth testing separately because the awkward
// parts here fail quietly: alpha and beta are host values or device pointers
// depending on a mode set earlier, their width comes from a scale type set
// earlier still, and the algo chosen by the heuristic is opaque data that has
// to return byte for byte.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./cublas_smoke

#include <cstdio>
#include <cstring>

#include <cublas_v2.h>
#include <cublasLt.h>

static int g_failures = 0;

#define CHECK(call)                                                        \
  do {                                                                     \
    cublasStatus_t s_ = (call);                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,   \
                   #call, s_);                                            \
      g_failures++;                                                        \
    }                                                                      \
  } while (0)

namespace {
// Must match tests/fake_cublas.cpp.
constexpr float kAlpha = 2.5f;
constexpr float kBeta = -0.75f;
constexpr int kM = 12, kN = 34, kK = 56;
constexpr int kLda = 78, kLdb = 90, kLdc = 21;
void* const kA = reinterpret_cast<void*>(0xA000ull);
void* const kB = reinterpret_cast<void*>(0xB000ull);
void* const kC = reinterpret_cast<void*>(0xC000ull);
void* const kD = reinterpret_cast<void*>(0xD000ull);
void* const kWorkspace = reinterpret_cast<void*>(0xE000ull);
constexpr size_t kWorkspaceSize = 4096;
}  // namespace

int main() {
  // --- cuBLAS -------------------------------------------------------------
  cublasHandle_t h = nullptr;
  CHECK(cublasCreate(&h));
  CHECK(cublasSetStream(h, nullptr));
  CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
  CHECK(cublasSetWorkspace(h, kWorkspace, kWorkspaceSize));

  int version = 0;
  CHECK(cublasGetVersion(h, &version));
  if (version <= 0) {
    std::fprintf(stderr, "FAIL: cublasGetVersion returned %d\n", version);
    g_failures++;
  }

  // The device pointers here are never dereferenced by anything; they stand
  // for values in the server's address space and only have to arrive intact.
  const float alpha = kAlpha, beta = kBeta;
  CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, kM, kN, kK, &alpha,
                    static_cast<const float*>(kA), kLda,
                    static_cast<const float*>(kB), kLdb, &beta,
                    static_cast<float*>(kC), kLdc));

  // A wrong scalar must be rejected, or the check above proves nothing.
  const float wrong = kAlpha + 1.0f;
  if (cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, kM, kN, kK, &wrong,
                  static_cast<const float*>(kA), kLda,
                  static_cast<const float*>(kB), kLdb, &beta,
                  static_cast<float*>(kC), kLdc) == CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: a gemm with the wrong alpha was accepted\n");
    g_failures++;
  }
  CHECK(cublasDestroy(h));

  // --- cuBLASLt -----------------------------------------------------------
  cublasLtHandle_t lt = nullptr;
  CHECK(cublasLtCreate(&lt));

  cublasLtMatmulDesc_t desc = nullptr;
  CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  int32_t op = CUBLAS_OP_T;
  CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op,
                                       sizeof(op)));

  cublasLtMatrixLayout_t layout = nullptr;
  CHECK(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, kM, kN, kLda));

  cublasLtMatmulPreference_t pref = nullptr;
  CHECK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = kWorkspaceSize;
  CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

  cublasLtMatmulHeuristicResult_t results[4];
  std::memset(results, 0, sizeof(results));
  int returned = 0;
  CHECK(cublasLtMatmulAlgoGetHeuristic(lt, desc, layout, layout, layout, layout,
                                       pref, 4, results, &returned));
  if (returned != 2) {
    std::fprintf(stderr, "FAIL: expected 2 heuristics, got %d\n", returned);
    g_failures++;
  } else {
    // The opaque algo has to come back whole; a truncated copy would leave
    // some of these bytes zero.
    const auto* a0 = reinterpret_cast<const unsigned char*>(&results[0].algo);
    for (size_t i = 0; i < sizeof(results[0].algo); i++) {
      if (a0[i] != 0x40) {
        std::fprintf(stderr, "FAIL: algo byte %zu is %02x, expected 40\n", i,
                     a0[i]);
        g_failures++;
        break;
      }
    }
    if (results[0].workspaceSize != kWorkspaceSize) {
      std::fprintf(stderr, "FAIL: heuristic workspaceSize did not survive\n");
      g_failures++;
    }
  }

  CHECK(cublasLtMatmul(lt, desc, &alpha, kA, layout, kB, layout, &beta, kC,
                       layout, kD, layout, &results[0].algo, kWorkspace,
                       kWorkspaceSize, nullptr));

  CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CHECK(cublasLtMatrixLayoutDestroy(layout));
  CHECK(cublasLtMatmulDescDestroy(desc));
  CHECK(cublasLtDestroy(lt));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: cuBLAS and cuBLASLt arguments survive the wire\n");
  return 0;
}
