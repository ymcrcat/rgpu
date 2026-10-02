// Real cuBLAS and cuBLASLt math on a real GPU, compared against the CPU.
//
// The smoke tests in tests/ check marshalling against fake libraries; this
// checks the numbers. It also launches a kernel with <<<>>>, so it covers the
// runtime API outside PyTorch. Build it with the runtime as a shared library,
// then run it natively and through the shims (tests/hw/math_check.sh does
// both):
//
//   nvcc -cudart shared -I third_party/cublas_include -o cublas_check \
//     tests/hw/cublas_check.cu -Xlinker $NV/cublas/lib/libcublas.so.12 \
//     -Xlinker $NV/cublas/lib/libcublasLt.so.12

#include <cmath>
#include <cstdio>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static int g_failures = 0;

#define CHECK(expr, ok)                                                   \
  do {                                                                    \
    auto r_ = (expr);                                                     \
    if (r_ != (ok)) {                                                     \
      std::printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr,    \
                  (int)r_);                                               \
      return 1;                                                           \
    }                                                                     \
  } while (0)
#define CUDA(expr) CHECK(expr, cudaSuccess)
#define BLAS(expr) CHECK(expr, CUBLAS_STATUS_SUCCESS)

// Small exact values, so GPU and CPU fill identically.
__host__ __device__ float value(int i, int salt) {
  return (float)((i * 7 + salt) % 9 - 4) * 0.25f;
}

__global__ void fill(float* p, int n, int salt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = value(i, salt);
}

// Column-major C = A(m x k) * B(k x n).
static std::vector<float> reference(int m, int n, int k) {
  std::vector<float> c(m * n, 0.f);
  for (int j = 0; j < n; j++)
    for (int p = 0; p < k; p++)
      for (int i = 0; i < m; i++)
        c[i + j * m] += value(i + p * m, 1) * value(p + j * k, 2);
  return c;
}

static void compare(const char* name, const std::vector<float>& got,
                    const std::vector<float>& want) {
  double worst = 0;
  for (size_t i = 0; i < got.size(); i++)
    worst = std::fmax(worst, std::fabs(got[i] - want[i]));
  bool ok = worst <= 1e-3;
  std::printf("%s %s (max error %.2g over %zu values)\n", ok ? "PASS" : "FAIL",
              name, worst, got.size());
  if (!ok) g_failures++;
}

int main() {
  const int m = 96, n = 80, k = 64;
  float *a, *b, *c;
  CUDA(cudaMalloc(&a, m * k * sizeof(float)));
  CUDA(cudaMalloc(&b, k * n * sizeof(float)));
  CUDA(cudaMalloc(&c, m * n * sizeof(float)));
  fill<<<(m * k + 255) / 256, 256>>>(a, m * k, 1);
  fill<<<(k * n + 255) / 256, 256>>>(b, k * n, 2);
  CUDA(cudaGetLastError());

  std::vector<float> host(m * k);
  CUDA(cudaMemcpy(host.data(), a, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> want_a(m * k);
  for (int i = 0; i < m * k; i++) want_a[i] = value(i, 1);
  compare("kernel launched with <<<>>>", host, want_a);

  const std::vector<float> want = reference(m, n, k);
  std::vector<float> got(m * n);
  const float alpha = 1.f, beta = 0.f;

  cublasHandle_t blas;
  BLAS(cublasCreate(&blas));
  BLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, m, b, k,
                   &beta, c, m));
  CUDA(cudaMemcpy(got.data(), c, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  compare("cublasSgemm", got, want);

  CUDA(cudaMemset(c, 0, m * n * sizeof(float)));
  BLAS(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, CUDA_R_32F,
                    m, b, CUDA_R_32F, k, &beta, c, CUDA_R_32F, m,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CUDA(cudaMemcpy(got.data(), c, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  compare("cublasGemmEx", got, want);
  BLAS(cublasDestroy(blas));

  cublasLtHandle_t lt;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulPreference_t pref;
  BLAS(cublasLtCreate(&lt));
  BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, m, k, m));
  BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, n, k));
  BLAS(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, m, n, m));
  const size_t ws_size = 4 << 20;
  void* ws;
  CUDA(cudaMalloc(&ws, ws_size));
  BLAS(cublasLtMatmulPreferenceCreate(&pref));
  BLAS(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));
  cublasLtMatmulHeuristicResult_t heur;
  int found = 0;
  BLAS(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &found));
  if (found == 0) {
    std::printf("FAIL cublasLtMatmulAlgoGetHeuristic found no algorithm\n");
    return 1;
  }
  CUDA(cudaMemset(c, 0, m * n * sizeof(float)));
  BLAS(cublasLtMatmul(lt, op, &alpha, a, la, b, lb, &beta, c, lc, c, lc,
                      &heur.algo, ws, ws_size, 0));
  CUDA(cudaMemcpy(got.data(), c, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  compare("cublasLtMatmul", got, want);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
  cublasLtDestroy(lt);

  cudaFree(ws);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return g_failures ? 1 : 0;
}
