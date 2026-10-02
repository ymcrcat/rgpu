// Real cuDNN math on a real GPU, compared against the CPU.
//
// Covers the two ways PyTorch reaches cuDNN through the shim: the backend
// graph API (a convolution here, which is how PyTorch runs every one) and the
// legacy batch-norm call. Build with the runtime as a shared library, then run
// it natively and through the shims (tests/hw/math_check.sh does both):
//
//   nvcc -cudart shared -I third_party/cudnn_include -o cudnn_check \
//     tests/hw/cudnn_check.cu -Xlinker $NV/cudnn/lib/libcudnn.so.9

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

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
#define DNN(expr) CHECK(expr, CUDNN_STATUS_SUCCESS)

static float value(int i, int salt) {
  return (float)((i * 7 + salt) % 9 - 4) * 0.25f;
}

static std::vector<float> filled(size_t n, int salt) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; i++) v[i] = value((int)i, salt);
  return v;
}

static float* upload(const std::vector<float>& v) {
  float* p = nullptr;
  if (cudaMalloc(&p, v.size() * sizeof(float)) != cudaSuccess) return nullptr;
  cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
  return p;
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

static int tensor(cudnnBackendDescriptor_t* t, int64_t id, const int64_t dims[4]) {
  const int64_t strides[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
  const cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  const int64_t align = 16;
  DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, t));
  DNN(cudnnBackendSetAttribute(*t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  DNN(cudnnBackendSetAttribute(*t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims));
  DNN(cudnnBackendSetAttribute(*t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides));
  DNN(cudnnBackendSetAttribute(*t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id));
  DNN(cudnnBackendSetAttribute(*t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align));
  DNN(cudnnBackendFinalize(*t));
  return 0;
}

// A 3x3 convolution, no padding, stride 1, through the backend graph API.
static int convolution(cudnnHandle_t dnn) {
  const int64_t N = 2, C = 3, H = 8, W = 8, K = 4, R = 3, S = 3;
  const int64_t P = H - R + 1, Q = W - S + 1;
  const int64_t xd[4] = {N, C, H, W}, wd[4] = {K, C, R, S}, yd[4] = {N, K, P, Q};
  const std::vector<float> x = filled(N * C * H * W, 1), w = filled(K * C * R * S, 2);

  std::vector<float> want(N * K * P * Q, 0.f);
  for (int64_t n = 0; n < N; n++)
    for (int64_t k = 0; k < K; k++)
      for (int64_t p = 0; p < P; p++)
        for (int64_t q = 0; q < Q; q++) {
          float sum = 0;
          for (int64_t c = 0; c < C; c++)
            for (int64_t r = 0; r < R; r++)
              for (int64_t s = 0; s < S; s++)
                sum += x[((n * C + c) * H + p + r) * W + q + s] *
                       w[((k * C + c) * R + r) * S + s];
          want[((n * K + k) * P + p) * Q + q] = sum;
        }

  cudnnBackendDescriptor_t tx, tw, ty, conv, op, graph, heur, plan = nullptr, pack;
  if (tensor(&tx, 'x', xd) || tensor(&tw, 'w', wd) || tensor(&ty, 'y', yd)) return 1;

  const cudnnDataType_t comp = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t dims = 2, ones[2] = {1, 1}, zeros[2] = {0, 0};
  DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &dims));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, ones));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, ones));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, zeros));
  DNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, zeros));
  DNN(cudnnBackendFinalize(conv));

  const float alpha = 1.f, beta = 0.f;
  DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));
  DNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tx));
  DNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tw));
  DNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ty));
  DNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv));
  DNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
  DNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  DNN(cudnnBackendFinalize(op));

  DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph));
  DNN(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  DNN(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &dnn));
  DNN(cudnnBackendFinalize(graph));

  const cudnnBackendHeurMode_t heur_mode = CUDNN_HEUR_MODE_A;
  DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));
  DNN(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph));
  DNN(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heur_mode));
  DNN(cudnnBackendFinalize(heur));
  enum { kConfigs = 8 };
  cudnnBackendDescriptor_t configs[kConfigs];
  for (auto& cfg : configs) DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfg));
  int64_t count = 0;
  DNN(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                               kConfigs, &count, configs));

  // The first configuration that finalizes into a plan is the one used.
  for (int64_t i = 0; i < count && !plan; i++) {
    cudnnBackendDescriptor_t candidate;
    DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &candidate));
    DNN(cudnnBackendSetAttribute(candidate, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &dnn));
    DNN(cudnnBackendSetAttribute(candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[i]));
    if (cudnnBackendFinalize(candidate) == CUDNN_STATUS_SUCCESS) plan = candidate;
    else cudnnBackendDestroyDescriptor(candidate);
  }
  if (!plan) {
    std::printf("FAIL no engine configuration finalized (%lld offered)\n", (long long)count);
    return 1;
  }
  int64_t ws_size = 0, got_count = 0;
  DNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64,
                               1, &got_count, &ws_size));

  float *dx = upload(x), *dw = upload(w), *dy = nullptr;
  void* ws = nullptr;
  CUDA(cudaMalloc(&dy, want.size() * sizeof(float)));
  if (ws_size) CUDA(cudaMalloc(&ws, ws_size));
  void* pointers[3] = {dx, dw, dy};
  const int64_t ids[3] = {'x', 'w', 'y'};
  DNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack));
  DNN(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers));
  DNN(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, ids));
  DNN(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws));
  DNN(cudnnBackendFinalize(pack));
  DNN(cudnnBackendExecute(dnn, plan, pack));

  std::vector<float> got(want.size());
  CUDA(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  compare("cuDNN backend-graph convolution", got, want);

  for (auto d : {pack, plan, heur, graph, op, conv, tx, tw, ty}) cudnnBackendDestroyDescriptor(d);
  for (auto& cfg : configs) cudnnBackendDestroyDescriptor(cfg);
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dy);
  if (ws) cudaFree(ws);
  return 0;
}

// Per-channel y = scale * (x - mean) / sqrt(var + eps) + bias.
static int batch_norm(cudnnHandle_t dnn) {
  const int N = 2, C = 3, H = 4, W = 5;
  const double eps = 1e-5;
  const std::vector<float> x = filled(N * C * H * W, 3);
  const std::vector<float> scale = {1.5f, -0.5f, 2.f}, bias = {0.25f, 0.f, -1.f};
  const std::vector<float> mean = {0.1f, -0.2f, 0.3f}, var = {1.f, 0.5f, 2.f};
  std::vector<float> want(x.size());
  for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
      for (int i = 0; i < H * W; i++) {
        size_t at = (n * C + c) * H * W + i;
        want[at] = scale[c] * (x[at] - mean[c]) / std::sqrt(var[c] + (float)eps) + bias[c];
      }

  cudnnTensorDescriptor_t xd, bnd;
  const int dims[4] = {N, C, H, W}, strides[4] = {C * H * W, H * W, W, 1};
  DNN(cudnnCreateTensorDescriptor(&xd));
  DNN(cudnnSetTensorNdDescriptor(xd, CUDNN_DATA_FLOAT, 4, dims, strides));
  DNN(cudnnCreateTensorDescriptor(&bnd));
  DNN(cudnnDeriveBNTensorDescriptor(bnd, xd, CUDNN_BATCHNORM_SPATIAL));

  float *dx = upload(x), *ds = upload(scale), *db = upload(bias), *dm = upload(mean),
        *dv = upload(var), *dy = nullptr;
  CUDA(cudaMalloc(&dy, x.size() * sizeof(float)));
  const float alpha = 1.f, beta = 0.f;
  DNN(cudnnBatchNormalizationForwardInference(dnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, xd, dx,
                                              xd, dy, bnd, ds, db, dm, dv, eps));
  std::vector<float> got(x.size());
  CUDA(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  compare("cudnnBatchNormalizationForwardInference", got, want);

  cudnnDestroyTensorDescriptor(xd);
  cudnnDestroyTensorDescriptor(bnd);
  for (float* p : {dx, ds, db, dm, dv, dy}) cudaFree(p);
  return 0;
}

int main() {
  cudnnHandle_t dnn;
  DNN(cudnnCreate(&dnn));
  std::printf("cuDNN %zu\n", cudnnGetVersion());
  int broken = convolution(dnn) | batch_norm(dnn);
  cudnnDestroy(dnn);
  return broken || g_failures ? 1 : 0;
}
