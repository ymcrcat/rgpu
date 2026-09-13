// The cuBLAS calls PyTorch makes, forwarded to the GPU host.
//
// cuBLAS cannot run on the client. Like the stock CUDA runtime, it calls
// cuGetExportTable while initialising and fails without it, and that table
// holds function pointers into the driver's own process, which cannot cross a
// network. So the real cuBLAS runs on the server and these functions carry the
// arguments to it.
//
// Matrix arguments are device pointers: values in the server's address space
// that we pass through untouched. The scalars alpha and beta are the exception,
// because whether they point at host or device memory depends on the handle's
// pointer mode, which we therefore track.
//
// Definitions here are strong and override the weak, logging ones in
// client/generated/cublas_stubs.cpp.

#include <cstring>
#include <map>
#include <mutex>

#include <cublas_v2.h>

#include "client/rpc.h"
#include "common/cublas_ids.h"

namespace {

// Every global below that a CUDA call can reach is allocated and never
// destroyed. Threads keep calling in while the process exits - a thread's
// late thread-local destructors free device memory, say - and a mutex or
// container that static destruction has already taken down is undefined
// behaviour (on libc++ a destroyed mutex throws, from places that cannot).

// Pointer mode per handle. cuBLAS defaults to host, meaning alpha and beta
// point at host memory whose values have to travel with the call. In device
// mode they are device pointers and pass through like any other.
auto& g_mode_mu = *new std::mutex();
auto& g_modes = *new std::map<cublasHandle_t, cublasPointerMode_t>();

cublasPointerMode_t pointer_mode(cublasHandle_t h) {
  std::lock_guard<std::mutex> lk(g_mode_mu);
  auto it = g_modes.find(h);
  return it == g_modes.end() ? CUBLAS_POINTER_MODE_HOST : it->second;
}

cublasStatus_t from_cu(CUresult r) {
  return r == CUDA_SUCCESS ? CUBLAS_STATUS_SUCCESS
                           : CUBLAS_STATUS_EXECUTION_FAILED;
}

// Every forwarded call returns a cublasStatus_t from the server, carried in
// the reply's payload rather than its result field, which is a CUresult.
cublasStatus_t send(uint32_t id, rgpu::Buffer& req, rgpu::Buffer* rsp) {
  CUresult r = rgpu::call(id, req, rsp);
  if (r != CUDA_SUCCESS) return from_cu(r);
  int32_t status = CUBLAS_STATUS_INTERNAL_ERROR;
  if (!rsp->get(&status)) return CUBLAS_STATUS_INTERNAL_ERROR;
  return static_cast<cublasStatus_t>(status);
}

void put_handle(rgpu::Buffer& b, cublasHandle_t h) {
  b.put<uint64_t>(reinterpret_cast<uint64_t>(h));
}

void put_devptr(rgpu::Buffer& b, const void* p) {
  b.put<uint64_t>(reinterpret_cast<uint64_t>(p));
}

// A scalar that is either a host value to copy or a device pointer to pass
// through, depending on the handle's pointer mode. The flag travels so the
// server knows which it received.
void put_scalar(rgpu::Buffer& b, cublasHandle_t h, const void* p, size_t size) {
  const bool host = pointer_mode(h) == CUBLAS_POINTER_MODE_HOST;
  b.put<uint8_t>(host ? 1 : 0);
  b.put<uint8_t>(p ? 1 : 0);
  if (!p) return;
  if (host) {
    b.put_bytes(p, size);
  } else {
    b.put<uint64_t>(reinterpret_cast<uint64_t>(p));
  }
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  cublasStatus_t s = send(rgpu::API_cublasCreate, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  uint64_t h = 0;
  if (!rsp.get(&h)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *handle = reinterpret_cast<cublasHandle_t>(h);
  return s;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  cublasStatus_t s = send(rgpu::API_cublasDestroy, req, &rsp);
  std::lock_guard<std::mutex> lk(g_mode_mu);
  g_modes.erase(handle);
  return s;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<uint64_t>(reinterpret_cast<uint64_t>(stream));
  return send(rgpu::API_cublasSetStream, req, &rsp);
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle,
                                  cudaStream_t* streamId) {
  if (!streamId) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  cublasStatus_t s = send(rgpu::API_cublasGetStream, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  uint64_t v = 0;
  if (!rsp.get(&v)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *streamId = reinterpret_cast<cudaStream_t>(v);
  return s;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t mode) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(static_cast<int32_t>(mode));
  cublasStatus_t s = send(rgpu::API_cublasSetPointerMode, req, &rsp);
  if (s == CUBLAS_STATUS_SUCCESS) {
    // Remembered locally because it decides how alpha and beta are marshalled.
    std::lock_guard<std::mutex> lk(g_mode_mu);
    g_modes[handle] = mode;
  }
  return s;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t* mode) {
  if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
  *mode = pointer_mode(handle);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(static_cast<int32_t>(mode));
  return send(rgpu::API_cublasSetMathMode, req, &rsp);
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
  if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  cublasStatus_t s = send(rgpu::API_cublasGetMathMode, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  int32_t v = 0;
  if (!rsp.get(&v)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *mode = static_cast<cublasMath_t>(v);
  return s;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                     size_t workspaceSizeInBytes) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  put_devptr(req, workspace);
  req.put<uint64_t>(workspaceSizeInBytes);
  return send(rgpu::API_cublasSetWorkspace, req, &rsp);
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version) {
  if (!version) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  cublasStatus_t s = send(rgpu::API_cublasGetVersion, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  int32_t v = 0;
  if (!rsp.get(&v)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *version = v;
  return s;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  req.put<int32_t>(static_cast<int32_t>(type));
  cublasStatus_t s = send(rgpu::API_cublasGetProperty, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  int32_t v = 0;
  if (!rsp.get(&v)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *value = v;
  return s;
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle,
                                      int smCountTarget) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(smCountTarget);
  return send(rgpu::API_cublasSetSmCountTarget, req, &rsp);
}

cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle,
                                      int* smCountTarget) {
  if (!smCountTarget) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  cublasStatus_t s = send(rgpu::API_cublasGetSmCountTarget, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  int32_t v = 0;
  if (!rsp.get(&v)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *smCountTarget = v;
  return s;
}

// --- the matrix multiplies -------------------------------------------------

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta,
                              float* C, int ldc) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(transa);
  req.put<int32_t>(transb);
  req.put<int32_t>(m);
  req.put<int32_t>(n);
  req.put<int32_t>(k);
  put_scalar(req, handle, alpha, sizeof(float));
  put_devptr(req, A);
  req.put<int32_t>(lda);
  put_devptr(req, B);
  req.put<int32_t>(ldb);
  put_scalar(req, handle, beta, sizeof(float));
  put_devptr(req, C);
  req.put<int32_t>(ldc);
  return send(rgpu::API_cublasSgemm, req, &rsp);
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double* alpha, const double* A, int lda,
                              const double* B, int ldb, const double* beta,
                              double* C, int ldc) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(transa);
  req.put<int32_t>(transb);
  req.put<int32_t>(m);
  req.put<int32_t>(n);
  req.put<int32_t>(k);
  put_scalar(req, handle, alpha, sizeof(double));
  put_devptr(req, A);
  req.put<int32_t>(lda);
  put_devptr(req, B);
  req.put<int32_t>(ldb);
  put_scalar(req, handle, beta, sizeof(double));
  put_devptr(req, C);
  req.put<int32_t>(ldc);
  return send(rgpu::API_cublasDgemm, req, &rsp);
}

// alpha and beta here are typed by computeType rather than by the matrices, so
// their size comes from that.
static size_t compute_scalar_size(cublasComputeType_t ct) {
  switch (ct) {
    case CUBLAS_COMPUTE_64F:
    case CUBLAS_COMPUTE_64F_PEDANTIC:
      return sizeof(double);
    case CUBLAS_COMPUTE_32I:
    case CUBLAS_COMPUTE_32I_PEDANTIC:
      return sizeof(int32_t);
    default:
      // Every 32F and 16F variant takes float scalars.
      return sizeof(float);
  }
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A, cudaDataType Atype,
                            int lda, const void* B, cudaDataType Btype, int ldb,
                            const void* beta, void* C, cudaDataType Ctype,
                            int ldc, cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo) {
  const size_t ss = compute_scalar_size(computeType);
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(transa);
  req.put<int32_t>(transb);
  req.put<int32_t>(m);
  req.put<int32_t>(n);
  req.put<int32_t>(k);
  req.put<int32_t>(static_cast<int32_t>(computeType));
  req.put<int32_t>(static_cast<int32_t>(algo));
  put_scalar(req, handle, alpha, ss);
  put_devptr(req, A);
  req.put<int32_t>(static_cast<int32_t>(Atype));
  req.put<int32_t>(lda);
  put_devptr(req, B);
  req.put<int32_t>(static_cast<int32_t>(Btype));
  req.put<int32_t>(ldb);
  put_scalar(req, handle, beta, ss);
  put_devptr(req, C);
  req.put<int32_t>(static_cast<int32_t>(Ctype));
  req.put<int32_t>(ldc);
  return send(rgpu::API_cublasGemmEx, req, &rsp);
}

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
    int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb,
    long long int strideB, const void* beta, void* C, cudaDataType Ctype,
    int ldc, long long int strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  const size_t ss = compute_scalar_size(computeType);
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(transa);
  req.put<int32_t>(transb);
  req.put<int32_t>(m);
  req.put<int32_t>(n);
  req.put<int32_t>(k);
  req.put<int32_t>(static_cast<int32_t>(computeType));
  req.put<int32_t>(static_cast<int32_t>(algo));
  req.put<int32_t>(batchCount);
  put_scalar(req, handle, alpha, ss);
  put_devptr(req, A);
  req.put<int32_t>(static_cast<int32_t>(Atype));
  req.put<int32_t>(lda);
  req.put<int64_t>(strideA);
  put_devptr(req, B);
  req.put<int32_t>(static_cast<int32_t>(Btype));
  req.put<int32_t>(ldb);
  req.put<int64_t>(strideB);
  put_scalar(req, handle, beta, ss);
  put_devptr(req, C);
  req.put<int32_t>(static_cast<int32_t>(Ctype));
  req.put<int32_t>(ldc);
  req.put<int64_t>(strideC);
  return send(rgpu::API_cublasGemmStridedBatchedEx, req, &rsp);
}

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batchCount) {
  rgpu::Buffer req, rsp;
  put_handle(req, handle);
  req.put<int32_t>(transa);
  req.put<int32_t>(transb);
  req.put<int32_t>(m);
  req.put<int32_t>(n);
  req.put<int32_t>(k);
  req.put<int32_t>(batchCount);
  put_scalar(req, handle, alpha, sizeof(float));
  put_devptr(req, A);
  req.put<int32_t>(lda);
  req.put<int64_t>(strideA);
  put_devptr(req, B);
  req.put<int32_t>(ldb);
  req.put<int64_t>(strideB);
  put_scalar(req, handle, beta, sizeof(float));
  put_devptr(req, C);
  req.put<int32_t>(ldc);
  req.put<int64_t>(strideC);
  return send(rgpu::API_cublasSgemmStridedBatched, req, &rsp);
}

}  // extern "C"
