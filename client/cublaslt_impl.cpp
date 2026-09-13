// The cuBLASLt calls PyTorch makes, forwarded to the GPU host.
//
// PyTorch's addmm goes through cuBLASLt by default, and the real library does
// not merely fail on a remoted driver, it segfaults inside
// cublasLtMatmulAlgoGetHeuristic. So it runs on the server instead.
//
// Three things need care:
//
//   Descriptors, layouts and preferences are opaque handles created and
//   destroyed through the API. We hold the server's values and pass them
//   through, so there is no translation table.
//
//   Attribute setters and getters carry an explicit size, which makes them the
//   easy case: the bytes travel verbatim. That is right even when an attribute
//   value is itself a device pointer, since such a pointer is already a value
//   in the server's address space.
//
//   alpha and beta are host values or device pointers depending on the
//   descriptor's pointer mode, and their width comes from its scale type.
//   Neither is visible at the call, so we track both per descriptor.
//
// Definitions here are strong and override the weak, logging ones in
// client/generated/cublaslt_stubs.cpp.

#include <cstring>
#include <map>
#include <mutex>

#include <cublasLt.h>

#include "client/rpc.h"
#include "common/cublaslt_ids.h"

namespace {

// Every global below that a CUDA call can reach is allocated and never
// destroyed. Threads keep calling in while the process exits - a thread's
// late thread-local destructors free device memory, say - and a mutex or
// container that static destruction has already taken down is undefined
// behaviour (on libc++ a destroyed mutex throws, from places that cannot).

struct DescState {
  cudaDataType_t scale_type = CUDA_R_32F;
  bool device_pointers = false;  // CUBLASLT_POINTER_MODE_DEVICE
};

auto& g_desc_mu = *new std::mutex();
auto& g_descs = *new std::map<cublasLtMatmulDesc_t, DescState>();

DescState desc_state(cublasLtMatmulDesc_t d) {
  std::lock_guard<std::mutex> lk(g_desc_mu);
  auto it = g_descs.find(d);
  return it == g_descs.end() ? DescState{} : it->second;
}

// Width of one alpha or beta value.
size_t scalar_size(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_64F: return 8;
    case CUDA_C_64F: return 16;
    case CUDA_R_32F: case CUDA_R_32I: case CUDA_R_32U: return 4;
    case CUDA_C_32F: case CUDA_C_32I: case CUDA_C_32U: return 8;
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_C_16F: case CUDA_C_16BF: return 4;
    case CUDA_R_8I: case CUDA_R_8U: return 1;
    case CUDA_C_8I: case CUDA_C_8U: return 2;
    default:
      rgpu::log("unknown cuBLASLt scale type %d; assuming four bytes",
                static_cast<int>(t));
      return 4;
  }
}

cublasStatus_t from_cu(CUresult r) {
  return r == CUDA_SUCCESS ? CUBLAS_STATUS_SUCCESS
                           : CUBLAS_STATUS_EXECUTION_FAILED;
}

// The status comes back in the payload; the frame's result field is a CUresult.
cublasStatus_t send(uint32_t id, rgpu::Buffer& req, rgpu::Buffer* rsp) {
  CUresult r = rgpu::call(id, req, rsp);
  if (r != CUDA_SUCCESS) return from_cu(r);
  int32_t status = CUBLAS_STATUS_INTERNAL_ERROR;
  if (!rsp->get(&status)) return CUBLAS_STATUS_INTERNAL_ERROR;
  return static_cast<cublasStatus_t>(status);
}

void put_ptr(rgpu::Buffer& b, const void* p) {
  b.put<uint64_t>(reinterpret_cast<uint64_t>(p));
}

// A create call: no inputs beyond what the caller passed, one handle back.
cublasStatus_t recv_handle(rgpu::Buffer& rsp, cublasStatus_t s, void** out) {
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  uint64_t h = 0;
  if (!rsp.get(&h)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *out = reinterpret_cast<void*>(h);
  return s;
}

// Shared shape of the three SetAttribute calls.
cublasStatus_t set_attribute(uint32_t id, const void* obj, int32_t attr,
                             const void* buf, size_t size) {
  rgpu::Buffer req, rsp;
  put_ptr(req, obj);
  req.put<int32_t>(attr);
  req.put<uint8_t>(buf ? 1 : 0);
  if (buf) req.put_sized(buf, size);
  return send(id, req, &rsp);
}

cublasStatus_t get_attribute(uint32_t id, const void* obj, int32_t attr,
                             void* buf, size_t size, size_t* written) {
  rgpu::Buffer req, rsp;
  put_ptr(req, obj);
  req.put<int32_t>(attr);
  req.put<uint64_t>(size);
  cublasStatus_t s = send(id, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  const uint8_t* bytes = nullptr;
  size_t n = 0;
  if (!rsp.get_sized(&bytes, &n)) return CUBLAS_STATUS_INTERNAL_ERROR;
  if (buf && n) std::memcpy(buf, bytes, n < size ? n : size);
  if (written) *written = n;
  return s;
}

}  // namespace

extern "C" {

cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle) {
  if (!lightHandle) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  cublasStatus_t s = send(rgpu::API_cublasLtCreate, req, &rsp);
  return recv_handle(rsp, s, reinterpret_cast<void**>(lightHandle));
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
  rgpu::Buffer req, rsp;
  put_ptr(req, lightHandle);
  return send(rgpu::API_cublasLtDestroy, req, &rsp);
}

size_t cublasLtGetVersion(void) {
  rgpu::Buffer req, rsp;
  if (send(rgpu::API_cublasLtGetVersion, req, &rsp) != CUBLAS_STATUS_SUCCESS) {
    return 0;
  }
  uint64_t v = 0;
  return rsp.get(&v) ? static_cast<size_t>(v) : 0;
}

size_t cublasLtGetCudartVersion(void) {
  rgpu::Buffer req, rsp;
  if (send(rgpu::API_cublasLtGetCudartVersion, req, &rsp) !=
      CUBLAS_STATUS_SUCCESS) {
    return 0;
  }
  uint64_t v = 0;
  return rsp.get(&v) ? static_cast<size_t>(v) : 0;
}

cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  req.put<int32_t>(static_cast<int32_t>(type));
  cublasStatus_t s = send(rgpu::API_cublasLtGetProperty, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  int32_t v = 0;
  if (!rsp.get(&v)) return CUBLAS_STATUS_INTERNAL_ERROR;
  *value = v;
  return s;
}

// --- matmul descriptors ----------------------------------------------------

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                        cublasComputeType_t computeType,
                                        cudaDataType_t scaleType) {
  if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  req.put<int32_t>(static_cast<int32_t>(computeType));
  req.put<int32_t>(static_cast<int32_t>(scaleType));
  cublasStatus_t s = send(rgpu::API_cublasLtMatmulDescCreate, req, &rsp);
  s = recv_handle(rsp, s, reinterpret_cast<void**>(matmulDesc));
  if (s == CUBLAS_STATUS_SUCCESS) {
    // The scale type fixes how wide alpha and beta are at the matmul.
    std::lock_guard<std::mutex> lk(g_desc_mu);
    g_descs[*matmulDesc] = DescState{scaleType, false};
  }
  return s;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
  rgpu::Buffer req, rsp;
  put_ptr(req, matmulDesc);
  cublasStatus_t s = send(rgpu::API_cublasLtMatmulDescDestroy, req, &rsp);
  std::lock_guard<std::mutex> lk(g_desc_mu);
  g_descs.erase(matmulDesc);
  return s;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
    const void* buf, size_t sizeInBytes) {
  // Two attributes change how a later matmul has to be marshalled, so note
  // them on the way past.
  if (buf) {
    std::lock_guard<std::mutex> lk(g_desc_mu);
    DescState& st = g_descs[matmulDesc];
    if (attr == CUBLASLT_MATMUL_DESC_POINTER_MODE &&
        sizeInBytes >= sizeof(int32_t)) {
      int32_t mode = 0;
      std::memcpy(&mode, buf, sizeof(mode));
      st.device_pointers = (mode != CUBLASLT_POINTER_MODE_HOST);
    } else if (attr == CUBLASLT_MATMUL_DESC_SCALE_TYPE &&
               sizeInBytes >= sizeof(int32_t)) {
      int32_t t = 0;
      std::memcpy(&t, buf, sizeof(t));
      st.scale_type = static_cast<cudaDataType_t>(t);
    }
  }
  return set_attribute(rgpu::API_cublasLtMatmulDescSetAttribute, matmulDesc,
                       static_cast<int32_t>(attr), buf, sizeInBytes);
}

cublasStatus_t cublasLtMatmulDescGetAttribute(
    cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten) {
  return get_attribute(rgpu::API_cublasLtMatmulDescGetAttribute, matmulDesc,
                       static_cast<int32_t>(attr), buf, sizeInBytes,
                       sizeWritten);
}

// --- matrix layouts --------------------------------------------------------

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout,
                                          cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
  if (!matLayout) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  req.put<int32_t>(static_cast<int32_t>(type));
  req.put<uint64_t>(rows);
  req.put<uint64_t>(cols);
  req.put<int64_t>(ld);
  cublasStatus_t s = send(rgpu::API_cublasLtMatrixLayoutCreate, req, &rsp);
  return recv_handle(rsp, s, reinterpret_cast<void**>(matLayout));
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
  rgpu::Buffer req, rsp;
  put_ptr(req, matLayout);
  return send(rgpu::API_cublasLtMatrixLayoutDestroy, req, &rsp);
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
    const void* buf, size_t sizeInBytes) {
  return set_attribute(rgpu::API_cublasLtMatrixLayoutSetAttribute, matLayout,
                       static_cast<int32_t>(attr), buf, sizeInBytes);
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten) {
  return get_attribute(rgpu::API_cublasLtMatrixLayoutGetAttribute, matLayout,
                       static_cast<int32_t>(attr), buf, sizeInBytes,
                       sizeWritten);
}

// --- preferences -----------------------------------------------------------

cublasStatus_t cublasLtMatmulPreferenceCreate(
    cublasLtMatmulPreference_t* pref) {
  if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
  rgpu::Buffer req, rsp;
  cublasStatus_t s = send(rgpu::API_cublasLtMatmulPreferenceCreate, req, &rsp);
  return recv_handle(rsp, s, reinterpret_cast<void**>(pref));
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(
    cublasLtMatmulPreference_t pref) {
  rgpu::Buffer req, rsp;
  put_ptr(req, pref);
  return send(rgpu::API_cublasLtMatmulPreferenceDestroy, req, &rsp);
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
    const void* buf, size_t sizeInBytes) {
  return set_attribute(rgpu::API_cublasLtMatmulPreferenceSetAttribute, pref,
                       static_cast<int32_t>(attr), buf, sizeInBytes);
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten) {
  return get_attribute(rgpu::API_cublasLtMatmulPreferenceGetAttribute, pref,
                       static_cast<int32_t>(attr), buf, sizeInBytes,
                       sizeWritten);
}

// --- the parts that do the work -------------------------------------------

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int* returnAlgoCount) {
  if (!heuristicResultsArray || !returnAlgoCount || requestedAlgoCount <= 0) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  rgpu::Buffer req, rsp;
  put_ptr(req, lightHandle);
  put_ptr(req, operationDesc);
  put_ptr(req, Adesc);
  put_ptr(req, Bdesc);
  put_ptr(req, Cdesc);
  put_ptr(req, Ddesc);
  put_ptr(req, preference);
  req.put<int32_t>(requestedAlgoCount);
  cublasStatus_t s = send(rgpu::API_cublasLtMatmulAlgoGetHeuristic, req, &rsp);
  if (s != CUBLAS_STATUS_SUCCESS) return s;

  // The results are plain data, including the algo handle inside each one,
  // which is an opaque array of integers rather than a pointer. They travel
  // verbatim and go back to the server unchanged at the matmul.
  int32_t count = 0;
  if (!rsp.get(&count)) return CUBLAS_STATUS_INTERNAL_ERROR;
  if (count < 0 || count > requestedAlgoCount) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  const uint8_t* bytes = nullptr;
  size_t n = 0;
  if (!rsp.get_sized(&bytes, &n)) return CUBLAS_STATUS_INTERNAL_ERROR;
  if (n != static_cast<size_t>(count) * sizeof(heuristicResultsArray[0])) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (n) std::memcpy(heuristicResultsArray, bytes, n);
  *returnAlgoCount = count;
  return s;
}

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  const DescState st = desc_state(computeDesc);
  const size_t ss = scalar_size(st.scale_type);

  rgpu::Buffer req, rsp;
  put_ptr(req, lightHandle);
  put_ptr(req, computeDesc);

  // In host mode the values travel; in device mode the pointers do. The width
  // goes first, because the server needs it to read the values that follow.
  req.put<uint8_t>(st.device_pointers ? 0 : 1);
  req.put<uint64_t>(ss);
  req.put<uint8_t>(alpha ? 1 : 0);
  if (alpha) {
    if (st.device_pointers) put_ptr(req, alpha);
    else req.put_bytes(alpha, ss);
  }
  req.put<uint8_t>(beta ? 1 : 0);
  if (beta) {
    if (st.device_pointers) put_ptr(req, beta);
    else req.put_bytes(beta, ss);
  }

  put_ptr(req, A);
  put_ptr(req, Adesc);
  put_ptr(req, B);
  put_ptr(req, Bdesc);
  put_ptr(req, C);
  put_ptr(req, Cdesc);
  put_ptr(req, D);
  put_ptr(req, Ddesc);

  req.put<uint8_t>(algo ? 1 : 0);
  if (algo) req.put_bytes(algo, sizeof(*algo));

  put_ptr(req, workspace);
  req.put<uint64_t>(workspaceSizeInBytes);
  req.put<uint64_t>(reinterpret_cast<uint64_t>(stream));
  return send(rgpu::API_cublasLtMatmul, req, &rsp);
}

}  // extern "C"
