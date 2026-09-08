// Executes forwarded cuBLAS calls against the real library on the GPU host.
//
// The library is opened on first use rather than linked, so a host without
// cuBLAS still runs everything else, and so the server does not have to be
// built against a particular cuBLAS.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cublas_v2.h>

#include "common/cublas_ids.h"
#include "common/wire.h"

namespace rgpu {
namespace {

void* cublas_handle() {
  static void* h = [] {
    const char* path = std::getenv("RGPU_CUBLAS");
    if (!path) path = "libcublas.so.12";
    void* lib = ::dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!lib) {
      std::fprintf(stderr,
                   "[rgpu-server] cannot open %s: %s\n"
                   "[rgpu-server] set RGPU_CUBLAS to its path; cuBLAS calls "
                   "will fail until then\n",
                   path, ::dlerror());
    }
    return lib;
  }();
  return h;
}

template <typename Fn>
Fn cublas_sym(const char* name) {
  void* lib = cublas_handle();
  if (!lib) return nullptr;
  void* fn = ::dlsym(lib, name);
  if (!fn) std::fprintf(stderr, "[rgpu-server] cuBLAS has no %s\n", name);
  return reinterpret_cast<Fn>(fn);
}

// Every handler writes the cuBLAS status into the reply payload; the frame's
// own result field carries a CUresult and stays CUDA_SUCCESS.
void put_status(Buffer* rsp, cublasStatus_t s) {
  rsp->put<int32_t>(static_cast<int32_t>(s));
}

cublasHandle_t get_handle(Buffer& req, bool* ok) {
  uint64_t v = 0;
  *ok = req.get(&v);
  return reinterpret_cast<cublasHandle_t>(v);
}

void* get_devptr(Buffer& req, bool* ok) {
  uint64_t v = 0;
  if (!req.get(&v)) {
    *ok = false;
    return nullptr;
  }
  return reinterpret_cast<void*>(v);
}

// The mirror of the client's put_scalar. In host pointer mode the value came
// with the call and is copied into `storage`; in device mode a pointer came
// instead and is passed straight through.
const void* get_scalar(Buffer& req, size_t size, std::vector<uint8_t>* storage,
                       bool* ok) {
  uint8_t host = 0, present = 0;
  if (!req.get(&host) || !req.get(&present)) {
    *ok = false;
    return nullptr;
  }
  if (!present) return nullptr;
  if (host) {
    const uint8_t* bytes = nullptr;
    if (!req.get_bytes(size, &bytes)) {
      *ok = false;
      return nullptr;
    }
    storage->assign(bytes, bytes + size);
    return storage->data();
  }
  uint64_t p = 0;
  if (!req.get(&p)) {
    *ok = false;
    return nullptr;
  }
  return reinterpret_cast<const void*>(p);
}

}  // namespace

bool dispatch_cublas(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out) {
  if (id < kCublasBase || id > kCublasBase + 1000) return false;
  *out = CUDA_SUCCESS;
  bool ok = true;

  switch (id) {
    case API_cublasCreate: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t*)>(
          "cublasCreate_v2");
      if (!fn) { put_status(rsp, CUBLAS_STATUS_NOT_INITIALIZED); return true; }
      cublasHandle_t h = nullptr;
      cublasStatus_t s = fn(&h);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(h));
      }
      return true;
    }
    case API_cublasDestroy: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t)>(
          "cublasDestroy_v2");
      cublasHandle_t h = get_handle(req, &ok);
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(h));
      return true;
    }
    case API_cublasSetStream: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, cudaStream_t)>(
          "cublasSetStream_v2");
      cublasHandle_t h = get_handle(req, &ok);
      uint64_t st = 0;
      if (!fn || !ok || !req.get(&st)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      put_status(rsp, fn(h, reinterpret_cast<cudaStream_t>(st)));
      return true;
    }
    case API_cublasGetStream: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, cudaStream_t*)>(
          "cublasGetStream_v2");
      cublasHandle_t h = get_handle(req, &ok);
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      cudaStream_t st = nullptr;
      cublasStatus_t s = fn(h, &st);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(st));
      }
      return true;
    }
    case API_cublasSetPointerMode: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t,
                                              cublasPointerMode_t)>(
          "cublasSetPointerMode_v2");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t mode = 0;
      if (!fn || !ok || !req.get(&mode)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      put_status(rsp, fn(h, static_cast<cublasPointerMode_t>(mode)));
      return true;
    }
    case API_cublasSetMathMode: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, cublasMath_t)>(
          "cublasSetMathMode");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t mode = 0;
      if (!fn || !ok || !req.get(&mode)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      put_status(rsp, fn(h, static_cast<cublasMath_t>(mode)));
      return true;
    }
    case API_cublasGetMathMode: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, cublasMath_t*)>(
          "cublasGetMathMode");
      cublasHandle_t h = get_handle(req, &ok);
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      cublasMath_t m = CUBLAS_DEFAULT_MATH;
      cublasStatus_t s = fn(h, &m);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) rsp->put<int32_t>(static_cast<int32_t>(m));
      return true;
    }
    case API_cublasSetWorkspace: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, void*, size_t)>(
          "cublasSetWorkspace_v2");
      cublasHandle_t h = get_handle(req, &ok);
      void* ws = get_devptr(req, &ok);
      uint64_t size = 0;
      if (!fn || !ok || !req.get(&size)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      put_status(rsp, fn(h, ws, static_cast<size_t>(size)));
      return true;
    }
    case API_cublasGetVersion: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, int*)>(
          "cublasGetVersion_v2");
      cublasHandle_t h = get_handle(req, &ok);
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      int v = 0;
      cublasStatus_t s = fn(h, &v);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) rsp->put<int32_t>(v);
      return true;
    }
    case API_cublasGetProperty: {
      auto fn = cublas_sym<cublasStatus_t (*)(libraryPropertyType, int*)>(
          "cublasGetProperty");
      int32_t type = 0;
      if (!fn || !req.get(&type)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      int v = 0;
      cublasStatus_t s = fn(static_cast<libraryPropertyType>(type), &v);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) rsp->put<int32_t>(v);
      return true;
    }
    case API_cublasSetSmCountTarget: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, int)>(
          "cublasSetSmCountTarget");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t v = 0;
      if (!fn || !ok || !req.get(&v)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      put_status(rsp, fn(h, v));
      return true;
    }
    case API_cublasGetSmCountTarget: {
      auto fn = cublas_sym<cublasStatus_t (*)(cublasHandle_t, int*)>(
          "cublasGetSmCountTarget");
      cublasHandle_t h = get_handle(req, &ok);
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      int v = 0;
      cublasStatus_t s = fn(h, &v);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) rsp->put<int32_t>(v);
      return true;
    }

    case API_cublasSgemm: {
      auto fn = cublas_sym<cublasStatus_t (*)(
          cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
          const float*, const float*, int, const float*, int, const float*,
          float*, int)>("cublasSgemm_v2");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t ta, tb, m, n, k, lda, ldb, ldc;
      std::vector<uint8_t> as, bs;
      if (!fn || !ok || !req.get(&ta) || !req.get(&tb) || !req.get(&m) ||
          !req.get(&n) || !req.get(&k)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      const void* alpha = get_scalar(req, sizeof(float), &as, &ok);
      const void* A = get_devptr(req, &ok);
      ok = ok && req.get(&lda);
      const void* B = get_devptr(req, &ok);
      ok = ok && req.get(&ldb);
      const void* beta = get_scalar(req, sizeof(float), &bs, &ok);
      void* C = get_devptr(req, &ok);
      ok = ok && req.get(&ldc);
      if (!ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(h, static_cast<cublasOperation_t>(ta),
                         static_cast<cublasOperation_t>(tb), m, n, k,
                         static_cast<const float*>(alpha),
                         static_cast<const float*>(A), lda,
                         static_cast<const float*>(B), ldb,
                         static_cast<const float*>(beta),
                         static_cast<float*>(C), ldc));
      return true;
    }
    case API_cublasDgemm: {
      auto fn = cublas_sym<cublasStatus_t (*)(
          cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
          const double*, const double*, int, const double*, int, const double*,
          double*, int)>("cublasDgemm_v2");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t ta, tb, m, n, k, lda, ldb, ldc;
      std::vector<uint8_t> as, bs;
      if (!fn || !ok || !req.get(&ta) || !req.get(&tb) || !req.get(&m) ||
          !req.get(&n) || !req.get(&k)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      const void* alpha = get_scalar(req, sizeof(double), &as, &ok);
      const void* A = get_devptr(req, &ok);
      ok = ok && req.get(&lda);
      const void* B = get_devptr(req, &ok);
      ok = ok && req.get(&ldb);
      const void* beta = get_scalar(req, sizeof(double), &bs, &ok);
      void* C = get_devptr(req, &ok);
      ok = ok && req.get(&ldc);
      if (!ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(h, static_cast<cublasOperation_t>(ta),
                         static_cast<cublasOperation_t>(tb), m, n, k,
                         static_cast<const double*>(alpha),
                         static_cast<const double*>(A), lda,
                         static_cast<const double*>(B), ldb,
                         static_cast<const double*>(beta),
                         static_cast<double*>(C), ldc));
      return true;
    }
    case API_cublasGemmEx: {
      auto fn = cublas_sym<cublasStatus_t (*)(
          cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
          const void*, const void*, cudaDataType, int, const void*,
          cudaDataType, int, const void*, void*, cudaDataType, int,
          cublasComputeType_t, cublasGemmAlgo_t)>("cublasGemmEx");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t ta, tb, m, n, k, ct, algo, at, lda, bt, ldb, cty, ldc;
      std::vector<uint8_t> as, bs;
      if (!fn || !ok || !req.get(&ta) || !req.get(&tb) || !req.get(&m) ||
          !req.get(&n) || !req.get(&k) || !req.get(&ct) || !req.get(&algo)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      const size_t ss =
          (ct == CUBLAS_COMPUTE_64F || ct == CUBLAS_COMPUTE_64F_PEDANTIC)
              ? sizeof(double)
              : sizeof(float);
      const void* alpha = get_scalar(req, ss, &as, &ok);
      const void* A = get_devptr(req, &ok);
      ok = ok && req.get(&at) && req.get(&lda);
      const void* B = get_devptr(req, &ok);
      ok = ok && req.get(&bt) && req.get(&ldb);
      const void* beta = get_scalar(req, ss, &bs, &ok);
      void* C = get_devptr(req, &ok);
      ok = ok && req.get(&cty) && req.get(&ldc);
      if (!ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(h, static_cast<cublasOperation_t>(ta),
                         static_cast<cublasOperation_t>(tb), m, n, k, alpha, A,
                         static_cast<cudaDataType>(at), lda, B,
                         static_cast<cudaDataType>(bt), ldb, beta, C,
                         static_cast<cudaDataType>(cty), ldc,
                         static_cast<cublasComputeType_t>(ct),
                         static_cast<cublasGemmAlgo_t>(algo)));
      return true;
    }
    case API_cublasGemmStridedBatchedEx: {
      auto fn = cublas_sym<cublasStatus_t (*)(
          cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
          const void*, const void*, cudaDataType, int, long long int,
          const void*, cudaDataType, int, long long int, const void*, void*,
          cudaDataType, int, long long int, int, cublasComputeType_t,
          cublasGemmAlgo_t)>("cublasGemmStridedBatchedEx");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t ta, tb, m, n, k, ct, algo, batch, at, lda, bt, ldb, cty, ldc;
      int64_t sa, sb, sc;
      std::vector<uint8_t> as, bs;
      if (!fn || !ok || !req.get(&ta) || !req.get(&tb) || !req.get(&m) ||
          !req.get(&n) || !req.get(&k) || !req.get(&ct) || !req.get(&algo) ||
          !req.get(&batch)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      const size_t ss =
          (ct == CUBLAS_COMPUTE_64F || ct == CUBLAS_COMPUTE_64F_PEDANTIC)
              ? sizeof(double)
              : sizeof(float);
      const void* alpha = get_scalar(req, ss, &as, &ok);
      const void* A = get_devptr(req, &ok);
      ok = ok && req.get(&at) && req.get(&lda) && req.get(&sa);
      const void* B = get_devptr(req, &ok);
      ok = ok && req.get(&bt) && req.get(&ldb) && req.get(&sb);
      const void* beta = get_scalar(req, ss, &bs, &ok);
      void* C = get_devptr(req, &ok);
      ok = ok && req.get(&cty) && req.get(&ldc) && req.get(&sc);
      if (!ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(h, static_cast<cublasOperation_t>(ta),
                         static_cast<cublasOperation_t>(tb), m, n, k, alpha, A,
                         static_cast<cudaDataType>(at), lda, sa, B,
                         static_cast<cudaDataType>(bt), ldb, sb, beta, C,
                         static_cast<cudaDataType>(cty), ldc, sc, batch,
                         static_cast<cublasComputeType_t>(ct),
                         static_cast<cublasGemmAlgo_t>(algo)));
      return true;
    }
    case API_cublasSgemmStridedBatched: {
      auto fn = cublas_sym<cublasStatus_t (*)(
          cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
          const float*, const float*, int, long long int, const float*, int,
          long long int, const float*, float*, int, long long int, int)>(
          "cublasSgemmStridedBatched");
      cublasHandle_t h = get_handle(req, &ok);
      int32_t ta, tb, m, n, k, batch, lda, ldb, ldc;
      int64_t sa, sb, sc;
      std::vector<uint8_t> as, bs;
      if (!fn || !ok || !req.get(&ta) || !req.get(&tb) || !req.get(&m) ||
          !req.get(&n) || !req.get(&k) || !req.get(&batch)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      const void* alpha = get_scalar(req, sizeof(float), &as, &ok);
      const void* A = get_devptr(req, &ok);
      ok = ok && req.get(&lda) && req.get(&sa);
      const void* B = get_devptr(req, &ok);
      ok = ok && req.get(&ldb) && req.get(&sb);
      const void* beta = get_scalar(req, sizeof(float), &bs, &ok);
      void* C = get_devptr(req, &ok);
      ok = ok && req.get(&ldc) && req.get(&sc);
      if (!ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(h, static_cast<cublasOperation_t>(ta),
                         static_cast<cublasOperation_t>(tb), m, n, k,
                         static_cast<const float*>(alpha),
                         static_cast<const float*>(A), lda, sa,
                         static_cast<const float*>(B), ldb, sb,
                         static_cast<const float*>(beta),
                         static_cast<float*>(C), ldc, sc, batch));
      return true;
    }

    default:
      std::fprintf(stderr, "[rgpu-server] unhandled cuBLAS id %u\n", id);
      put_status(rsp, CUBLAS_STATUS_NOT_SUPPORTED);
      return true;
  }
}

}  // namespace rgpu
