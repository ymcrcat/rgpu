// Executes forwarded cuBLASLt calls against the real library on the GPU host.
//
// Opened on first use rather than linked, so a host without cuBLASLt still
// runs everything else.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cublasLt.h>

#include "common/cublaslt_ids.h"
#include "common/wire.h"
#include "server/inventory.h"

namespace rgpu {
namespace {

void* lt_handle() {
  static void* h = [] {
    const char* path = std::getenv("RGPU_CUBLASLT");
    if (!path) path = "libcublasLt.so.12";
    void* lib = ::dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!lib) {
      std::fprintf(stderr,
                   "[rgpu-server] cannot open %s: %s\n"
                   "[rgpu-server] set RGPU_CUBLASLT to its path\n",
                   path, ::dlerror());
    }
    return lib;
  }();
  return h;
}

template <typename Fn>
Fn lt_sym(const char* name) {
  // Whatever is already in the process wins, which is how the fake library
  // in the test build gets used without opening anything.
  if (void* here = ::dlsym(RTLD_DEFAULT, name)) return reinterpret_cast<Fn>(here);
  void* lib = lt_handle();
  if (!lib) return nullptr;
  void* fn = ::dlsym(lib, name);
  if (!fn) std::fprintf(stderr, "[rgpu-server] cuBLASLt has no %s\n", name);
  return reinterpret_cast<Fn>(fn);
}

// How a handle is given back when the session that made it never comes back.
CUresult destroy_handle(uint64_t h) {
  auto fn = lt_sym<cublasStatus_t (*)(cublasLtHandle_t)>("cublasLtDestroy");
  if (!fn) return CUDA_ERROR_NOT_SUPPORTED;
  return fn(reinterpret_cast<cublasLtHandle_t>(h)) == CUBLAS_STATUS_SUCCESS
             ? CUDA_SUCCESS
             : CUDA_ERROR_UNKNOWN;
}

void put_status(Buffer* rsp, cublasStatus_t s) {
  rsp->put<int32_t>(static_cast<int32_t>(s));
}

void* get_ptr(Buffer& req, bool* ok) {
  uint64_t v = 0;
  if (!req.get(&v)) {
    *ok = false;
    return nullptr;
  }
  return reinterpret_cast<void*>(v);
}

// The three SetAttribute calls share a shape, as do the three getters, so they
// are handled once each rather than six times.
bool handle_set_attribute(uint32_t id, Buffer& req, Buffer* rsp) {
  const char* name = nullptr;
  switch (id) {
    case API_cublasLtMatmulDescSetAttribute:
      name = "cublasLtMatmulDescSetAttribute"; break;
    case API_cublasLtMatrixLayoutSetAttribute:
      name = "cublasLtMatrixLayoutSetAttribute"; break;
    case API_cublasLtMatmulPreferenceSetAttribute:
      name = "cublasLtMatmulPreferenceSetAttribute"; break;
    default: return false;
  }
  auto fn = lt_sym<cublasStatus_t (*)(void*, int, const void*, size_t)>(name);
  bool ok = true;
  void* obj = get_ptr(req, &ok);
  int32_t attr = 0;
  uint8_t present = 0;
  if (!fn || !ok || !req.get(&attr) || !req.get(&present)) {
    put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
    return true;
  }
  const uint8_t* bytes = nullptr;
  size_t n = 0;
  if (present && !req.get_sized(&bytes, &n)) {
    put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
    return true;
  }
  put_status(rsp, fn(obj, attr, present ? bytes : nullptr, n));
  return true;
}

bool handle_get_attribute(uint32_t id, Buffer& req, Buffer* rsp) {
  const char* name = nullptr;
  switch (id) {
    case API_cublasLtMatmulDescGetAttribute:
      name = "cublasLtMatmulDescGetAttribute"; break;
    case API_cublasLtMatrixLayoutGetAttribute:
      name = "cublasLtMatrixLayoutGetAttribute"; break;
    case API_cublasLtMatmulPreferenceGetAttribute:
      name = "cublasLtMatmulPreferenceGetAttribute"; break;
    default: return false;
  }
  auto fn = lt_sym<cublasStatus_t (*)(void*, int, void*, size_t, size_t*)>(name);
  bool ok = true;
  void* obj = get_ptr(req, &ok);
  int32_t attr = 0;
  uint64_t size = 0;
  if (!fn || !ok || !req.get(&attr) || !req.get(&size)) {
    put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
    return true;
  }
  // Bounded so a bad size cannot make the server allocate wildly.
  if (size > (1u << 20)) {
    put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
    return true;
  }
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  size_t written = 0;
  cublasStatus_t s = fn(obj, attr, buf.empty() ? nullptr : buf.data(),
                        buf.size(), &written);
  put_status(rsp, s);
  if (s == CUBLAS_STATUS_SUCCESS) {
    if (written > buf.size()) written = buf.size();
    rsp->put_sized(buf.data(), written);
  }
  return true;
}

}  // namespace

bool dispatch_cublaslt(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out) {
  if (id < kCublasLtBase || id > kCublasLtBase + 1000) return false;
  *out = CUDA_SUCCESS;
  bool ok = true;

  switch (id) {
    case API_cublasLtCreate: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtHandle_t*)>("cublasLtCreate");
      if (!fn) { put_status(rsp, CUBLAS_STATUS_NOT_INITIALIZED); return true; }
      cublasLtHandle_t h = nullptr;
      cublasStatus_t s = fn(&h);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(h));
        inventory_note_handle(reinterpret_cast<uint64_t>(h), "cuBLASLt",
                              &destroy_handle);
      }
      return true;
    }
    case API_cublasLtDestroy: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtHandle_t)>("cublasLtDestroy");
      auto h = static_cast<cublasLtHandle_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      cublasStatus_t s = fn(h);
      if (s == CUBLAS_STATUS_SUCCESS) {
        inventory_forget_handle(reinterpret_cast<uint64_t>(h));
      }
      put_status(rsp, s);
      return true;
    }
    case API_cublasLtGetVersion:
    case API_cublasLtGetCudartVersion: {
      const char* name = id == API_cublasLtGetVersion
                             ? "cublasLtGetVersion"
                             : "cublasLtGetCudartVersion";
      auto fn = lt_sym<size_t (*)(void)>(name);
      if (!fn) { put_status(rsp, CUBLAS_STATUS_NOT_INITIALIZED); return true; }
      put_status(rsp, CUBLAS_STATUS_SUCCESS);
      rsp->put<uint64_t>(static_cast<uint64_t>(fn()));
      return true;
    }
    case API_cublasLtGetProperty: {
      auto fn = lt_sym<cublasStatus_t (*)(libraryPropertyType, int*)>(
          "cublasLtGetProperty");
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

    case API_cublasLtMatmulDescCreate: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtMatmulDesc_t*,
                                          cublasComputeType_t, cudaDataType_t)>(
          "cublasLtMatmulDescCreate");
      int32_t ct = 0, st = 0;
      if (!fn || !req.get(&ct) || !req.get(&st)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      cublasLtMatmulDesc_t d = nullptr;
      cublasStatus_t s = fn(&d, static_cast<cublasComputeType_t>(ct),
                            static_cast<cudaDataType_t>(st));
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(d));
      }
      return true;
    }
    case API_cublasLtMatmulDescDestroy: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtMatmulDesc_t)>(
          "cublasLtMatmulDescDestroy");
      auto d = static_cast<cublasLtMatmulDesc_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(d));
      return true;
    }

    case API_cublasLtMatrixLayoutCreate: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtMatrixLayout_t*, cudaDataType,
                                          uint64_t, uint64_t, int64_t)>(
          "cublasLtMatrixLayoutCreate");
      int32_t type = 0;
      uint64_t rows = 0, cols = 0;
      int64_t ld = 0;
      if (!fn || !req.get(&type) || !req.get(&rows) || !req.get(&cols) ||
          !req.get(&ld)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      cublasLtMatrixLayout_t l = nullptr;
      cublasStatus_t s = fn(&l, static_cast<cudaDataType>(type), rows, cols, ld);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(l));
      }
      return true;
    }
    case API_cublasLtMatrixLayoutDestroy: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtMatrixLayout_t)>(
          "cublasLtMatrixLayoutDestroy");
      auto l = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(l));
      return true;
    }

    case API_cublasLtMatmulPreferenceCreate: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtMatmulPreference_t*)>(
          "cublasLtMatmulPreferenceCreate");
      if (!fn) { put_status(rsp, CUBLAS_STATUS_NOT_INITIALIZED); return true; }
      cublasLtMatmulPreference_t p = nullptr;
      cublasStatus_t s = fn(&p);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(p));
      }
      return true;
    }
    case API_cublasLtMatmulPreferenceDestroy: {
      auto fn = lt_sym<cublasStatus_t (*)(cublasLtMatmulPreference_t)>(
          "cublasLtMatmulPreferenceDestroy");
      auto p = static_cast<cublasLtMatmulPreference_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUBLAS_STATUS_INVALID_VALUE); return true; }
      put_status(rsp, fn(p));
      return true;
    }

    case API_cublasLtMatmulDescSetAttribute:
    case API_cublasLtMatrixLayoutSetAttribute:
    case API_cublasLtMatmulPreferenceSetAttribute:
      return handle_set_attribute(id, req, rsp);

    case API_cublasLtMatmulDescGetAttribute:
    case API_cublasLtMatrixLayoutGetAttribute:
    case API_cublasLtMatmulPreferenceGetAttribute:
      return handle_get_attribute(id, req, rsp);

    case API_cublasLtMatmulAlgoGetHeuristic: {
      auto fn = lt_sym<cublasStatus_t (*)(
          cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
          cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
          cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int,
          cublasLtMatmulHeuristicResult_t*, int*)>(
          "cublasLtMatmulAlgoGetHeuristic");
      auto h = static_cast<cublasLtHandle_t>(get_ptr(req, &ok));
      auto op = static_cast<cublasLtMatmulDesc_t>(get_ptr(req, &ok));
      auto a = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      auto b = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      auto c = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      auto d = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      auto pref = static_cast<cublasLtMatmulPreference_t>(get_ptr(req, &ok));
      int32_t want = 0;
      if (!fn || !ok || !req.get(&want) || want <= 0 || want > 256) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      std::vector<cublasLtMatmulHeuristicResult_t> results(
          static_cast<size_t>(want));
      int got = 0;
      cublasStatus_t s = fn(h, op, a, b, c, d, pref, want, results.data(), &got);
      put_status(rsp, s);
      if (s == CUBLAS_STATUS_SUCCESS) {
        if (got < 0) got = 0;
        if (got > want) got = want;
        rsp->put<int32_t>(got);
        // Plain data, including each algo, which is an opaque array of
        // integers rather than a pointer, so it travels verbatim.
        rsp->put_sized(results.data(),
                       static_cast<size_t>(got) * sizeof(results[0]));
      }
      return true;
    }

    case API_cublasLtMatmul: {
      auto fn = lt_sym<cublasStatus_t (*)(
          cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*,
          cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
          const void*, const void*, cublasLtMatrixLayout_t, void*,
          cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t,
          cudaStream_t)>("cublasLtMatmul");
      auto h = static_cast<cublasLtHandle_t>(get_ptr(req, &ok));
      auto desc = static_cast<cublasLtMatmulDesc_t>(get_ptr(req, &ok));

      // Host flag and scalar width come before the scalars, because the width
      // is what says how many bytes each one occupies.
      uint8_t host = 0;
      uint64_t ss = 0;
      if (!fn || !ok || !req.get(&host) || !req.get(&ss) || ss > 64) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }

      // Host-mode scalars are copied out because the pointer must stay valid
      // for the duration of the call.
      std::vector<uint8_t> alpha_buf, beta_buf;
      auto read_scalar = [&](std::vector<uint8_t>* buf) -> const void* {
        uint8_t present = 0;
        if (!req.get(&present)) { ok = false; return nullptr; }
        if (!present) return nullptr;
        if (host) {
          const uint8_t* bytes = nullptr;
          if (!req.get_bytes(static_cast<size_t>(ss), &bytes)) {
            ok = false;
            return nullptr;
          }
          buf->assign(bytes, bytes + ss);
          return buf->data();
        }
        uint64_t p = 0;
        if (!req.get(&p)) { ok = false; return nullptr; }
        return reinterpret_cast<const void*>(p);
      };
      const void* alpha = read_scalar(&alpha_buf);
      const void* beta = read_scalar(&beta_buf);

      const void* A = get_ptr(req, &ok);
      auto Adesc = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      const void* B = get_ptr(req, &ok);
      auto Bdesc = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      const void* C = get_ptr(req, &ok);
      auto Cdesc = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));
      void* D = get_ptr(req, &ok);
      auto Ddesc = static_cast<cublasLtMatrixLayout_t>(get_ptr(req, &ok));

      uint8_t has_algo = 0;
      cublasLtMatmulAlgo_t algo{};
      if (!ok || !req.get(&has_algo)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }
      if (has_algo) {
        const uint8_t* bytes = nullptr;
        if (!req.get_bytes(sizeof(algo), &bytes)) {
          put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
          return true;
        }
        std::memcpy(&algo, bytes, sizeof(algo));
      }

      void* workspace = get_ptr(req, &ok);
      uint64_t ws_size = 0, stream = 0;
      if (!ok || !req.get(&ws_size) || !req.get(&stream)) {
        put_status(rsp, CUBLAS_STATUS_INVALID_VALUE);
        return true;
      }

      put_status(rsp, fn(h, desc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D,
                         Ddesc, has_algo ? &algo : nullptr, workspace,
                         static_cast<size_t>(ws_size),
                         reinterpret_cast<cudaStream_t>(stream)));
      return true;
    }

    default:
      std::fprintf(stderr, "[rgpu-server] unhandled cuBLASLt id %u\n", id);
      put_status(rsp, CUBLAS_STATUS_NOT_SUPPORTED);
      return true;
  }
}

}  // namespace rgpu
