// Executes forwarded cuDNN calls against the real library on the GPU host.
//
// Opened on first use rather than linked, so a host without cuDNN still runs
// everything else.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

#include "common/cudnn_ids.h"
#include "common/wire.h"

namespace rgpu {
namespace {

void* cudnn_handle() {
  static void* h = [] {
    const char* path = std::getenv("RGPU_CUDNN");
    if (!path) path = "libcudnn.so.9";
    void* lib = ::dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!lib) {
      std::fprintf(stderr,
                   "[rgpu-server] cannot open %s: %s\n"
                   "[rgpu-server] set RGPU_CUDNN to its path\n",
                   path, ::dlerror());
    }
    return lib;
  }();
  return h;
}

template <typename Fn>
Fn cudnn_sym(const char* name) {
  // Whatever is already in the process wins, which is how the fake library in
  // the test build gets used without opening anything.
  if (void* here = ::dlsym(RTLD_DEFAULT, name)) return reinterpret_cast<Fn>(here);
  void* lib = cudnn_handle();
  if (!lib) return nullptr;
  void* fn = ::dlsym(lib, name);
  if (!fn) std::fprintf(stderr, "[rgpu-server] cuDNN has no %s\n", name);
  return reinterpret_cast<Fn>(fn);
}

void put_status(Buffer* rsp, cudnnStatus_t s) {
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

// Bounds an attribute array so a bad count cannot make the server allocate
// wildly. cuDNN's real arrays are small; this is far above anything genuine.
constexpr int64_t kMaxElements = 1 << 20;

}  // namespace

bool dispatch_cudnn(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out) {
  if (id < kCudnnBase || id > kCudnnBase + 1000) return false;
  *out = CUDA_SUCCESS;
  bool ok = true;

  switch (id) {
    case API_cudnnCreate: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t*)>("cudnnCreate");
      if (!fn) { put_status(rsp, CUDNN_STATUS_NOT_INITIALIZED); return true; }
      cudnnHandle_t h = nullptr;
      cudnnStatus_t s = fn(&h);
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(h));
      }
      return true;
    }
    case API_cudnnDestroy: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t)>("cudnnDestroy");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, fn(h));
      return true;
    }
    case API_cudnnSetStream: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t)>(
          "cudnnSetStream");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      auto st = static_cast<cudaStream_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, fn(h, st));
      return true;
    }
    case API_cudnnGetStream: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t*)>(
          "cudnnGetStream");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUDNN_STATUS_BAD_PARAM); return true; }
      cudaStream_t st = nullptr;
      cudnnStatus_t s = fn(h, &st);
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(st));
      }
      return true;
    }
    case API_cudnnGetVersion:
    case API_cudnnGetCudartVersion: {
      const char* name = id == API_cudnnGetVersion ? "cudnnGetVersion"
                                                   : "cudnnGetCudartVersion";
      auto fn = cudnn_sym<size_t (*)(void)>(name);
      if (!fn) { put_status(rsp, CUDNN_STATUS_NOT_INITIALIZED); return true; }
      put_status(rsp, CUDNN_STATUS_SUCCESS);
      rsp->put<uint64_t>(static_cast<uint64_t>(fn()));
      return true;
    }
    case API_cudnnGetLastErrorString: {
      auto fn = cudnn_sym<void (*)(char*, size_t)>("cudnnGetLastErrorString");
      uint64_t max_size = 0;
      if (!fn || !req.get(&max_size)) {
        put_status(rsp, CUDNN_STATUS_NOT_SUPPORTED);
        return true;
      }
      // The caller's buffer can be any size it likes; this one only has to be
      // big enough for a message.
      if (max_size > 4096) max_size = 4096;
      std::vector<char> buf(max_size ? max_size : 1, '\0');
      fn(buf.data(), buf.size());
      buf.back() = '\0';
      put_status(rsp, CUDNN_STATUS_SUCCESS);
      rsp->put_sized(buf.data(), std::strlen(buf.data()));
      return true;
    }
    case API_cudnnGetMaxDeviceVersion: {
      auto fn = cudnn_sym<size_t (*)(void)>("cudnnGetMaxDeviceVersion");
      if (!fn) { put_status(rsp, CUDNN_STATUS_NOT_SUPPORTED); return true; }
      put_status(rsp, CUDNN_STATUS_SUCCESS);
      rsp->put<uint64_t>(static_cast<uint64_t>(fn()));
      return true;
    }
    case API_cudnnGetProperty: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(libraryPropertyType, int*)>(
          "cudnnGetProperty");
      int32_t type = 0;
      if (!fn || !req.get(&type)) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      int v = 0;
      cudnnStatus_t s = fn(static_cast<libraryPropertyType>(type), &v);
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) rsp->put<int32_t>(v);
      return true;
    }

    case API_cudnnBackendCreateDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnBackendDescriptorType_t,
                                            cudnnBackendDescriptor_t*)>(
          "cudnnBackendCreateDescriptor");
      int32_t type = 0;
      if (!fn || !req.get(&type)) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnBackendDescriptor_t d = nullptr;
      cudnnStatus_t s =
          fn(static_cast<cudnnBackendDescriptorType_t>(type), &d);
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(d));
      }
      return true;
    }
    case API_cudnnBackendDestroyDescriptor:
    case API_cudnnBackendInitialize:
    case API_cudnnBackendFinalize: {
      const char* name =
          id == API_cudnnBackendDestroyDescriptor
              ? "cudnnBackendDestroyDescriptor"
              : (id == API_cudnnBackendInitialize ? "cudnnBackendInitialize"
                                                  : "cudnnBackendFinalize");
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnBackendDescriptor_t)>(name);
      auto d = static_cast<cudnnBackendDescriptor_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, fn(d));
      return true;
    }
    case API_cudnnBackendSetAttribute: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnBackendDescriptor_t,
                                            cudnnBackendAttributeName_t,
                                            cudnnBackendAttributeType_t,
                                            int64_t, const void*)>(
          "cudnnBackendSetAttribute");
      auto d = static_cast<cudnnBackendDescriptor_t>(get_ptr(req, &ok));
      int32_t name = 0, type = 0;
      int64_t count = 0;
      uint8_t present = 0;
      if (!fn || !ok || !req.get(&name) || !req.get(&type) ||
          !req.get(&count) || !req.get(&present) || count < 0 ||
          count > kMaxElements) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      const uint8_t* bytes = nullptr;
      size_t n = 0;
      if (present && !req.get_sized(&bytes, &n)) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, fn(d, static_cast<cudnnBackendAttributeName_t>(name),
                         static_cast<cudnnBackendAttributeType_t>(type), count,
                         present ? bytes : nullptr));
      return true;
    }
    case API_cudnnBackendGetAttribute: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnBackendDescriptor_t,
                                            cudnnBackendAttributeName_t,
                                            cudnnBackendAttributeType_t,
                                            int64_t, int64_t*, void*)>(
          "cudnnBackendGetAttribute");
      auto d = static_cast<cudnnBackendDescriptor_t>(get_ptr(req, &ok));
      int32_t name = 0, type = 0;
      int64_t requested = 0;
      uint8_t want_count = 0, has_array = 0;
      if (!fn || !ok || !req.get(&name) || !req.get(&type) ||
          !req.get(&requested) || !req.get(&want_count) ||
          !req.get(&has_array) || requested < 0 || requested > kMaxElements) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      // The caller's current contents came with the request, because for
      // descriptor arrays they are inputs that cuDNN fills in place.
      std::vector<uint8_t> buf;
      if (has_array) {
        const uint8_t* bytes = nullptr;
        size_t n = 0;
        if (!req.get_sized(&bytes, &n)) {
          put_status(rsp, CUDNN_STATUS_BAD_PARAM);
          return true;
        }
        buf.assign(bytes, bytes + n);
      }
      int64_t produced = 0;
      cudnnStatus_t s = fn(d, static_cast<cudnnBackendAttributeName_t>(name),
                           static_cast<cudnnBackendAttributeType_t>(type),
                           requested, want_count ? &produced : nullptr,
                           has_array && !buf.empty() ? buf.data() : nullptr);
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        if (want_count) rsp->put<int64_t>(produced);
        if (has_array) rsp->put_sized(buf.data(), buf.size());
      }
      return true;
    }
    case API_cudnnBackendExecute: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t,
                                            cudnnBackendDescriptor_t,
                                            cudnnBackendDescriptor_t)>(
          "cudnnBackendExecute");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      auto plan = static_cast<cudnnBackendDescriptor_t>(get_ptr(req, &ok));
      auto pack = static_cast<cudnnBackendDescriptor_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, fn(h, plan, pack));
      return true;
    }

    case API_cudnnCreateTensorDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t*)>(
          "cudnnCreateTensorDescriptor");
      if (!fn) { put_status(rsp, CUDNN_STATUS_NOT_SUPPORTED); return true; }
      cudnnTensorDescriptor_t d = nullptr;
      cudnnStatus_t s = fn(&d);
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(d));
      }
      return true;
    }
    case API_cudnnDestroyTensorDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t)>(
          "cudnnDestroyTensorDescriptor");
      auto d = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, fn(d));
      return true;
    }
    case API_cudnnSetTensorNdDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t,
                                            cudnnDataType_t, int, const int*,
                                            const int*)>(
          "cudnnSetTensorNdDescriptor");
      auto d = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      int32_t type = 0, nb = 0;
      const uint8_t* dims = nullptr;
      const uint8_t* strides = nullptr;
      size_t dn = 0, sn = 0;
      if (!fn || !ok || !req.get(&type) || !req.get(&nb) ||
          !req.get_sized(&dims, &dn) || !req.get_sized(&strides, &sn) ||
          nb <= 0 || nb > CUDNN_DIM_MAX ||
          dn != static_cast<size_t>(nb) * sizeof(int) || sn != dn) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, fn(d, static_cast<cudnnDataType_t>(type), nb,
                         reinterpret_cast<const int*>(dims),
                         reinterpret_cast<const int*>(strides)));
      return true;
    }
    case API_cudnnGetTensorNdDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t, int,
                                            cudnnDataType_t*, int*, int*,
                                            int*)>("cudnnGetTensorNdDescriptor");
      auto d = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      int32_t requested = 0;
      if (!fn || !ok || !req.get(&requested) || requested <= 0 ||
          requested > CUDNN_DIM_MAX) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnDataType_t type = CUDNN_DATA_FLOAT;
      int nb = 0;
      std::vector<int> dims(requested, 0), strides(requested, 0);
      cudnnStatus_t s = fn(d, requested, &type, &nb, dims.data(),
                           strides.data());
      put_status(rsp, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        rsp->put<int32_t>(static_cast<int32_t>(type));
        rsp->put<int32_t>(nb);
        rsp->put_sized(dims.data(), dims.size() * sizeof(int));
        rsp->put_sized(strides.data(), strides.size() * sizeof(int));
      }
      return true;
    }
    case API_cudnnDeriveBNTensorDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t,
                                            cudnnTensorDescriptor_t,
                                            cudnnBatchNormMode_t)>(
          "cudnnDeriveBNTensorDescriptor");
      auto derived = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      auto x = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      int32_t mode = 0;
      if (!fn || !ok || !req.get(&mode)) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp,
                 fn(derived, x, static_cast<cudnnBatchNormMode_t>(mode)));
      return true;
    }
    case API_cudnnBatchNormalizationForwardInference: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(
          cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*,
          cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
          cudnnTensorDescriptor_t, const void*, const void*, const void*,
          const void*, double)>("cudnnBatchNormalizationForwardInference");
      auto handle = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      int32_t mode = 0;
      uint32_t width = 0;
      const uint8_t* alpha = nullptr;
      const uint8_t* beta = nullptr;
      size_t an = 0, bn = 0;
      if (!fn || !ok || !req.get(&mode) || !req.get(&width) ||
          !req.get_sized(&alpha, &an) || !req.get_sized(&beta, &bn) ||
          an != width || bn != width ||
          (width != sizeof(float) && width != sizeof(double))) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      // Copied out of the frame so they are aligned for their type. A union
      // rather than a double: at four bytes wide cuDNN reads these as floats,
      // and a float widened to a double would be a different number.
      union Scalar { float f; double d; } a{}, b{};
      std::memcpy(&a, alpha, width);
      std::memcpy(&b, beta, width);
      auto xDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* x = get_ptr(req, &ok);
      auto yDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* y = get_ptr(req, &ok);
      auto bnDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* scale = get_ptr(req, &ok);
      void* bias = get_ptr(req, &ok);
      void* mean = get_ptr(req, &ok);
      void* var = get_ptr(req, &ok);
      double epsilon = 0;
      if (!ok || !req.get(&epsilon)) {
        put_status(rsp, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, fn(handle, static_cast<cudnnBatchNormMode_t>(mode), &a,
                         &b, xDesc, x, yDesc, y, bnDesc, scale, bias, mean,
                         var, epsilon));
      return true;
    }

    default:
      std::fprintf(stderr, "[rgpu-server] unhandled cuDNN id %u\n", id);
      put_status(rsp, CUDNN_STATUS_NOT_SUPPORTED);
      return true;
  }
}

}  // namespace rgpu
