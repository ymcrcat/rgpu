// Executes forwarded cuDNN calls against the real library on the GPU host.
//
// Opened on first use rather than linked, so a host without cuDNN still runs
// everything else.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

#include "common/cudnn_ids.h"
#include "common/cudnn_sizes.h"
#include "common/wire.h"
#include "server/inventory.h"

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

// How a handle is given back when the session that made it never comes back.
// The descriptors a client minted are not tracked this way: they are held in
// t_handles below, which belongs to the thread, and they are host-side objects
// rather than device memory.
CUresult destroy_handle(uint64_t h) {
  auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t)>("cudnnDestroy");
  if (!fn) return CUDA_ERROR_NOT_SUPPORTED;
  return fn(reinterpret_cast<cudnnHandle_t>(h)) == CUDNN_STATUS_SUCCESS
             ? CUDA_SUCCESS
             : CUDA_ERROR_UNKNOWN;
}

// The status goes in the payload, where a caller that waited for a reply can
// read the exact value. A failure also marks the frame itself, because a call
// sent without waiting has no payload to read: the frame result is what the
// server's deferred-error path carries to the next call that does reply.
void put_status(Buffer* rsp, CUresult* out, cudnnStatus_t s) {
  rsp->put<int32_t>(static_cast<int32_t>(s));
  if (s != CUDNN_STATUS_SUCCESS) *out = CUDA_ERROR_UNKNOWN;
}

// Descriptor handles the client minted for itself, mapped to the descriptors
// they stand for. One map per connection, because it is one thread per
// connection and the client's counter is only unique within its own process:
// two clients minting the same value must not find each other's descriptors.
constexpr uint64_t kHandleTag = 0x52475055ull << 32;  // "RGPU"

bool minted(uint64_t v) { return (v >> 32) == (kHandleTag >> 32); }

thread_local std::unordered_map<uint64_t, void*> t_handles;

// A minted handle resolves to its descriptor; anything else is a value in this
// process already, such as a device pointer or a cuDNN handle, and passes
// through. A minted handle with no entry is a client bug or a stale value, and
// is refused rather than dereferenced.
void* resolve(uint64_t v, bool* ok) {
  if (!minted(v)) return reinterpret_cast<void*>(v);
  auto it = t_handles.find(v);
  if (it == t_handles.end()) {
    std::fprintf(stderr, "[rgpu-server] unknown cuDNN handle %llx\n",
                 (unsigned long long)v);
    *ok = false;
    return nullptr;
  }
  return it->second;
}

uint64_t get_raw(Buffer& req, bool* ok) {
  uint64_t v = 0;
  if (!req.get(&v)) {
    *ok = false;
    return 0;
  }
  return v;
}

void* get_ptr(Buffer& req, bool* ok) {
  uint64_t v = get_raw(req, ok);
  if (!*ok) return nullptr;
  return resolve(v, ok);
}

// An attribute array of descriptors carries minted handles, one per element,
// which have to become real descriptors before cuDNN sees them. Returns false
// if any of them is unknown, which would otherwise reach the library as a
// wild pointer.
bool resolve_descriptors(const uint8_t* in, size_t n, std::vector<uint8_t>* out) {
  if (n % sizeof(uint64_t)) return false;
  out->assign(in, in + n);
  auto* v = reinterpret_cast<uint64_t*>(out->data());
  for (size_t i = 0; i < n / sizeof(uint64_t); i++) {
    bool ok = true;
    void* real = resolve(v[i], &ok);
    if (!ok) return false;
    v[i] = reinterpret_cast<uint64_t>(real);
  }
  return true;
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
      if (!fn) { put_status(rsp, out, CUDNN_STATUS_NOT_INITIALIZED); return true; }
      cudnnHandle_t h = nullptr;
      cudnnStatus_t s = fn(&h);
      put_status(rsp, out, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        rsp->put<uint64_t>(reinterpret_cast<uint64_t>(h));
        inventory_note_handle(reinterpret_cast<uint64_t>(h), "cuDNN",
                              &destroy_handle);
      }
      return true;
    }
    case API_cudnnDestroy: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t)>("cudnnDestroy");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      cudnnStatus_t s = fn(h);
      if (s == CUDNN_STATUS_SUCCESS) {
        inventory_forget_handle(reinterpret_cast<uint64_t>(h));
      }
      put_status(rsp, out, s);
      return true;
    }
    case API_cudnnSetStream: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t)>(
          "cudnnSetStream");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      auto st = static_cast<cudaStream_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, out, fn(h, st));
      return true;
    }
    case API_cudnnGetStream: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t*)>(
          "cudnnGetStream");
      auto h = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      cudaStream_t st = nullptr;
      cudnnStatus_t s = fn(h, &st);
      put_status(rsp, out, s);
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
      if (!fn) { put_status(rsp, out, CUDNN_STATUS_NOT_INITIALIZED); return true; }
      put_status(rsp, out, CUDNN_STATUS_SUCCESS);
      rsp->put<uint64_t>(static_cast<uint64_t>(fn()));
      return true;
    }
    case API_cudnnGetLastErrorString: {
      auto fn = cudnn_sym<void (*)(char*, size_t)>("cudnnGetLastErrorString");
      uint64_t max_size = 0;
      if (!fn || !req.get(&max_size)) {
        put_status(rsp, out, CUDNN_STATUS_NOT_SUPPORTED);
        return true;
      }
      // The caller's buffer can be any size it likes; this one only has to be
      // big enough for a message.
      if (max_size > 4096) max_size = 4096;
      std::vector<char> buf(max_size ? max_size : 1, '\0');
      fn(buf.data(), buf.size());
      buf.back() = '\0';
      put_status(rsp, out, CUDNN_STATUS_SUCCESS);
      rsp->put_sized(buf.data(), std::strlen(buf.data()));
      return true;
    }
    case API_cudnnGetMaxDeviceVersion: {
      auto fn = cudnn_sym<size_t (*)(void)>("cudnnGetMaxDeviceVersion");
      if (!fn) { put_status(rsp, out, CUDNN_STATUS_NOT_SUPPORTED); return true; }
      put_status(rsp, out, CUDNN_STATUS_SUCCESS);
      rsp->put<uint64_t>(static_cast<uint64_t>(fn()));
      return true;
    }
    case API_cudnnGetProperty: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(libraryPropertyType, int*)>(
          "cudnnGetProperty");
      int32_t type = 0;
      if (!fn || !req.get(&type)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      int v = 0;
      cudnnStatus_t s = fn(static_cast<libraryPropertyType>(type), &v);
      put_status(rsp, out, s);
      if (s == CUDNN_STATUS_SUCCESS) rsp->put<int32_t>(v);
      return true;
    }

    case API_cudnnBackendCreateDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnBackendDescriptorType_t,
                                            cudnnBackendDescriptor_t*)>(
          "cudnnBackendCreateDescriptor");
      int32_t type = 0;
      uint64_t handle = 0;
      if (!fn || !req.get(&type) || !(handle = get_raw(req, &ok), ok) ||
          !minted(handle)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnBackendDescriptor_t d = nullptr;
      cudnnStatus_t s =
          fn(static_cast<cudnnBackendDescriptorType_t>(type), &d);
      if (s == CUDNN_STATUS_SUCCESS) t_handles[handle] = d;
      put_status(rsp, out, s);
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
      uint64_t handle = get_raw(req, &ok);
      auto d = static_cast<cudnnBackendDescriptor_t>(resolve(handle, &ok));
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      cudnnStatus_t s = fn(d);
      // The handle is gone whatever the library said: the client has already
      // dropped it, so keeping the entry would only let a later stale use of
      // that value find a destroyed descriptor.
      if (id == API_cudnnBackendDestroyDescriptor) t_handles.erase(handle);
      put_status(rsp, out, s);
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      const uint8_t* bytes = nullptr;
      size_t n = 0;
      if (present && !req.get_sized(&bytes, &n)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      // cuDNN reads count elements of the type's width. The bytes that
      // arrived have to be exactly that, or it reads past them.
      if (present && n != static_cast<size_t>(count) *
                              rgpu::cudnn_element_size(
                                  static_cast<cudnnBackendAttributeType_t>(type))) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      std::vector<uint8_t> resolved;
      if (present && type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
        if (!resolve_descriptors(bytes, n, &resolved)) {
          put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
          return true;
        }
        bytes = resolved.data();
      }
      put_status(rsp, out, fn(d, static_cast<cudnnBackendAttributeName_t>(name),
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      // The caller's current contents came with the request, because for
      // descriptor arrays they are inputs that cuDNN fills in place.
      std::vector<uint8_t> buf;
      std::vector<uint8_t> as_sent;
      if (has_array) {
        const uint8_t* bytes = nullptr;
        size_t n = 0;
        if (!req.get_sized(&bytes, &n)) {
          put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
          return true;
        }
        // cuDNN writes up to the requested count of elements into this
        // buffer, so it has to have room for exactly that many.
        if (n != static_cast<size_t>(requested) *
                     rgpu::cudnn_element_size(
                         static_cast<cudnnBackendAttributeType_t>(type))) {
          put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
          return true;
        }
        as_sent.assign(bytes, bytes + n);
        if (type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
          // cuDNN fills the caller's own descriptors in place, so what comes
          // back is the same set it was given. The client gets its own handles
          // back rather than the values behind them.
          if (!resolve_descriptors(bytes, n, &buf)) {
            put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
            return true;
          }
        } else {
          buf.assign(bytes, bytes + n);
        }
      }
      int64_t produced = 0;
      cudnnStatus_t s = fn(d, static_cast<cudnnBackendAttributeName_t>(name),
                           static_cast<cudnnBackendAttributeType_t>(type),
                           requested, want_count ? &produced : nullptr,
                           has_array && !buf.empty() ? buf.data() : nullptr);
      put_status(rsp, out, s);
      if (s == CUDNN_STATUS_SUCCESS) {
        if (want_count) rsp->put<int64_t>(produced);
        if (has_array) {
          const std::vector<uint8_t>& back =
              type == CUDNN_TYPE_BACKEND_DESCRIPTOR ? as_sent : buf;
          rsp->put_sized(back.data(), back.size());
        }
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
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      put_status(rsp, out, fn(h, plan, pack));
      return true;
    }

    case API_cudnnCreateTensorDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t*)>(
          "cudnnCreateTensorDescriptor");
      uint64_t handle = get_raw(req, &ok);
      if (!fn || !ok || !minted(handle)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnTensorDescriptor_t d = nullptr;
      cudnnStatus_t s = fn(&d);
      if (s == CUDNN_STATUS_SUCCESS) t_handles[handle] = d;
      put_status(rsp, out, s);
      return true;
    }
    case API_cudnnDestroyTensorDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnTensorDescriptor_t)>(
          "cudnnDestroyTensorDescriptor");
      uint64_t handle = get_raw(req, &ok);
      auto d = static_cast<cudnnTensorDescriptor_t>(resolve(handle, &ok));
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      cudnnStatus_t s = fn(d);
      t_handles.erase(handle);
      put_status(rsp, out, s);
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, out, fn(d, static_cast<cudnnDataType_t>(type), nb,
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnDataType_t type = CUDNN_DATA_FLOAT;
      int nb = 0;
      std::vector<int> dims(requested, 0), strides(requested, 0);
      cudnnStatus_t s = fn(d, requested, &type, &nb, dims.data(),
                           strides.data());
      put_status(rsp, out, s);
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, out,
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
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
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, out, fn(handle, static_cast<cudnnBatchNormMode_t>(mode), &a,
                         &b, xDesc, x, yDesc, y, bnDesc, scale, bias, mean,
                         var, epsilon));
      return true;
    }

    case API_cudnnCreateActivationDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnActivationDescriptor_t*)>(
          "cudnnCreateActivationDescriptor");
      uint64_t handle = get_raw(req, &ok);
      if (!fn || !ok || !minted(handle)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnActivationDescriptor_t d = nullptr;
      cudnnStatus_t s = fn(&d);
      if (s == CUDNN_STATUS_SUCCESS) t_handles[handle] = d;
      put_status(rsp, out, s);
      return true;
    }
    case API_cudnnSetActivationDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnActivationDescriptor_t,
                                            cudnnActivationMode_t,
                                            cudnnNanPropagation_t, double)>(
          "cudnnSetActivationDescriptor");
      auto d = static_cast<cudnnActivationDescriptor_t>(get_ptr(req, &ok));
      int32_t mode = 0, nan_opt = 0;
      double coef = 0;
      if (!fn || !ok || !req.get(&mode) || !req.get(&nan_opt) ||
          !req.get(&coef)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, out,
                 fn(d, static_cast<cudnnActivationMode_t>(mode),
                    static_cast<cudnnNanPropagation_t>(nan_opt), coef));
      return true;
    }
    case API_cudnnDestroyActivationDescriptor: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(cudnnActivationDescriptor_t)>(
          "cudnnDestroyActivationDescriptor");
      uint64_t handle = get_raw(req, &ok);
      auto d = static_cast<cudnnActivationDescriptor_t>(resolve(handle, &ok));
      if (!fn || !ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }
      cudnnStatus_t s = fn(d);
      t_handles.erase(handle);
      put_status(rsp, out, s);
      return true;
    }

    case API_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize:
    case API_cudnnGetBatchNormalizationBackwardExWorkspaceSize:
    case API_cudnnGetBatchNormalizationTrainingExReserveSpaceSize: {
      // The three differ only in how many descriptors they take, and the
      // client sends that count, so one case reads all of them.
      auto handle = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      int32_t mode = 0, bn_ops = 0;
      uint32_t count = 0;
      cudnnActivationDescriptor_t act = nullptr;
      if (!ok || !req.get(&mode) || !req.get(&bn_ops)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      act = static_cast<cudnnActivationDescriptor_t>(get_ptr(req, &ok));
      if (!ok || !req.get(&count) || count > 8) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      cudnnTensorDescriptor_t d[8] = {};
      for (uint32_t i = 0; i < count; i++) {
        d[i] = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      }
      if (!ok) { put_status(rsp, out, CUDNN_STATUS_BAD_PARAM); return true; }

      size_t size = 0;
      cudnnStatus_t s = CUDNN_STATUS_NOT_SUPPORTED;
      const auto m = static_cast<cudnnBatchNormMode_t>(mode);
      const auto ops = static_cast<cudnnBatchNormOps_t>(bn_ops);
      if (id == API_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize) {
        auto fn = cudnn_sym<cudnnStatus_t (*)(
            cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
            cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
            cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
            cudnnActivationDescriptor_t, size_t*)>(
            "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
        if (fn && count == 4) {
          s = fn(handle, m, ops, d[0], d[1], d[2], d[3], act, &size);
        }
      } else if (id == API_cudnnGetBatchNormalizationBackwardExWorkspaceSize) {
        auto fn = cudnn_sym<cudnnStatus_t (*)(
            cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
            cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
            cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
            cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
            cudnnActivationDescriptor_t, size_t*)>(
            "cudnnGetBatchNormalizationBackwardExWorkspaceSize");
        if (fn && count == 6) {
          s = fn(handle, m, ops, d[0], d[1], d[2], d[3], d[4], d[5], act,
                 &size);
        }
      } else {
        auto fn = cudnn_sym<cudnnStatus_t (*)(
            cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
            cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, size_t*)>(
            "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
        if (fn && count == 1) s = fn(handle, m, ops, act, d[0], &size);
      }
      put_status(rsp, out, s);
      if (s == CUDNN_STATUS_SUCCESS) rsp->put<uint64_t>(size);
      return true;
    }

    case API_cudnnBatchNormalizationForwardTrainingEx: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(
          cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
          const void*, const void*, cudnnTensorDescriptor_t, const void*,
          cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
          cudnnTensorDescriptor_t, const void*, const void*, double, void*,
          void*, double, void*, void*, cudnnActivationDescriptor_t, void*,
          size_t, void*, size_t)>(
          "cudnnBatchNormalizationForwardTrainingEx");
      auto handle = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      int32_t mode = 0, bn_ops = 0;
      uint32_t width = 0;
      const uint8_t* alpha = nullptr;
      const uint8_t* beta = nullptr;
      size_t an = 0, bn = 0;
      if (!fn || !ok || !req.get(&mode) || !req.get(&bn_ops) ||
          !req.get(&width) || !req.get_sized(&alpha, &an) ||
          !req.get_sized(&beta, &bn) || an != width || bn != width ||
          (width != sizeof(float) && width != sizeof(double))) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      union Scalar { float f; double d; } a{}, b{};
      std::memcpy(&a, alpha, width);
      std::memcpy(&b, beta, width);

      auto xDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* x = get_ptr(req, &ok);
      auto zDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* z = get_ptr(req, &ok);
      auto yDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* y = get_ptr(req, &ok);
      auto bnDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* scale = get_ptr(req, &ok);
      void* bias = get_ptr(req, &ok);
      double average = 0, epsilon = 0;
      if (!ok || !req.get(&average)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      void* running_mean = get_ptr(req, &ok);
      void* running_var = get_ptr(req, &ok);
      if (!ok || !req.get(&epsilon)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      void* save_mean = get_ptr(req, &ok);
      void* save_inv_var = get_ptr(req, &ok);
      auto act = static_cast<cudnnActivationDescriptor_t>(get_ptr(req, &ok));
      void* workspace = get_ptr(req, &ok);
      uint64_t workspace_size = 0, reserve_size = 0;
      if (!ok || !req.get(&workspace_size)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      void* reserve = get_ptr(req, &ok);
      if (!ok || !req.get(&reserve_size)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, out,
                 fn(handle, static_cast<cudnnBatchNormMode_t>(mode),
                    static_cast<cudnnBatchNormOps_t>(bn_ops), &a, &b, xDesc, x,
                    zDesc, z, yDesc, y, bnDesc, scale, bias, average,
                    running_mean, running_var, epsilon, save_mean,
                    save_inv_var, act, workspace, workspace_size, reserve,
                    reserve_size));
      return true;
    }

    case API_cudnnBatchNormalizationBackwardEx: {
      auto fn = cudnn_sym<cudnnStatus_t (*)(
          cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
          const void*, const void*, const void*, const void*,
          cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t,
          const void*, cudnnTensorDescriptor_t, const void*,
          cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, void*,
          cudnnTensorDescriptor_t, const void*, const void*, void*, void*,
          double, const void*, const void*, cudnnActivationDescriptor_t,
          void*, size_t, void*, size_t)>("cudnnBatchNormalizationBackwardEx");
      auto handle = static_cast<cudnnHandle_t>(get_ptr(req, &ok));
      int32_t mode = 0, bn_ops = 0;
      uint32_t width = 0;
      const uint8_t* s4[4] = {};
      size_t n4[4] = {};
      if (!fn || !ok || !req.get(&mode) || !req.get(&bn_ops) ||
          !req.get(&width) ||
          (width != sizeof(float) && width != sizeof(double))) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      for (int i = 0; i < 4; i++) {
        if (!req.get_sized(&s4[i], &n4[i]) || n4[i] != width) {
          put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
          return true;
        }
      }
      union Scalar { float f; double d; } sc[4]{};
      for (int i = 0; i < 4; i++) std::memcpy(&sc[i], s4[i], width);

      auto xDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* x = get_ptr(req, &ok);
      auto yDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* y = get_ptr(req, &ok);
      auto dyDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* dy = get_ptr(req, &ok);
      auto dzDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* dz = get_ptr(req, &ok);
      auto dxDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* dx = get_ptr(req, &ok);
      auto bnDesc = static_cast<cudnnTensorDescriptor_t>(get_ptr(req, &ok));
      void* scale = get_ptr(req, &ok);
      void* bias = get_ptr(req, &ok);
      void* dscale = get_ptr(req, &ok);
      void* dbias = get_ptr(req, &ok);
      double epsilon = 0;
      if (!ok || !req.get(&epsilon)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      void* saved_mean = get_ptr(req, &ok);
      void* saved_inv_var = get_ptr(req, &ok);
      auto act = static_cast<cudnnActivationDescriptor_t>(get_ptr(req, &ok));
      void* workspace = get_ptr(req, &ok);
      uint64_t workspace_size = 0, reserve_size = 0;
      if (!ok || !req.get(&workspace_size)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      void* reserve = get_ptr(req, &ok);
      if (!ok || !req.get(&reserve_size)) {
        put_status(rsp, out, CUDNN_STATUS_BAD_PARAM);
        return true;
      }
      put_status(rsp, out,
                 fn(handle, static_cast<cudnnBatchNormMode_t>(mode),
                    static_cast<cudnnBatchNormOps_t>(bn_ops), &sc[0], &sc[1],
                    &sc[2], &sc[3], xDesc, x, yDesc, y, dyDesc, dy, dzDesc, dz,
                    dxDesc, dx, bnDesc, scale, bias, dscale, dbias, epsilon,
                    saved_mean, saved_inv_var, act, workspace, workspace_size,
                    reserve, reserve_size));
      return true;
    }

    default:
      std::fprintf(stderr, "[rgpu-server] unhandled cuDNN id %u\n", id);
      put_status(rsp, out, CUDNN_STATUS_NOT_SUPPORTED);
      return true;
  }
}

}  // namespace rgpu
