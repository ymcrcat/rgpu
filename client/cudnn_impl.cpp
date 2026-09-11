// The cuDNN calls PyTorch makes, forwarded to the GPU host.
//
// cuDNN cannot run on the client. Like the stock CUDA runtime and cuBLAS, it
// initialises through the driver's undocumented export tables, which hold
// pointers into the driver's own process and cannot cross a machine boundary.
//
// PyTorch reaches cuDNN through the backend graph API, which is why this file
// is short. Convolutions, matmuls, normalisations and the rest are all built
// out of descriptors and attributes, so nine functions carry the whole surface
// rather than the two hundred entry points the library exports.
//
// Attribute values are arrays of fixed-size elements whose width follows from
// the attribute type, so they travel verbatim. That is right even for the
// pointer-shaped types: a device pointer, a handle and a descriptor are all
// already values in the server's address space.
//
// Definitions here are strong and override the weak, logging ones in
// client/generated/cudnn_stubs.cpp.

#include <atomic>
#include <initializer_list>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cudnn.h>

#include "client/rpc.h"
#include "common/cudnn_ids.h"

namespace {

// Width of one element of an attribute array. Getting this wrong would send
// the wrong number of bytes, so the unknown case refuses rather than guesses.
size_t element_size(cudnnBackendAttributeType_t t) {
  switch (t) {
    case CUDNN_TYPE_INT64:
    case CUDNN_TYPE_DOUBLE:
      return 8;
    // A handle, a device pointer and a descriptor are all pointer-sized values
    // belonging to the server. Copying the bytes is exactly right.
    case CUDNN_TYPE_VOID_PTR:
    case CUDNN_TYPE_HANDLE:
    case CUDNN_TYPE_BACKEND_DESCRIPTOR:
      return sizeof(void*);
    case CUDNN_TYPE_FLOAT:
    case CUDNN_TYPE_INT32:
      return 4;
    case CUDNN_TYPE_BOOLEAN:
      return sizeof(bool);
    case CUDNN_TYPE_CHAR:
      return 1;
    case CUDNN_TYPE_FRACTION:
      return sizeof(cudnnFraction_t);
    default:
      // Every remaining type is an enumeration, which is int-sized. Listing
      // them all would be a maintenance burden for no gain, but an unexpected
      // value is worth saying out loud.
      if (t < CUDNN_TYPE_HANDLE || t > CUDNN_TYPE_TENSOR_REORDERING_MODE) {
        rgpu::log("unknown cuDNN attribute type %d; assuming an enumeration",
                  static_cast<int>(t));
      }
      return sizeof(int);
  }
}

cudnnStatus_t from_cu(CUresult r) {
  return r == CUDA_SUCCESS ? CUDNN_STATUS_SUCCESS
                           : CUDNN_STATUS_EXECUTION_FAILED;
}

// The status travels in the payload; the frame's result field is a CUresult.
// This call's own status is the more precise of the two and wins, because the
// caller may act on the difference: a cuDNN frontend treats "not supported" as
// "try another engine" and anything else as fatal. A failed frame with a
// successful payload is a call that was sent without waiting for a reply and
// failed back then, which is reported here for the same reason CUDA reports
// launch failures at the next synchronization.
cudnnStatus_t send(uint32_t id, rgpu::Buffer& req, rgpu::Buffer* rsp) {
  CUresult r = rgpu::call(id, req, rsp);
  int32_t status = CUDNN_STATUS_SUCCESS;
  const bool have = rsp->get(&status);
  if (have && status != CUDNN_STATUS_SUCCESS) {
    return static_cast<cudnnStatus_t>(status);
  }
  if (r != CUDA_SUCCESS) return from_cu(r);
  return have ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_INTERNAL_ERROR;
}

// Sent without waiting for a reply. Only for calls that return nothing but a
// status, and only where a failure is still caught: an attribute that fails to
// set makes the finalize that follows it fail, and finalize does wait. That is
// the same bargain CUDA itself makes with asynchronous launches, and it is
// worth roughly 200 round trips per inference.
cudnnStatus_t send_async(uint32_t id, rgpu::Buffer& req) {
  return from_cu(rgpu::call_async(id, req));
}

// A descriptor handle minted here instead of asked for.
//
// Creating a descriptor was a round trip purely because the caller needed a
// value back, 60 per inference. It does not have to be the server's value:
// the client picks one, tells the server what it picked, and the server maps
// it to the real descriptor. The call then needs no reply.
//
// The tag makes these unmistakable, so a value that was never minted, or that
// belongs to a connection that has gone, is a clear error on the server rather
// than a pointer it might dereference. The counter is per process and the map
// is per connection, so two clients cannot collide.
constexpr uint64_t kHandleTag = 0x52475055ull << 32;  // "RGPU"
std::atomic<uint32_t> g_next_handle{1};

void* mint_handle() {
  return reinterpret_cast<void*>(kHandleTag | g_next_handle.fetch_add(1));
}

void put_ptr(rgpu::Buffer& b, const void* p) {
  b.put<uint64_t>(reinterpret_cast<uint64_t>(p));
}

// A tensor descriptor's element type, remembered when it is set. The
// descriptor itself lives on the server, but alpha and beta are host scalars
// whose width follows from it: double for a double tensor, float otherwise.
// Reading the wrong width off the caller's stack is the sort of bug that only
// shows up as slightly wrong numbers, so this is worth tracking.
std::mutex g_type_mu;
std::unordered_map<const void*, int> g_tensor_type;

void remember_type(const void* desc, int type) {
  std::lock_guard<std::mutex> lock(g_type_mu);
  g_tensor_type[desc] = type;
}

void forget_type(const void* desc) {
  std::lock_guard<std::mutex> lock(g_type_mu);
  g_tensor_type.erase(desc);
}

size_t scalar_width(const void* desc) {
  std::lock_guard<std::mutex> lock(g_type_mu);
  auto it = g_tensor_type.find(desc);
  // An unset descriptor cannot be used in a call anyway; float is the answer
  // for every type except double.
  return it != g_tensor_type.end() && it->second == CUDNN_DATA_DOUBLE
             ? sizeof(double)
             : sizeof(float);
}

}  // namespace

extern "C" {

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
  if (!handle) return CUDNN_STATUS_BAD_PARAM;
  rgpu::Buffer req, rsp;
  cudnnStatus_t s = send(rgpu::API_cudnnCreate, req, &rsp);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  uint64_t h = 0;
  if (!rsp.get(&h)) return CUDNN_STATUS_INTERNAL_ERROR;
  *handle = reinterpret_cast<cudnnHandle_t>(h);
  return s;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  rgpu::Buffer req, rsp;
  put_ptr(req, handle);
  return send(rgpu::API_cudnnDestroy, req, &rsp);
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  rgpu::Buffer req, rsp;
  put_ptr(req, handle);
  put_ptr(req, streamId);
  return send_async(rgpu::API_cudnnSetStream, req);
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* streamId) {
  if (!streamId) return CUDNN_STATUS_BAD_PARAM;
  rgpu::Buffer req, rsp;
  put_ptr(req, handle);
  cudnnStatus_t s = send(rgpu::API_cudnnGetStream, req, &rsp);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  uint64_t v = 0;
  if (!rsp.get(&v)) return CUDNN_STATUS_INTERNAL_ERROR;
  *streamId = reinterpret_cast<cudaStream_t>(v);
  return s;
}

size_t cudnnGetVersion(void) {
  rgpu::Buffer req, rsp;
  if (send(rgpu::API_cudnnGetVersion, req, &rsp) != CUDNN_STATUS_SUCCESS) {
    return 0;
  }
  uint64_t v = 0;
  return rsp.get(&v) ? static_cast<size_t>(v) : 0;
}

size_t cudnnGetCudartVersion(void) {
  rgpu::Buffer req, rsp;
  if (send(rgpu::API_cudnnGetCudartVersion, req, &rsp) !=
      CUDNN_STATUS_SUCCESS) {
    return 0;
  }
  uint64_t v = 0;
  return rsp.get(&v) ? static_cast<size_t>(v) : 0;
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUDNN_STATUS_BAD_PARAM;
  rgpu::Buffer req, rsp;
  req.put<int32_t>(static_cast<int32_t>(type));
  cudnnStatus_t s = send(rgpu::API_cudnnGetProperty, req, &rsp);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  int32_t v = 0;
  if (!rsp.get(&v)) return CUDNN_STATUS_INTERNAL_ERROR;
  *value = v;
  return s;
}

// The last error happened on the GPU host, so its text has to come from
// there. The caller owns the buffer, so the reply is copied into it.
void cudnnGetLastErrorString(char* message, size_t max_size) {
  if (!message || max_size == 0) return;
  message[0] = '\0';
  rgpu::Buffer req, rsp;
  req.put<uint64_t>(static_cast<uint64_t>(max_size));
  if (send(rgpu::API_cudnnGetLastErrorString, req, &rsp) !=
      CUDNN_STATUS_SUCCESS) {
    return;
  }
  const uint8_t* bytes = nullptr;
  size_t n = 0;
  if (!rsp.get_sized(&bytes, &n)) return;
  if (n > max_size - 1) n = max_size - 1;
  std::memcpy(message, bytes, n);
  message[n] = '\0';
}

size_t cudnnGetMaxDeviceVersion(void) {
  rgpu::Buffer req, rsp;
  if (send(rgpu::API_cudnnGetMaxDeviceVersion, req, &rsp) !=
      CUDNN_STATUS_SUCCESS) {
    return 0;
  }
  uint64_t v = 0;
  return rsp.get(&v) ? static_cast<size_t>(v) : 0;
}

// Error strings are static data in the real library, so answering locally
// avoids a round trip and gives the caller a pointer that stays valid.
const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
    default: return "CUDNN_STATUS_UNKNOWN";
  }
}

// --- the backend graph API -------------------------------------------------

cudnnStatus_t cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t* descriptor) {
  if (!descriptor) return CUDNN_STATUS_BAD_PARAM;
  void* h = mint_handle();
  rgpu::Buffer req;
  req.put<int32_t>(static_cast<int32_t>(descriptorType));
  put_ptr(req, h);
  cudnnStatus_t s = send_async(rgpu::API_cudnnBackendCreateDescriptor, req);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  *descriptor = static_cast<cudnnBackendDescriptor_t>(h);
  return s;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(
    cudnnBackendDescriptor_t descriptor) {
  rgpu::Buffer req, rsp;
  put_ptr(req, descriptor);
  return send_async(rgpu::API_cudnnBackendDestroyDescriptor, req);
}

cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor) {
  rgpu::Buffer req, rsp;
  put_ptr(req, descriptor);
  return send(rgpu::API_cudnnBackendInitialize, req, &rsp);
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
  rgpu::Buffer req, rsp;
  put_ptr(req, descriptor);
  return send(rgpu::API_cudnnBackendFinalize, req, &rsp);
}

cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                                       cudnnBackendAttributeName_t attributeName,
                                       cudnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements) {
  if (elementCount < 0) return CUDNN_STATUS_BAD_PARAM;
  const size_t width = element_size(attributeType);
  rgpu::Buffer req, rsp;
  put_ptr(req, descriptor);
  req.put<int32_t>(static_cast<int32_t>(attributeName));
  req.put<int32_t>(static_cast<int32_t>(attributeType));
  req.put<int64_t>(elementCount);
  req.put<uint8_t>(arrayOfElements ? 1 : 0);
  if (arrayOfElements) {
    req.put_sized(arrayOfElements, static_cast<size_t>(elementCount) * width);
  }
  return send_async(rgpu::API_cudnnBackendSetAttribute, req);
}

cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t descriptor,
                                       cudnnBackendAttributeName_t attributeName,
                                       cudnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) {
  if (requestedElementCount < 0) return CUDNN_STATUS_BAD_PARAM;
  const size_t width = element_size(attributeType);
  const size_t bytes = static_cast<size_t>(requestedElementCount) * width;

  rgpu::Buffer req, rsp;
  put_ptr(req, descriptor);
  req.put<int32_t>(static_cast<int32_t>(attributeName));
  req.put<int32_t>(static_cast<int32_t>(attributeType));
  req.put<int64_t>(requestedElementCount);
  req.put<uint8_t>(elementCount ? 1 : 0);
  // The caller's buffer goes out as well as coming back. For descriptor
  // arrays the caller passes descriptors it created and cuDNN fills them in
  // place, so the outgoing contents are part of the request, not just space
  // to be overwritten.
  req.put<uint8_t>(arrayOfElements ? 1 : 0);
  if (arrayOfElements) req.put_sized(arrayOfElements, bytes);

  cudnnStatus_t s = send(rgpu::API_cudnnBackendGetAttribute, req, &rsp);
  if (s != CUDNN_STATUS_SUCCESS) return s;

  if (elementCount) {
    int64_t n = 0;
    if (!rsp.get(&n)) return CUDNN_STATUS_INTERNAL_ERROR;
    *elementCount = n;
  }
  if (arrayOfElements) {
    const uint8_t* b = nullptr;
    size_t n = 0;
    if (!rsp.get_sized(&b, &n)) return CUDNN_STATUS_INTERNAL_ERROR;
    std::memcpy(arrayOfElements, b, n < bytes ? n : bytes);
  }
  return s;
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle,
                                  cudnnBackendDescriptor_t executionPlan,
                                  cudnnBackendDescriptor_t variantPack) {
  rgpu::Buffer req, rsp;
  put_ptr(req, handle);
  put_ptr(req, executionPlan);
  put_ptr(req, variantPack);
  return send_async(rgpu::API_cudnnBackendExecute, req);
}

// --- the legacy descriptor API ---------------------------------------------
//
// Only what batch normalisation needs. Convolution goes through the graph API
// above; PyTorch has no graph path for batch norm, so ResNet comes through
// here.

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* desc) {
  if (!desc) return CUDNN_STATUS_BAD_PARAM;
  void* h = mint_handle();
  rgpu::Buffer req;
  put_ptr(req, h);
  cudnnStatus_t s = send_async(rgpu::API_cudnnCreateTensorDescriptor, req);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  *desc = static_cast<cudnnTensorDescriptor_t>(h);
  return s;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc) {
  rgpu::Buffer req, rsp;
  put_ptr(req, desc);
  cudnnStatus_t s = send_async(rgpu::API_cudnnDestroyTensorDescriptor, req);
  forget_type(desc);
  return s;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t desc,
                                         cudnnDataType_t dataType, int nbDims,
                                         const int dimA[],
                                         const int strideA[]) {
  if (!dimA || !strideA || nbDims <= 0 || nbDims > CUDNN_DIM_MAX) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  rgpu::Buffer req, rsp;
  put_ptr(req, desc);
  req.put<int32_t>(static_cast<int32_t>(dataType));
  req.put<int32_t>(nbDims);
  req.put_sized(dimA, nbDims * sizeof(int));
  req.put_sized(strideA, nbDims * sizeof(int));
  cudnnStatus_t s = send_async(rgpu::API_cudnnSetTensorNdDescriptor, req);
  if (s == CUDNN_STATUS_SUCCESS) remember_type(desc, dataType);
  return s;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t desc,
                                         int nbDimsRequested,
                                         cudnnDataType_t* dataType,
                                         int* nbDims, int dimA[],
                                         int strideA[]) {
  if (nbDimsRequested <= 0 || nbDimsRequested > CUDNN_DIM_MAX) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  rgpu::Buffer req, rsp;
  put_ptr(req, desc);
  req.put<int32_t>(nbDimsRequested);
  cudnnStatus_t s = send(rgpu::API_cudnnGetTensorNdDescriptor, req, &rsp);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  int32_t type = 0, dims = 0;
  const uint8_t* d = nullptr;
  const uint8_t* st = nullptr;
  size_t dn = 0, sn = 0;
  if (!rsp.get(&type) || !rsp.get(&dims) || !rsp.get_sized(&d, &dn) ||
      !rsp.get_sized(&st, &sn)) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }
  if (dataType) *dataType = static_cast<cudnnDataType_t>(type);
  if (nbDims) *nbDims = dims;
  if (dimA) std::memcpy(dimA, d, dn);
  if (strideA) std::memcpy(strideA, st, sn);
  return s;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode) {
  rgpu::Buffer req, rsp;
  put_ptr(req, derivedBnDesc);
  put_ptr(req, xDesc);
  req.put<int32_t>(static_cast<int32_t>(mode));
  cudnnStatus_t s = send(rgpu::API_cudnnDeriveBNTensorDescriptor, req, &rsp);
  if (s == CUDNN_STATUS_SUCCESS) {
    // The derived descriptor carries the same element type in every case
    // PyTorch uses, and it is only ever the scale/bias descriptor, never the
    // one alpha is scaled against.
    remember_type(derivedBnDesc, CUDNN_DATA_FLOAT);
  }
  return s;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha,
    const void* beta, const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnTensorDescriptor_t yDesc, void* y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale,
    const void* bnBias, const void* estimatedMean,
    const void* estimatedVariance, double epsilon) {
  if (!alpha || !beta) return CUDNN_STATUS_BAD_PARAM;
  const size_t width = scalar_width(yDesc);
  rgpu::Buffer req, rsp;
  put_ptr(req, handle);
  req.put<int32_t>(static_cast<int32_t>(mode));
  // The width goes first so the server knows how to read what follows.
  req.put<uint32_t>(static_cast<uint32_t>(width));
  req.put_sized(alpha, width);
  req.put_sized(beta, width);
  put_ptr(req, xDesc);
  put_ptr(req, x);
  put_ptr(req, yDesc);
  put_ptr(req, y);
  put_ptr(req, bnScaleBiasMeanVarDesc);
  put_ptr(req, bnScale);
  put_ptr(req, bnBias);
  put_ptr(req, estimatedMean);
  put_ptr(req, estimatedVariance);
  req.put<double>(epsilon);
  return send_async(rgpu::API_cudnnBatchNormalizationForwardInference, req);
}

// --- batch normalisation in training mode ----------------------------------
//
// A different set of entry points from the inference form: these compute the
// batch's own statistics, update the running ones, and write a reserve space
// that the backward pass reads. The workspace and reserve are device memory
// the caller allocated, so they travel as pointers like any other.

cudnnStatus_t cudnnCreateActivationDescriptor(
    cudnnActivationDescriptor_t* activationDesc) {
  if (!activationDesc) return CUDNN_STATUS_BAD_PARAM;
  void* h = mint_handle();
  rgpu::Buffer req;
  put_ptr(req, h);
  cudnnStatus_t s = send_async(rgpu::API_cudnnCreateActivationDescriptor, req);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  *activationDesc = static_cast<cudnnActivationDescriptor_t>(h);
  return s;
}

cudnnStatus_t cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef) {
  rgpu::Buffer req;
  put_ptr(req, activationDesc);
  req.put<int32_t>(static_cast<int32_t>(mode));
  req.put<int32_t>(static_cast<int32_t>(reluNanOpt));
  req.put<double>(coef);
  return send_async(rgpu::API_cudnnSetActivationDescriptor, req);
}

cudnnStatus_t cudnnDestroyActivationDescriptor(
    cudnnActivationDescriptor_t activationDesc) {
  rgpu::Buffer req;
  put_ptr(req, activationDesc);
  cudnnStatus_t s =
      send_async(rgpu::API_cudnnDestroyActivationDescriptor, req);
  forget_type(activationDesc);
  return s;
}

namespace {

// The three size queries differ only in which descriptors they take, so they
// share a frame: the fixed arguments, then the descriptors in order.
cudnnStatus_t size_query(uint32_t id, cudnnHandle_t handle,
                         cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
                         cudnnActivationDescriptor_t activationDesc,
                         std::initializer_list<const void*> descriptors,
                         size_t* sizeInBytes) {
  if (!sizeInBytes) return CUDNN_STATUS_BAD_PARAM;
  rgpu::Buffer req, rsp;
  put_ptr(req, handle);
  req.put<int32_t>(static_cast<int32_t>(mode));
  req.put<int32_t>(static_cast<int32_t>(bnOps));
  put_ptr(req, activationDesc);
  req.put<uint32_t>(static_cast<uint32_t>(descriptors.size()));
  for (const void* d : descriptors) put_ptr(req, d);
  cudnnStatus_t s = send(id, req, &rsp);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  uint64_t v = 0;
  if (!rsp.get(&v)) return CUDNN_STATUS_INTERNAL_ERROR;
  *sizeInBytes = static_cast<size_t>(v);
  return s;
}

}  // namespace

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t* sizeInBytes) {
  return size_query(
      rgpu::API_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
      handle, mode, bnOps, activationDesc,
      {xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc}, sizeInBytes);
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t* sizeInBytes) {
  return size_query(
      rgpu::API_cudnnGetBatchNormalizationBackwardExWorkspaceSize, handle, mode,
      bnOps, activationDesc,
      {xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc}, sizeInBytes);
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes) {
  return size_query(
      rgpu::API_cudnnGetBatchNormalizationTrainingExReserveSpaceSize, handle,
      mode, bnOps, activationDesc, {xDesc}, sizeInBytes);
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void* alpha, const void* beta, const cudnnTensorDescriptor_t xDesc,
    const void* xData, const cudnnTensorDescriptor_t zDesc, const void* zData,
    const cudnnTensorDescriptor_t yDesc, void* yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScaleData, const void* bnBiasData,
    double exponentialAverageFactor, void* resultRunningMeanData,
    void* resultRunningVarianceData, double epsilon, void* saveMean,
    void* saveInvVariance, const cudnnActivationDescriptor_t activationDesc,
    void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  if (!alpha || !beta) return CUDNN_STATUS_BAD_PARAM;
  const size_t width = scalar_width(yDesc);
  rgpu::Buffer req;
  put_ptr(req, handle);
  req.put<int32_t>(static_cast<int32_t>(mode));
  req.put<int32_t>(static_cast<int32_t>(bnOps));
  req.put<uint32_t>(static_cast<uint32_t>(width));
  req.put_sized(alpha, width);
  req.put_sized(beta, width);
  const void* const forward_ptrs[] = {
      xDesc, xData, zDesc, zData, yDesc, yData,
      bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData};
  for (const void* p : forward_ptrs) put_ptr(req, p);
  req.put<double>(exponentialAverageFactor);
  put_ptr(req, resultRunningMeanData);
  put_ptr(req, resultRunningVarianceData);
  req.put<double>(epsilon);
  put_ptr(req, saveMean);
  put_ptr(req, saveInvVariance);
  put_ptr(req, activationDesc);
  put_ptr(req, workspace);
  req.put<uint64_t>(workSpaceSizeInBytes);
  put_ptr(req, reserveSpace);
  req.put<uint64_t>(reserveSpaceSizeInBytes);
  return send_async(rgpu::API_cudnnBatchNormalizationForwardTrainingEx, req);
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void* alphaDataDiff, const void* betaDataDiff,
    const void* alphaParamDiff, const void* betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void* xData,
    const cudnnTensorDescriptor_t yDesc, const void* yData,
    const cudnnTensorDescriptor_t dyDesc, const void* dyData,
    const cudnnTensorDescriptor_t dzDesc, void* dzData,
    const cudnnTensorDescriptor_t dxDesc, void* dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void* bnScaleData,
    const void* bnBiasData, void* dBnScaleData, void* dBnBiasData,
    double epsilon, const void* savedMean, const void* savedInvVariance,
    const cudnnActivationDescriptor_t activationDesc, void* workspace,
    size_t workSpaceSizeInBytes, void* reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  if (!alphaDataDiff || !betaDataDiff || !alphaParamDiff || !betaParamDiff) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const size_t width = scalar_width(dxDesc);
  rgpu::Buffer req;
  put_ptr(req, handle);
  req.put<int32_t>(static_cast<int32_t>(mode));
  req.put<int32_t>(static_cast<int32_t>(bnOps));
  req.put<uint32_t>(static_cast<uint32_t>(width));
  req.put_sized(alphaDataDiff, width);
  req.put_sized(betaDataDiff, width);
  req.put_sized(alphaParamDiff, width);
  req.put_sized(betaParamDiff, width);
  const void* const backward_ptrs[] = {
      xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData,
      dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData,
      dBnScaleData, dBnBiasData};
  for (const void* p : backward_ptrs) put_ptr(req, p);
  req.put<double>(epsilon);
  put_ptr(req, savedMean);
  put_ptr(req, savedInvVariance);
  put_ptr(req, activationDesc);
  put_ptr(req, workspace);
  req.put<uint64_t>(workSpaceSizeInBytes);
  put_ptr(req, reserveSpace);
  req.put<uint64_t>(reserveSpaceSizeInBytes);
  return send_async(rgpu::API_cudnnBatchNormalizationBackwardEx, req);
}

}  // extern "C"
