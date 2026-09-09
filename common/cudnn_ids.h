// Wire ids for forwarded cuDNN calls.
//
// cuDNN cannot run on the client for the same reason cuBLAS cannot: it
// initialises through the driver's undocumented export tables. The real
// library runs on the GPU host and these calls carry the arguments to it.
#pragma once

#include <cstdint>

namespace rgpu {

enum CudnnId : uint32_t {
  kCudnnBase = 0x52000000u,

  API_cudnnCreate = kCudnnBase + 1,
  API_cudnnDestroy = kCudnnBase + 2,
  API_cudnnSetStream = kCudnnBase + 3,
  API_cudnnGetStream = kCudnnBase + 4,
  API_cudnnGetVersion = kCudnnBase + 5,
  API_cudnnGetCudartVersion = kCudnnBase + 6,
  API_cudnnGetProperty = kCudnnBase + 7,

  // Not generated: these two do not return a status, so the stub generator
  // skips them, and torch links against the first of them by name.
  API_cudnnGetLastErrorString = kCudnnBase + 8,
  API_cudnnGetMaxDeviceVersion = kCudnnBase + 9,

  // The backend graph API, which is what PyTorch actually uses. Everything a
  // convolution needs is expressed through descriptors and attributes, so
  // these few functions carry the whole surface.
  API_cudnnBackendCreateDescriptor = kCudnnBase + 10,
  API_cudnnBackendDestroyDescriptor = kCudnnBase + 11,
  API_cudnnBackendInitialize = kCudnnBase + 12,
  API_cudnnBackendFinalize = kCudnnBase + 13,
  API_cudnnBackendSetAttribute = kCudnnBase + 14,
  API_cudnnBackendGetAttribute = kCudnnBase + 15,
  API_cudnnBackendExecute = kCudnnBase + 16,

  // The legacy descriptor API. Convolution goes through the graph API above,
  // but batch normalisation still has no graph equivalent in PyTorch, so
  // ResNet needs these.
  API_cudnnCreateTensorDescriptor = kCudnnBase + 20,
  API_cudnnDestroyTensorDescriptor = kCudnnBase + 21,
  API_cudnnSetTensorNdDescriptor = kCudnnBase + 22,
  API_cudnnGetTensorNdDescriptor = kCudnnBase + 23,
  API_cudnnDeriveBNTensorDescriptor = kCudnnBase + 24,
  API_cudnnBatchNormalizationForwardInference = kCudnnBase + 25,
};

}  // namespace rgpu
