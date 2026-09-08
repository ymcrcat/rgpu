// Ids for calls that are ours rather than CUDA's. Kept well above the
// generated range so the two can never collide.
#pragma once

#include <cstdint>

namespace rgpu {

enum InternalId : uint32_t {
  kInternalBase = 0x40000000u,
  // Returns the device-side parameter layout of a CUfunction: the offset and
  // size of each kernel argument. The driver API does not tell the caller how
  // big a kernel argument is, so the client asks the server, which can see the
  // loaded module. See cuFuncGetParamInfo.
  API_rgpu_param_layout = kInternalBase + 1,
  // cuLaunchKernel with arguments already packed into one device-layout blob.
  API_rgpu_launch = kInternalBase + 2,
  // Server identification and version handshake.
  API_rgpu_hello = kInternalBase + 3,
};

}  // namespace rgpu
