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
  // cuStreamGetCaptureInfo_v2. Generated code cannot carry it: one of its
  // outputs is a pointer to an array the driver owns, which has no meaning on
  // the other side of a wire. The server copies the array out instead.
  API_rgpu_capture_info = kInternalBase + 4,
  // cuGraphGetNodes. Its count parameter is both the caller's capacity and the
  // number written, which the generator has no way to express: it treated the
  // array as a single handle and never sent the capacity, so the call would
  // have quietly reported no nodes at all.
  API_rgpu_graph_nodes = kInternalBase + 5,
};

}  // namespace rgpu
