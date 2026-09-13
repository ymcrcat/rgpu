// Ids for calls that are ours rather than CUDA's. Kept well above the
// generated range so the two can never collide.
#pragma once

#include <cstdint>

#include "common/generated/api_ids.h"

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
  // cuThreadExchangeStreamCaptureMode. Its one parameter is read as well as
  // written: the caller passes the mode it wants and gets the old one back.
  // The generator saw only a pointer being written to, sent no value, and so
  // asked for mode zero whatever the caller wanted.
  API_rgpu_capture_mode = kInternalBase + 6,
  // Client threads that have exited, so the server can drop whatever it keeps
  // for them. Payload: a uint32_t count, then that many thread ids. Sent
  // without a reply, ahead of the next frame any thread queues; best-effort,
  // since a lost one only leaks a little until the session expires.
  API_rgpu_thread_gone = kInternalBase + 7,
};

// api_name() is generated from cuda.h and so knows only CUDA's own ids;
// everything above comes out as "?". That includes the kernel launch, which is
// the call a log line most needs to name, because it is the one that carries a
// failure nothing else will report. Answers for both ranges.
inline const char* call_name(uint32_t id) {
  switch (id) {
    case API_rgpu_param_layout: return "rgpu_param_layout";
    case API_rgpu_launch: return "rgpu_launch";
    case API_rgpu_hello: return "rgpu_hello";
    case API_rgpu_capture_info: return "rgpu_capture_info";
    case API_rgpu_graph_nodes: return "rgpu_graph_nodes";
    case API_rgpu_capture_mode: return "rgpu_capture_mode";
    case API_rgpu_thread_gone: return "rgpu_thread_gone";
    default: return api_name(id);
  }
}

}  // namespace rgpu
