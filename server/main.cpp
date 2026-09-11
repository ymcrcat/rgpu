// rgpu server: executes forwarded CUDA driver calls against a real GPU.
//
// Runs on the GPU host and links the real libcuda, so the generated dispatch
// can call driver functions by name.

#include <netinet/in.h>
#include <signal.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"

namespace rgpu {

// Defined in server/generated/dispatch.cpp.
bool dispatch_generated(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out);
// Defined in server/cublas_server.cpp, when cuBLAS support is built in.
bool dispatch_cublas(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out)
    __attribute__((weak));
// Defined in server/cublaslt_server.cpp likewise.
bool dispatch_cublaslt(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out)
    __attribute__((weak));
// Defined in server/cudnn_server.cpp likewise.
bool dispatch_cudnn(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out)
    __attribute__((weak));

namespace {

bool g_verbose = false;

void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[rgpu-server] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

// --- internal calls -------------------------------------------------------

// Reports each kernel argument's offset and size. The client cannot know these
// from the driver API alone, which is what makes remoting cuLaunchKernel hard.
CUresult handle_param_layout(Buffer& req, Buffer* rsp) {
  uint64_t fh = 0;
  if (!req.get(&fh)) return CUDA_ERROR_INVALID_VALUE;
  auto f = reinterpret_cast<CUfunction>(fh);

  std::vector<std::pair<uint64_t, uint64_t>> slots;
  for (size_t i = 0;; i++) {
    size_t offset = 0, size = 0;
    CUresult r = cuFuncGetParamInfo(f, i, &offset, &size);
    if (r == CUDA_ERROR_INVALID_VALUE) break;  // past the last parameter
    if (r != CUDA_SUCCESS) {
      logf("cuFuncGetParamInfo failed (%d); driver may predate CUDA 12.4", r);
      return r;
    }
    slots.emplace_back(offset, size);
    if (i > 1024) return CUDA_ERROR_INVALID_VALUE;  // runaway guard
  }

  rsp->put<uint32_t>(static_cast<uint32_t>(slots.size()));
  for (const auto& s : slots) {
    rsp->put<uint64_t>(s.first);
    rsp->put<uint64_t>(s.second);
  }
  return CUDA_SUCCESS;
}

// Launch with arguments already packed into the device-side layout. Handing
// the blob to the driver through extra[] is what lets us avoid reconstructing
// an argument pointer array.
CUresult handle_launch(Buffer& req, Buffer* rsp) {
  (void)rsp;
  uint64_t fh = 0, sh = 0;
  uint32_t gx, gy, gz, bx, by, bz, shmem;
  if (!req.get(&fh) || !req.get(&gx) || !req.get(&gy) || !req.get(&gz) ||
      !req.get(&bx) || !req.get(&by) || !req.get(&bz) || !req.get(&shmem) ||
      !req.get(&sh)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const uint8_t* args = nullptr;
  size_t args_len = 0;
  if (!req.get_sized(&args, &args_len)) return CUDA_ERROR_INVALID_VALUE;

  auto f = reinterpret_cast<CUfunction>(fh);
  auto stream = reinterpret_cast<CUstream>(sh);

  if (args_len == 0) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, nullptr,
                          nullptr);
  }
  // The driver reads from this buffer during the call only.
  void* buf = const_cast<uint8_t*>(args);
  size_t size = args_len;
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, buf,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, nullptr,
                        extra);
}

// cuStreamGetCaptureInfo_v2, with the dependency array copied out rather than
// pointed at. The driver owns that array and promises it stays valid until the
// next call that changes the capture; the client keeps its copy under the same
// rule, which is all the caller is allowed to assume.
CUresult handle_capture_info(Buffer& req, Buffer* rsp) {
  uint64_t stream_v = 0;
  uint8_t want_deps = 0;
  if (!req.get(&stream_v) || !req.get(&want_deps)) return CUDA_ERROR_INVALID_VALUE;
  auto stream = reinterpret_cast<CUstream>(stream_v);

  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  const CUgraphNode* deps = nullptr;
  size_t ndeps = 0;
  CUresult r = cuStreamGetCaptureInfo_v2(stream, &status, &id, &graph,
                                         want_deps ? &deps : nullptr, &ndeps);
  if (r != CUDA_SUCCESS) return r;

  rsp->put<int32_t>(static_cast<int32_t>(status));
  rsp->put<uint64_t>(id);
  rsp->put<uint64_t>(reinterpret_cast<uint64_t>(graph));
  rsp->put<uint64_t>(static_cast<uint64_t>(ndeps));
  if (want_deps) {
    rsp->put_sized(deps, deps ? ndeps * sizeof(CUgraphNode) : 0);
  }
  return CUDA_SUCCESS;
}

// cuGraphGetNodes, with the array sized by the caller's capacity. A capacity
// of zero is the documented way to ask how many there are.
CUresult handle_graph_nodes(Buffer& req, Buffer* rsp) {
  uint64_t graph_v = 0, capacity = 0;
  uint8_t want_nodes = 0;
  if (!req.get(&graph_v) || !req.get(&want_nodes) || !req.get(&capacity)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto graph = reinterpret_cast<CUgraph>(graph_v);

  std::vector<CUgraphNode> nodes(want_nodes ? capacity : 0);
  size_t n = nodes.size();
  CUresult r = cuGraphGetNodes(graph, want_nodes ? nodes.data() : nullptr, &n);
  if (r != CUDA_SUCCESS) return r;
  rsp->put<uint64_t>(static_cast<uint64_t>(n));
  if (want_nodes) {
    rsp->put_sized(nodes.data(),
                   (n < nodes.size() ? n : nodes.size()) * sizeof(CUgraphNode));
  }
  return CUDA_SUCCESS;
}

// cuThreadExchangeStreamCaptureMode, whose parameter is both the mode being
// asked for and the mode that was in force. PyTorch uses it to make an
// allocation legal in the middle of a capture, so getting it wrong invalidates
// the capture rather than failing anything obvious.
CUresult handle_capture_mode(Buffer& req, Buffer* rsp) {
  int32_t wanted = 0;
  if (!req.get(&wanted)) return CUDA_ERROR_INVALID_VALUE;
  auto mode = static_cast<CUstreamCaptureMode>(wanted);
  CUresult r = cuThreadExchangeStreamCaptureMode(&mode);
  if (r != CUDA_SUCCESS) return r;
  rsp->put<int32_t>(static_cast<int32_t>(mode));
  return CUDA_SUCCESS;
}

CUresult handle_hello(Buffer& req, Buffer* rsp) {
  (void)req;
  int version = 0;
  cuDriverGetVersion(&version);
  rsp->put<int>(version);
  return CUDA_SUCCESS;
}

bool dispatch_internal(uint32_t id, Buffer& req, Buffer* rsp, CUresult* out) {
  switch (id) {
    case API_rgpu_param_layout: *out = handle_param_layout(req, rsp); return true;
    case API_rgpu_launch: *out = handle_launch(req, rsp); return true;
    case API_rgpu_hello: *out = handle_hello(req, rsp); return true;
    case API_rgpu_capture_info: *out = handle_capture_info(req, rsp); return true;
    case API_rgpu_graph_nodes: *out = handle_graph_nodes(req, rsp); return true;
    case API_rgpu_capture_mode: *out = handle_capture_mode(req, rsp); return true;
    default: return false;
  }
}

// --- connection handling --------------------------------------------------

void serve(int fd) {
  tune_socket(fd);
  // A call sent without expecting a reply has nowhere to report a failure, so
  // we hold the first one and hand it to the next call that does reply. CUDA
  // reports asynchronous failures the same way, at a later call rather than
  // the one that caused them.
  CUresult pending_async = CUDA_SUCCESS;

  // Each connection is one client process. Its CUDA objects live in this
  // server process and die with the connection.
  for (;;) {
    ReqHeader h{};
    std::vector<uint8_t> payload;
    if (!recv_frame(fd, kMagicReq, &h, &payload)) break;

    Buffer req(std::move(payload));
    Buffer rsp;
    CUresult result = CUDA_ERROR_NOT_SUPPORTED;

    const bool handled =
        dispatch_internal(h.api_id, req, &rsp, &result) ||
        (dispatch_cublas && dispatch_cublas(h.api_id, req, &rsp, &result)) ||
        (dispatch_cublaslt &&
         dispatch_cublaslt(h.api_id, req, &rsp, &result)) ||
        (dispatch_cudnn && dispatch_cudnn(h.api_id, req, &rsp, &result)) ||
        dispatch_generated(h.api_id, req, &rsp, &result);
    if (!handled) {
      logf("unknown api id %u (%s)", h.api_id, api_name(h.api_id));
      result = CUDA_ERROR_NOT_SUPPORTED;
    } else if (!req.ok()) {
      // A short read means client and server disagree about the wire layout,
      // which corrupts everything after it. Better to drop the connection.
      logf("malformed request for %s; dropping connection",
           api_name(h.api_id));
      break;
    }

    if (g_verbose) {
      logf("%s -> %d (%zu bytes back)", api_name(h.api_id), result, rsp.size());
    }

    if (h.flags & kFlagNoReply) {
      if (result != CUDA_SUCCESS && pending_async == CUDA_SUCCESS) {
        pending_async = result;
        // Always logged: an application that ignores the next return value
        // would otherwise never learn this happened.
        logf("%s failed with %d and had no reply to report it in; the next "
             "call that replies will carry it", api_name(h.api_id), result);
      }
      continue;
    }

    if (pending_async != CUDA_SUCCESS) {
      result = pending_async;
      pending_async = CUDA_SUCCESS;
    }

    RspHeader rh{};
    rh.magic = kMagicRsp;
    rh.req_id = h.req_id;
    rh.result = static_cast<int32_t>(result);
    rh.payload_len = static_cast<uint32_t>(rsp.size());
    if (!send_frame(fd, rh, rsp)) break;
  }
  ::close(fd);
  logf("client disconnected");
}

}  // namespace
}  // namespace rgpu

int main(int argc, char** argv) {
  int port = 9713;
  const char* env_port = std::getenv("RGPU_PORT");
  if (env_port) port = std::atoi(env_port);
  if (argc > 1) port = std::atoi(argv[1]);
  rgpu::g_verbose = std::getenv("RGPU_VERBOSE") != nullptr;

  // A client that dies mid-write must not take the server with it.
  ::signal(SIGPIPE, SIG_IGN);

  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    rgpu::logf("cuInit failed: %d (no GPU or no driver?)", r);
    return 1;
  }
  int count = 0, version = 0;
  cuDeviceGetCount(&count);
  cuDriverGetVersion(&version);
  rgpu::logf("driver %d, %d device(s)", version, count);
  for (int i = 0; i < count; i++) {
    CUdevice d;
    char name[256] = {0};
    if (cuDeviceGet(&d, i) == CUDA_SUCCESS &&
        cuDeviceGetName(name, sizeof(name), d) == CUDA_SUCCESS) {
      rgpu::logf("  device %d: %s", i, name);
    }
  }

  int srv = ::socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) {
    rgpu::logf("socket: %s", std::strerror(errno));
    return 1;
  }
  int one = 1;
  ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    rgpu::logf("bind %d: %s", port, std::strerror(errno));
    return 1;
  }
  if (::listen(srv, 16) != 0) {
    rgpu::logf("listen: %s", std::strerror(errno));
    return 1;
  }
  // The protocol has no authentication, so it must not face an untrusted
  // network. Anyone who can connect can run arbitrary kernels on this GPU.
  rgpu::logf("listening on port %d (no auth: bind to a trusted network only)",
             port);

  for (;;) {
    int fd = ::accept(srv, nullptr, nullptr);
    if (fd < 0) {
      if (errno == EINTR) continue;
      rgpu::logf("accept: %s", std::strerror(errno));
      break;
    }
    rgpu::logf("client connected");
    std::thread(rgpu::serve, fd).detach();
  }
  return 0;
}
