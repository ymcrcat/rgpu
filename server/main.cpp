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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda.h>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"
#include "server/inventory.h"

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

  // The capacity is the client's word, so it is not what gets allocated:
  // a huge one used to throw here and take every session down with it. The
  // graph's real node count bounds it.
  constexpr uint64_t kMaxGraphNodes = 1ull << 24;
  if (capacity > kMaxGraphNodes) return CUDA_ERROR_INVALID_VALUE;
  if (want_nodes && capacity > 0) {
    size_t real = 0;
    CUresult r = cuGraphGetNodes(graph, nullptr, &real);
    if (r != CUDA_SUCCESS) return r;
    if (capacity > real) capacity = real;
  }

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

// --- sessions -------------------------------------------------------------
//
// A session is a client process; a connection is one attempt by it to reach
// us. They are not the same thing, and the difference is what makes a dropped
// connection survivable: when a connection goes, the session waits rather than
// dying, and the next connection carrying the same id is handed to the very
// thread that was serving it.
//
// It has to be the same thread. The current CUDA context is thread state, and
// so is the map of the client's minted cuDNN handles. Handing the socket back
// to the thread that owns them means everything the client is holding - device
// pointers, modules, descriptors - is still valid, because nothing was ever
// torn down.

struct Session {
  std::mutex mu;
  std::condition_variable cv;
  int fd = -1;                 // the current connection, -1 while waiting
  bool finished = false;       // the serving thread has given up and gone
  uint32_t last_req = 0;       // last request this session actually completed
  // A call sent without expecting a reply has nowhere to report a failure, so
  // we hold the first one and hand it to the next call that does reply. CUDA
  // reports asynchronous failures the same way, at a later call rather than
  // the one that caused them.
  //
  // It belongs to the session and not to the connection. A dropped connection
  // is not an acknowledgement: the client was told the call completed, so it
  // will not send it again, and an error left behind on the old connection
  // would turn a failed launch into an apparent success.
  CUresult pending_async = CUDA_SUCCESS;
  // The most recent reply, kept in case the connection died between running
  // the call and answering it. Without this the client has a request that was
  // executed and never answered: resending it would run it twice, and not
  // resending it would wait forever.
  uint32_t last_reply_id = 0;
  std::vector<uint8_t> last_reply;
  // Everything the client asked the driver for and has not given back. It is
  // this process that holds it, so when the session finally expires this is
  // the only record of what to release. See server/inventory.h.
  Inventory inventory;
};

using SessionKey = std::pair<uint64_t, uint64_t>;

std::mutex g_sessions_mu;
std::map<SessionKey, std::shared_ptr<Session>> g_sessions;

// How long a session waits for its client to come back. Long enough to outlast
// a lost route or a tunnel restarting, short enough that a client that is
// genuinely gone does not hold a GPU forever.
int session_grace_seconds() {
  const char* v = std::getenv("RGPU_SESSION_GRACE");
  int n = v ? std::atoi(v) : 120;
  return n > 0 ? n : 120;
}

// --- connection handling --------------------------------------------------

void serve_session(std::shared_ptr<Session> session, SessionKey key);

void serve(int fd, const std::shared_ptr<Session>& session) {
  tune_socket(fd);

  // Each connection is one client process. Its CUDA objects live in this
  // server process and die with the connection.
  for (;;) {
    ReqHeader h{};
    std::vector<uint8_t> payload;
    // The header's length is the client's word too; a frame declaring more
    // than can be allocated ends this connection, not the server.
    bool received = false;
    try {
      received = recv_frame(fd, kMagicReq, &h, &payload);
    } catch (const std::bad_alloc&) {
      logf("a frame declared more bytes than could be allocated; closing it");
    }
    if (!received) break;

    // A deliberate break, for the test that proves a real one is survivable.
    // Counted in frames read, so it lands in the middle of a conversation.
    static const int drop_after = [] {
      const char* v = std::getenv("RGPU_DROP_AFTER");
      return v ? std::atoi(v) : 0;
    }();
    static std::atomic<int> frames_seen{0};
    if (drop_after > 0 && frames_seen.fetch_add(1) + 1 == drop_after) {
      logf("dropping the connection after %d frames (RGPU_DROP_AFTER)",
           drop_after);
      break;
    }

    Buffer req(std::move(payload));
    Buffer rsp;
    CUresult result = CUDA_ERROR_NOT_SUPPORTED;

    bool handled = false;
    try {
      handled =
          dispatch_internal(h.api_id, req, &rsp, &result) ||
          (dispatch_cublas && dispatch_cublas(h.api_id, req, &rsp, &result)) ||
          (dispatch_cublaslt &&
           dispatch_cublaslt(h.api_id, req, &rsp, &result)) ||
          (dispatch_cudnn && dispatch_cudnn(h.api_id, req, &rsp, &result)) ||
          dispatch_generated(h.api_id, req, &rsp, &result);
    } catch (const std::bad_alloc&) {
      // A request asking for more memory than there is. An error for this
      // call, not a reason to end the process and every session in it.
      handled = true;
      result = CUDA_ERROR_OUT_OF_MEMORY;
      rsp = Buffer();
      logf("%s asked for more memory than could be allocated", call_name(h.api_id));
    } catch (const std::exception& e) {
      handled = true;
      result = CUDA_ERROR_UNKNOWN;
      rsp = Buffer();
      logf("%s failed: %s", call_name(h.api_id), e.what());
    }
    if (!handled) {
      logf("unknown api id %u (%s)", h.api_id, call_name(h.api_id));
      result = CUDA_ERROR_NOT_SUPPORTED;
    } else if (!req.ok()) {
      // A short read means client and server disagree about the wire layout,
      // which corrupts everything after it. Better to drop the connection.
      logf("malformed request for %s; dropping connection",
           call_name(h.api_id));
      break;
    }

    if (g_verbose) {
      logf("%s -> %d (%zu bytes back)", call_name(h.api_id), result, rsp.size());
    }

    if (h.flags & kFlagNoReply) {
      bool held = false;
      {
        std::lock_guard<std::mutex> lk(session->mu);
        session->last_req = h.req_id;
        if (result != CUDA_SUCCESS && session->pending_async == CUDA_SUCCESS) {
          session->pending_async = result;
          held = true;
        }
      }
      if (held) {
        // Always logged: this is the only place the failure is named, and an
        // application that ignores the next return value would otherwise never
        // learn it happened at all.
        logf("%s failed with %d and had no reply to report it in; the next "
             "call that replies will carry it", call_name(h.api_id), result);
      } else if (result != CUDA_SUCCESS && g_verbose) {
        // One slot holds one error, so everything that fails behind the first
        // one is dropped. CUDA's own sticky error behaves the same way, and
        // the first is the one worth having, but a session that keeps failing
        // looks silent from the outside unless we say so here.
        logf("%s also failed with %d, behind an error already waiting",
             call_name(h.api_id), result);
      }
      continue;
    }

    {
      std::lock_guard<std::mutex> lk(session->mu);
      // Only a call that succeeded on its own can carry someone else's error.
      // Handing the older one to a call that just failed would report the
      // wrong failure and lose the real one, which is the opposite of the
      // point: the deferred error stays held for the next call that succeeds.
      if (result == CUDA_SUCCESS && session->pending_async != CUDA_SUCCESS) {
        result = session->pending_async;
        session->pending_async = CUDA_SUCCESS;
      }
      session->last_req = h.req_id;
    }

    RspHeader rh{};
    rh.magic = kMagicRsp;
    rh.req_id = h.req_id;
    rh.result = static_cast<int32_t>(result);
    rh.payload_len = static_cast<uint32_t>(rsp.size());
    {
      const auto* rb = reinterpret_cast<const uint8_t*>(&rh);
      std::lock_guard<std::mutex> lk(session->mu);
      session->last_reply_id = h.req_id;
      session->last_reply.assign(rb, rb + sizeof(rh));
      session->last_reply.insert(session->last_reply.end(), rsp.data().begin(),
                                 rsp.data().end());
    }
    if (!send_frame(fd, rh, rsp)) break;
  }
  ::close(fd);
  logf("connection closed");
}

// Serves one session across however many connections it takes. Between them
// the thread waits, holding everything the client owns.
void serve_session(std::shared_ptr<Session> session, SessionKey key) {
  // Everything this thread does from here belongs to this session, including
  // the release at the end.
  inventory_bind(&session->inventory);
  for (;;) {
    int fd = -1;
    {
      std::unique_lock<std::mutex> lk(session->mu);
      if (session->fd < 0) {
        const auto grace = std::chrono::seconds(session_grace_seconds());
        if (!session->cv.wait_for(lk, grace,
                                  [&] { return session->fd >= 0; })) {
          // Nobody came back. Everything this session holds goes with it.
          session->finished = true;
          break;
        }
      }
      fd = session->fd;
    }

    serve(fd, session);

    std::lock_guard<std::mutex> lk(session->mu);
    if (session->fd == fd) session->fd = -1;
    logf("session %llx waiting up to %ds for the client to come back",
         (unsigned long long)key.first, session_grace_seconds());
  }

  {
    std::lock_guard<std::mutex> lk(g_sessions_mu);
    // Only if it is still this session under that key. A client that
    // reconnected after we gave up has a new session there, and erasing it
    // would lose a live one.
    auto it = g_sessions.find(key);
    if (it != g_sessions.end() && it->second == session) g_sessions.erase(it);
  }

  // The client is gone but its GPU resources are not: they were created in
  // this process and nothing else will ever free them. This is the only point
  // at which that can be put right, and it has to happen on this thread,
  // because the contexts they live in are this thread's.
  const std::string summary = release_inventory(session->inventory);
  inventory_bind(nullptr);
  logf("session %llx expired; %s", (unsigned long long)key.first,
       summary.c_str());
}

// Reads the handshake and either starts a session or hands the connection to
// the thread already serving one.
void accept_connection(int fd) {
  Handshake hello{};
  if (!read_exact(fd, &hello, sizeof(hello)) || hello.magic != kMagicHello) {
    logf("connection did not begin with a handshake; closing");
    ::close(fd);
    return;
  }
  if (hello.version != kProtocolVersion) {
    logf("client speaks protocol %u, this server speaks %u; closing",
         hello.version, kProtocolVersion);
    // Say which protocol this server speaks before closing. A client that
    // only sees the connection close can report nothing more precise than a
    // failed handshake; one that reads this reply can name both versions. The
    // handshake has the same layout in every version, so any client can read
    // it. Nothing else happens: no session is made for a client that cannot
    // be served.
    HandshakeReply refusal{};
    refusal.magic = kMagicHello;
    refusal.version = kProtocolVersion;
    write_exact(fd, &refusal, sizeof(refusal));
    ::close(fd);
    return;
  }

  const SessionKey key{hello.session_hi, hello.session_lo};
  std::shared_ptr<Session> session;
  bool resumed = false;
  {
    std::lock_guard<std::mutex> lk(g_sessions_mu);
    auto it = g_sessions.find(key);
    if (it != g_sessions.end()) {
      std::lock_guard<std::mutex> slk(it->second->mu);
      if (!it->second->finished) {
        session = it->second;
        resumed = true;
      }
    }
    if (!session) {
      session = std::make_shared<Session>();
      g_sessions[key] = session;
    }
  }

  HandshakeReply reply{};
  reply.magic = kMagicHello;
  reply.version = kProtocolVersion;
  {
    std::lock_guard<std::mutex> lk(session->mu);
    reply.resumed = resumed ? 1u : 0u;
    reply.last_req_id = session->last_req;
  }
  if (!write_exact(fd, &reply, sizeof(reply))) {
    ::close(fd);
    return;
  }

  if (resumed) {
    logf("session %llx resumed; %u requests completed before the break",
         (unsigned long long)key.first, reply.last_req_id);
    std::lock_guard<std::mutex> lk(session->mu);
    // A reply the client never received. It is safe to send again and it is
    // the only way that request can be answered, because running it a second
    // time would not be the same thing.
    if (session->last_reply_id > hello.last_req_id &&
        !session->last_reply.empty()) {
      if (!write_exact(fd, session->last_reply.data(),
                       session->last_reply.size())) {
        ::close(fd);
        return;
      }
      logf("  resent the reply to request %u", session->last_reply_id);
    }
    session->fd = fd;
    session->cv.notify_all();
    return;
  }

  logf("session %llx started", (unsigned long long)key.first);
  {
    std::lock_guard<std::mutex> lk(session->mu);
    session->fd = fd;
  }
  std::thread(serve_session, session, key).detach();
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
    std::thread(rgpu::accept_connection, fd).detach();
  }
  return 0;
}
