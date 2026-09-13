// rgpu server: executes forwarded CUDA driver calls against a real GPU.
//
// Runs on the GPU host and links the real libcuda, so the generated dispatch
// can call driver functions by name.

#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <new>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"
#include "server/client_threads.h"
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
  // The mode is the issuing client thread's, kept with its context. See
  // server/client_threads.h.
  CUresult r = client_threads_exchange_capture_mode(&mode);
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

// --- client threads ---------------------------------------------------------
//
// CUDA's current context, and the stack it tops, belong to the calling thread,
// and one server thread serves every thread of a client process. So each
// client thread's stack is kept (server/client_threads.h), its top put back
// before that thread's request runs, and read back from the driver after it.
// See docs/superpowers/specs/2026-09-12-per-thread-cuda-contexts.md.

// How many recently retired threads keep their slot. A thread's own late
// thread-local destructors can still call after the client has announced it
// gone - and the client announces it again after them - so a late call finds
// its thread's context rather than none. A late call from beyond this many
// retirements ago gets an empty slot, which the next notice drops again.
constexpr size_t kRetiredSlots = 64;

// The most client threads a session may have live at once. A client mints
// thread ids itself and the protocol has no authentication, so without a cap
// a client could grow this table without bound. A few thousand is far beyond
// any real process's thread count and still small.
size_t max_client_threads() {
  static const size_t n = [] {
    const char* v = std::getenv("RGPU_MAX_CLIENT_THREADS");
    const long parsed = v ? std::atol(v) : 0;
    return parsed > 0 ? static_cast<size_t>(parsed) : size_t{4096};
  }();
  return n;
}

// The slot for a thread not in the live table: its retired slot if it has one,
// otherwise an empty one. nullptr if the session already has as many live
// threads as it may.
__attribute__((noinline)) ClientThread* admit_thread(ClientThreads& threads,
                                                      uint32_t id,
                                                      uint64_t session) {
  if (threads.live.size() >= max_client_threads()) {
    if (!threads.said_full) {
      threads.said_full = true;
      logf("session %llx: refusing calls from client thread %u: the session "
           "already has %zu live client threads, the most allowed "
           "(RGPU_MAX_CLIENT_THREADS)",
           (unsigned long long)session, id, threads.live.size());
    }
    return nullptr;
  }
  ClientThread slot;
  for (auto it = threads.retired.begin(); it != threads.retired.end(); ++it) {
    if (it->first == id) {
      slot = std::move(it->second);
      threads.retired.erase(it);
      if (g_verbose) {
        // Either announced gone and calling late, or refused at the cap with
        // a failure parked for it (deferred_home) and now let in.
        logf("client thread %u was given back the slot kept for it: it "
             "called after it was announced gone, or was refused earlier at "
             "the cap", id);
      }
      break;
    }
  }
  if (threads.first_id == 0) {
    threads.first_id = id;
  } else if (id != threads.first_id && !threads.said_many) {
    threads.said_many = true;
    logf("session %llx has more than one client thread. Each keeps its own "
         "current context, context stack and stream capture mode, and the "
         "failure of a call it sent without a reply is reported to it alone, "
         "so a multithreaded client is served correctly - but still one call "
         "at a time, not concurrently. Not kept per client thread: which "
         "thread began a stream capture - to the driver every client "
         "thread's capture is this one server thread's own, so a capture not "
         "begun RELAXED may be ended from another client thread, and it "
         "restricts every client thread not in RELAXED mode as its own "
         "capture would, THREAD_LOCAL included - and anything else the "
         "driver holds per thread",
         (unsigned long long)session);
  }
  return &threads.live.emplace(id, slot).first->second;
}

// The slot for the thread that issued a request: one lookup when the thread is
// known, which is every call but a thread's first.
inline ClientThread* thread_slot(ClientThreads& threads, uint32_t id,
                                 uint64_t session) {
  auto it = threads.live.find(id);
  if (it != threads.live.end()) return &it->second;
  return admit_thread(threads, id, session);
}

// Says so when a client thread's slot is forgotten while it still holds the
// failure of a call it sent without a reply: the thread will never make the
// call that would have carried it. This is the last place it is named, but a
// client can have slots forgotten as fast as it sends frames - calls from
// threads refused at the cap each park a failure in a new retired slot, which
// evicts the oldest - so the first in a session is said in full, every one
// with RGPU_VERBOSE, and the rest are counted and the count given at expiry.
//
// CUDA has no exact counterpart to lose. A sticky error belongs to the context,
// not the thread, and survives the thread's exit - every later call in that
// context fails with it, from any thread, and a real driver still does that
// here, on its own. A thread's last error dies with the thread. What is held
// here is neither: it is the ordinary result of one call, which CUDA would have
// returned to the thread at the call, so it goes with its thread.
void say_unreported(ClientThreads& threads, uint64_t session, uint32_t id,
                    const ClientThread& t, const char* why) {
  if (t.pending_async == CUDA_SUCCESS) return;
  if (threads.unreported_failures++ > 0 && !g_verbose) return;
  logf("session %llx: client thread %u %s without learning that a call it "
       "sent without a reply failed with %d; no call from it that replies "
       "came after. %s",
       (unsigned long long)session, id, why, t.pending_async,
       g_verbose ? ""
                 : "Any more in this session are counted, and the count "
                   "given when it expires");
}

// Forgets the oldest retired slots past the bound.
void trim_retired(ClientThreads& threads, uint64_t session) {
  while (threads.retired.size() > kRetiredSlots) {
    say_unreported(threads, session, threads.retired.front().first,
                   threads.retired.front().second, "was forgotten");
    threads.retired.pop_front();
  }
}

// Drops a thread's slot into the retired set, evicting the oldest past the
// bound. Ids are never reused, so nothing but a late call from the same
// thread can find it there.
void retire_thread(ClientThreads& threads, uint32_t id, uint64_t session) {
  auto it = threads.live.find(id);
  if (it == threads.live.end()) return;
  threads.retired.emplace_back(id, std::move(it->second));
  threads.live.erase(it);
  trim_retired(threads, session);
}

// Client threads that have exited. The ids are all read before any is
// dropped, so a malformed notice - caught afterwards like any other malformed
// request - drops nothing.
CUresult handle_thread_gone(Buffer& req, ClientThreads& threads,
                            uint64_t session) {
  uint32_t count = 0;
  if (!req.get(&count)) return CUDA_ERROR_INVALID_VALUE;
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < count; i++) {
    uint32_t id = 0;
    if (!req.get(&id)) return CUDA_ERROR_INVALID_VALUE;
    ids.push_back(id);
  }
  for (uint32_t id : ids) retire_thread(threads, id, session);
  return CUDA_SUCCESS;
}

// Where the deferred failure of a request that was given no slot belongs: a
// thread notice, which runs without one, and a call refused because the
// session was at its cap. Its thread's slot if it has one, live or retired.
// Otherwise, if there is a failure to hold (`make`), a new empty slot in the
// retired set: a thread refused at the cap is let in once there is room, and
// admit_thread gives it its retired slot back, failure and all. nullptr for a
// request naming no thread, which no thread can be told of.
ClientThread* deferred_home(ClientThreads& threads, uint32_t id, bool make,
                            uint64_t session) {
  if (id == 0) return nullptr;
  auto it = threads.live.find(id);
  if (it != threads.live.end()) return &it->second;
  for (auto& entry : threads.retired) {
    if (entry.first == id) return &entry.second;
  }
  if (!make) return nullptr;
  threads.retired.emplace_back(id, ClientThread{});
  // The new slot is the back; trimming takes from the front.
  trim_retired(threads, session);
  return &threads.retired.back().second;
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
  // A connection handed over by a handshake and not yet taken up by the
  // thread serving the session; -1 if there is none. Whoever takes it out of
  // here owns it and closes it, and a handshake that finds one still waiting
  // closes it as it puts its own in: the descriptor a connection had says
  // nothing about which connection it is, because the number is handed out
  // again as soon as it is closed - a reconnect can be given the number of the
  // connection that just ended.
  int pending_fd = -1;
  bool finished = false;       // the serving thread has given up and gone
  // Last request this session actually completed; 0 while it has completed
  // none. Request ids wrap, so compare with req_at_or_before, never with <.
  uint32_t last_req = 0;
  // A call sent without expecting a reply has nowhere to report a failure;
  // the first one is held in its client thread's slot (ClientThread::
  // pending_async) and handed to that thread's next call that replies.
  //
  // The most recent reply, kept in case the connection died between running
  // the call and answering it. Without this the client has a request that was
  // executed and never answered: resending it would run it twice, and not
  // resending it would wait forever.
  uint32_t last_reply_id = 0;  // 0 while there is none
  std::vector<uint8_t> last_reply;
  // Whether the session has already logged a frame naming no request (id 0),
  // and a copy of a request that replies whose reply is no longer kept. Any
  // client can send as many of either as it likes, so each is said once.
  // Touched only by the thread serving the session.
  bool said_no_req_id = false;
  bool said_stale_copy = false;
  // Everything the client asked the driver for and has not given back. It is
  // this process that holds it, so when the session finally expires this is
  // the only record of what to release. See server/inventory.h.
  Inventory inventory;
  // What each client thread has current. Deliberately apart from the
  // inventory: that says what the session owns, this says what each thread
  // was looking at, and it owns nothing. Not under `mu`; see ClientThreads.
  ClientThreads threads;
};

using SessionKey = std::pair<uint64_t, uint64_t>;

std::mutex g_sessions_mu;
std::map<SessionKey, std::shared_ptr<Session>> g_sessions;

// The most sessions the server keeps at once: served, waiting out a grace
// period, or starting. The protocol has no authentication, so without a cap
// any peer could make sessions - each a thread, and whatever it takes from the
// driver - as fast as it can connect. A reconnect to a session the server
// already has is never refused; only a new one is.
size_t max_sessions() {
  static const size_t n = [] {
    const char* v = std::getenv("RGPU_MAX_SESSIONS");
    const long parsed = v ? std::atol(v) : 0;
    return parsed > 0 ? static_cast<size_t>(parsed) : size_t{64};
  }();
  return n;
}

// How long a session waits for its client to come back. Long enough to outlast
// a lost route or a tunnel restarting, short enough that a client that is
// genuinely gone does not hold a GPU forever.
int session_grace_seconds() {
  const char* v = std::getenv("RGPU_SESSION_GRACE");
  int n = v ? std::atoi(v) : 120;
  return n > 0 ? n : 120;
}

// How long the whole handshake read may take. The protocol has no
// authentication, so a peer that connects and then sends nothing - or dribbles
// bytes one at a time - must not be able to tie up an accept thread and an fd
// for good. RGPU_MAX_SESSIONS does not bound this, because no session exists
// until the handshake is in; and tune_socket, which the serving loop relies on,
// sets no receive timeout. So this deadline on the untrusted first bytes is the
// only thing that bounds them.
int handshake_timeout_seconds() {
  const char* v = std::getenv("RGPU_HANDSHAKE_TIMEOUT_SECONDS");
  int n = v ? std::atoi(v) : 10;
  return n > 0 ? n : 10;
}

// Sets, or with 0 clears, a receive timeout on `fd`. A recv that waits longer
// than this returns as if the peer had gone quiet, which is what breaks a
// stalled handshake read out of its wait.
void set_recv_timeout(int fd, int seconds) {
  struct timeval tv{};
  tv.tv_sec = seconds;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

// --- connection handling --------------------------------------------------

void serve_session(std::shared_ptr<Session> session, SessionKey key);

void serve(int fd, const std::shared_ptr<Session>& session, uint64_t key) {
  tune_socket(fd);

  // One connection of a session. What the client made - its CUDA objects,
  // its client threads' slots, a failure held for a thread - belongs to the
  // session, not to this connection, and outlives it: it goes only when the
  // session expires (serve_session).
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

    // At most once. A request that already ran is not run again, whatever
    // connection it arrives on. The client sends again every frame after the
    // last request the handshake says completed, and that is a snapshot: this
    // thread may have been in the middle of the request when the handshake
    // was answered, and has since finished it. So the copy is caught here,
    // where requests run, and not at the handshake.
    //
    // Before anything else looks at the request: not the refusals, not the
    // thread table, not the context restore or readback. A copy changes no
    // client thread's slot and no deferred error; it is only answered.
    //
    // Relies on the client numbering requests in the order it sends them and
    // sending a copy with its original id, which client/rpc.cpp does. Ids
    // wrap, so the order is req_at_or_before's (common/wire.h). A request
    // naming id 0, which is no request, is never run.
    {
      bool already_ran = false;
      bool answer = false;
      uint32_t kept = 0;
      std::vector<uint8_t> cached;
      {
        std::lock_guard<std::mutex> lk(session->mu);
        if (req_at_or_before(h.req_id, session->last_req)) {
          already_ran = true;
          kept = session->last_reply_id;
          if (!(h.flags & kFlagNoReply) && h.req_id == session->last_reply_id &&
              !session->last_reply.empty()) {
            answer = true;
            cached = session->last_reply;
          }
        }
      }
      if (already_ran) {
        if (h.req_id == 0) {
          // Not a copy of anything: a zero-filled or corrupt header, or a
          // client that does not number its requests. Never run.
          if (!session->said_no_req_id) {
            session->said_no_req_id = true;
            logf("session %llx: skipping %s: its request names no request id "
                 "(0); every such frame in this session is skipped, and this "
                 "is said once",
                 (unsigned long long)key, call_name(h.api_id));
          }
        } else if (answer) {
          // Its reply went to a connection that is gone. This is it.
          if (g_verbose) {
            logf("%s (request %u) already ran; sending its reply again",
                 call_name(h.api_id), h.req_id);
          }
          if (!write_exact(fd, cached.data(), cached.size())) break;
        } else if (!(h.flags & kFlagNoReply)) {
          // Should not happen. The client forgets a frame once its reply
          // arrives, and sends nothing behind a call that is waiting for one,
          // so the only request that replies it can send again is the last
          // one - whose reply is the one kept. There is nothing right to
          // answer this with, and running it is what must not happen.
          if (!session->said_stale_copy || g_verbose) {
            session->said_stale_copy = true;
            logf("session %llx: request %u (%s) already ran and its reply is "
                 "no longer kept (the last reply kept is for request %u); not "
                 "running it again and not answering it. Said once per "
                 "session unless RGPU_VERBOSE is set",
                 (unsigned long long)key, h.req_id, call_name(h.api_id),
                 kept);
          }
        } else if (g_verbose) {
          logf("%s (request %u, no reply) already ran; skipping it",
               call_name(h.api_id), h.req_id);
        }
        continue;
      }
    }

    Buffer req(std::move(payload));
    Buffer rsp;
    CUresult result = CUDA_ERROR_NOT_SUPPORTED;
    ClientThreads& threads = session->threads;

    // Whether the request was looked at at all. One that is refused before
    // dispatch has an error for its answer and nothing to check.
    bool handled = false;
    bool refused = false;
    ClientThread* thread = nullptr;
    threads.caller = nullptr;
    if (h.thread_id == 0) {
      // Never a real thread: a zero-filled or corrupt header. Attributing it
      // to some thread's context would be guessing.
      refused = true;
      result = CUDA_ERROR_INVALID_VALUE;
      if (!threads.said_zero) {
        threads.said_zero = true;
        logf("session %llx: refusing %s: its request names no client thread. "
             "Every such request in this session is refused, one sent "
             "without a reply has its failure reported to no thread, and "
             "this is said once",
             (unsigned long long)key, call_name(h.api_id));
      }
    } else if (h.api_id == API_rgpu_thread_gone) {
      // Changes the table rather than running under it. It needs no context,
      // and must not need a slot: the thread sending it may be new, and a
      // session at its cap needs the notice most of all.
      handled = true;
      result = handle_thread_gone(req, threads, key);
    } else {
      thread = thread_slot(threads, h.thread_id, key);
      if (!thread) {
        refused = true;
        result = CUDA_ERROR_INVALID_VALUE;
      } else if (!client_thread_shown(*thread, threads)) {
        // Only when the thread's context or capture mode is not already the
        // serving thread's, so a client with one thread - or a burst of calls
        // from one thread, or threads sharing a context and a mode - never
        // pays for a switch.
        result = client_thread_show(threads, *thread);
        refused = result != CUDA_SUCCESS;
      }
    }

    if (!refused && !handled) {
      // The calls that select, push, pop, create or destroy a context act on
      // this thread's stack. See server/client_threads.h.
      threads.caller = thread;
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
      threads.caller = nullptr;
      // What the call left current, as the driver says, rather than only what
      // the intercepted calls were asked for: a call nobody listed can change
      // it too.
      client_thread_after(threads, *thread, h.api_id, &result);
    }
    if (refused) {
      // Answered below with the error; the payload was never read.
    } else if (!handled) {
      logf("unknown api id %u (%s)", h.api_id, call_name(h.api_id));
      result = CUDA_ERROR_NOT_SUPPORTED;
    }
    // A short read means client and server disagree about the wire layout,
    // which corrupts everything after it, so the connection is dropped. But
    // only after the request is recorded as completed, with an error, like
    // any other: it was dispatched, and the handler may already have called
    // the driver before it found the arguments short. A request not recorded
    // would be sent again by the client after it reconnects, and run again.
    const bool malformed = !refused && handled && !req.ok();
    if (malformed) {
      logf("malformed request for %s; answering it with an error and dropping "
           "the connection", call_name(h.api_id));
      result = CUDA_ERROR_INVALID_VALUE;
      rsp = Buffer();
    }

    if (g_verbose) {
      logf("%s -> %d (%zu bytes back)", call_name(h.api_id), result, rsp.size());
    }

    // The slot a failure of this request is held in, or a held one taken
    // from: the issuing thread's. Found before the lock, because finding it
    // for a request that was given no slot can make one; used only under it.
    // Not looked for at all for a call without a reply that succeeded, which
    // has nothing to hold and nothing to take.
    const bool no_reply = (h.flags & kFlagNoReply) != 0;
    ClientThread* home = thread;
    if (!home && (!no_reply || result != CUDA_SUCCESS)) {
      home = deferred_home(threads, h.thread_id, no_reply, key);
    }

    if (no_reply) {
      bool held = false;
      {
        std::lock_guard<std::mutex> lk(session->mu);
        // Completed, and its failure held, in one step.
        session->last_req = h.req_id;
        if (result != CUDA_SUCCESS && home &&
            home->pending_async == CUDA_SUCCESS) {
          home->pending_async = result;
          held = true;
        }
      }
      if (held) {
        // Logged, because an application that ignores the next return value
        // would otherwise never learn it happened at all. But a client can
        // make these as fast as it sends frames, so the first in a session is
        // said in full, every one with RGPU_VERBOSE, and the rest counted
        // and the count given at expiry.
        if (threads.held_failures++ == 0 || g_verbose) {
          logf("%s from client thread %u failed with %d and had no reply to "
               "report it in; that thread's next call that replies will carry "
               "it. %s",
               call_name(h.api_id), h.thread_id, result,
               g_verbose ? ""
                         : "Any more in this session are counted, and the "
                           "count given when it expires");
        }
      } else if (result != CUDA_SUCCESS && g_verbose) {
        // One slot holds one error, so everything that fails behind the first
        // one is dropped. CUDA's own sticky error behaves the same way, and
        // the first is the one worth having, but a thread that keeps failing
        // looks silent from the outside unless we say so here.
        logf("%s from client thread %u also failed with %d, %s",
             call_name(h.api_id), h.thread_id, result,
             home ? "behind an error already waiting"
                  : "and names no thread to report it to");
      }
      if (malformed) break;
      continue;
    }

    RspHeader rh{};
    rh.magic = kMagicRsp;
    rh.req_id = h.req_id;
    rh.payload_len = static_cast<uint32_t>(rsp.size());
    {
      std::lock_guard<std::mutex> lk(session->mu);
      // Only a call that succeeded on its own can carry an earlier error.
      // Handing the older one to a call that just failed would report the
      // wrong failure and lose the real one, which is the opposite of the
      // point: the deferred error stays held for the thread's next call that
      // succeeds.
      if (result == CUDA_SUCCESS && home &&
          home->pending_async != CUDA_SUCCESS) {
        result = home->pending_async;
        home->pending_async = CUDA_SUCCESS;
      }
      rh.result = static_cast<int32_t>(result);
      // Completed and its reply kept in one step. A handshake that saw the
      // one without the other would tell the client this request is done and
      // not resend its reply, and the client would wait for it forever.
      const auto* rb = reinterpret_cast<const uint8_t*>(&rh);
      session->last_req = h.req_id;
      session->last_reply_id = h.req_id;
      session->last_reply.assign(rb, rb + sizeof(rh));
      session->last_reply.insert(session->last_reply.end(), rsp.data().begin(),
                                 rsp.data().end());
    }
    if (!send_frame(fd, rh, rsp) || malformed) break;
  }
  ::close(fd);
  // With the descriptor, because a reconnect can be given the same number,
  // and a session must not mistake it for the connection that just closed.
  logf("connection closed (descriptor %d)", fd);
}

// Serves one session across however many connections it takes. Between them
// the thread waits, holding everything the client owns.
void serve_session(std::shared_ptr<Session> session, SessionKey key) {
  // Everything this thread does from here belongs to this session, including
  // the release at the end.
  inventory_bind(&session->inventory);
  client_threads_bind(&session->threads);
  for (;;) {
    int fd = -1;
    {
      std::unique_lock<std::mutex> lk(session->mu);
      if (session->pending_fd < 0) {
        const auto grace = std::chrono::seconds(session_grace_seconds());
        if (!session->cv.wait_for(lk, grace,
                                  [&] { return session->pending_fd >= 0; })) {
          // Nobody came back. Everything this session holds goes with it.
          session->finished = true;
          break;
        }
      }
      // Taken, not borrowed: from here this thread owns the connection and
      // closes it, and nothing else can close it behind its back.
      fd = session->pending_fd;
      session->pending_fd = -1;
    }

    serve(fd, session, key.first);

    // A test hook, off by default: holds this thread back between a
    // connection closing and its looking for the next, so that a test can
    // reconnect in that gap.
    static const int reconnect_gap_ms = [] {
      const char* v = std::getenv("RGPU_TEST_RECONNECT_GAP_MS");
      return v ? std::atoi(v) : 0;
    }();
    if (reconnect_gap_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_gap_ms));
    }

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

  // Forgotten first, and only forgotten. The slots name contexts the
  // inventory owns and is about to release, and the release makes contexts
  // current as it goes, so a slot that outlived this line would name a
  // context that is gone. Nothing is ever released from the slots.
  client_threads_bind(nullptr);
  ClientThreads& threads = session->threads;
  for (const auto& entry : threads.live) {
    say_unreported(threads, key.first, entry.first, entry.second, "expired");
  }
  for (const auto& entry : threads.retired) {
    say_unreported(threads, key.first, entry.first, entry.second, "expired");
  }
  // What was said once above, in numbers.
  char failures[256] = "";
  if (threads.held_failures > 0) {
    std::snprintf(failures, sizeof(failures),
                  "; %llu failure(s) of calls sent without a reply were held, "
                  "%llu of them never reached their client thread",
                  (unsigned long long)threads.held_failures,
                  (unsigned long long)threads.unreported_failures);
  }
  threads.clear();

  // The client is gone but its GPU resources are not: they were created in
  // this process and nothing else will ever free them. This is the only point
  // at which that can be put right, and it has to happen on this thread,
  // because the contexts they live in are this thread's.
  //
  // In RELAXED capture mode, not whatever mode the last client thread served
  // left this thread in: RELAXED is immune to the restrictions another
  // session's live GLOBAL capture would put on this thread's frees. The release
  // then ends any capture this session left open before it frees anything
  // (release_inventory).
  client_threads_relax_capture_mode(threads);
  const std::string summary = release_inventory(session->inventory);
  inventory_bind(nullptr);
  logf("session %llx expired; %s%s", (unsigned long long)key.first,
       summary.c_str(), failures);
}

// Tells a client that the session it is coming back to is not here, and closes
// the connection. Not resumed, and nothing completed: the client, which has
// talked to a session under this id before, takes that to mean the session is
// gone and stops. No session is made.
void session_gone(int fd, uint64_t session, uint32_t last_req) {
  if (last_req != 0) {
    logf("session %llx is not here: its client has had replies up to request "
         "%u, so it is coming back to a session that expired or belonged to a "
         "server that restarted. Telling it the session is gone",
         (unsigned long long)session, last_req);
  } else {
    logf("session %llx is not here: its client says it has had a session "
         "under this id, so it is coming back to one that expired or belonged "
         "to a server that restarted. Telling it the session is gone",
         (unsigned long long)session);
  }
  HandshakeReply gone{};
  gone.magic = kMagicHello;
  gone.version = kProtocolVersion;
  write_exact(fd, &gone, sizeof(gone));
  ::close(fd);
}

// Tells a client that the server will not start a session for it, because it
// already has as many as it allows, and closes the connection. No session is
// made.
void session_refused(int fd, uint64_t session, size_t live) {
  logf("session %llx: refusing a new session: the server already has %zu, the "
       "most allowed (RGPU_MAX_SESSIONS)",
       (unsigned long long)session, live);
  HandshakeReply busy{};
  busy.magic = kMagicBusy;
  busy.version = kProtocolVersion;
  write_exact(fd, &busy, sizeof(busy));
  ::close(fd);
}

// A new session that was registered but will never be served: its handshake
// reply could not be written, so no thread was started for it. Left in the
// table it would never expire, and a client coming back under its id would be
// told it resumed and then wait forever on a connection nobody reads. So it is
// taken out, and marked finished for a reconnect that found it in the
// meantime, whose hand-over then closes that connection.
void forget_unserved(const std::shared_ptr<Session>& session,
                     const SessionKey& key) {
  logf("session %llx: could not answer its handshake; forgetting the session, "
       "which was never served",
       (unsigned long long)key.first);
  std::lock_guard<std::mutex> lk(g_sessions_mu);
  auto it = g_sessions.find(key);
  if (it != g_sessions.end() && it->second == session) g_sessions.erase(it);
  std::lock_guard<std::mutex> slk(session->mu);
  session->finished = true;
  // A reconnect already handed over, which nothing will read.
  if (session->pending_fd >= 0) {
    ::close(session->pending_fd);
    session->pending_fd = -1;
  }
}

// Reads the handshake and either starts a session or hands the connection to
// the thread already serving one.
void accept_connection(int fd) {
  // Bound the untrusted first bytes: a peer that sends nothing, or dribbles,
  // gets the connection closed at the deadline rather than holding this thread
  // and this fd forever. Nothing is registered on the way out.
  set_recv_timeout(fd, handshake_timeout_seconds());
  Handshake hello{};
  if (!read_exact(fd, &hello, sizeof(hello)) || hello.magic != kMagicHello) {
    logf("connection did not begin with a handshake within the deadline, or "
         "with the wrong bytes; closing");
    ::close(fd);
    return;
  }
  // The handshake is in. Restore the normal blocking socket: the serving loop's
  // reads wait on a live client with no deadline, and tune_socket does not
  // touch the receive timeout, so clearing it here is what keeps a long idle
  // session from being cut off at the handshake deadline.
  set_recv_timeout(fd, 0);
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
  size_t refused_at = 0;  // the session count a new session was refused at
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
    // A client that has already had replies, or says it has had a session
    // at all, is not starting: it is coming back to a session this server no
    // longer has - it expired, or the server restarted - and every handle it
    // holds names nothing here. An empty session made for it would be found
    // by its next attempt and reported as resumed, and the client would send
    // its unacknowledged calls into it; one that never had a reply would
    // leave the empty session waiting out its grace period for nothing. A
    // client starting afresh always says 0 and does not say it is resuming.
    const bool starting =
        hello.last_req_id == 0 && !(hello.flags & kHelloResuming);
    if (!session && starting) {
      // A finished session still under this key is on its way out, and the
      // new one takes its place rather than adding to the count.
      if (it == g_sessions.end() && g_sessions.size() >= max_sessions()) {
        refused_at = g_sessions.size();
      } else {
        session = std::make_shared<Session>();
        g_sessions[key] = session;
      }
    }
  }
  if (refused_at > 0) {
    session_refused(fd, key.first, refused_at);
    return;
  }
  if (!session) {
    session_gone(fd, key.first, hello.last_req_id);
    return;
  }

  HandshakeReply reply{};
  reply.magic = kMagicHello;
  reply.version = kProtocolVersion;
  // What the session has completed, and the reply the client never received,
  // read together. The count is only advice to the client about what to send
  // again - a request can still be running as it is read, and the serving
  // loop is what makes sure no copy runs twice - but the two have to agree:
  // a count that includes a request whose reply is not resent would leave the
  // client waiting for that reply forever, and a reply resent for a request
  // the count leaves out would be answered again when its copy arrives.
  std::vector<uint8_t> unanswered;
  uint32_t unanswered_id = 0;
  {
    std::lock_guard<std::mutex> lk(session->mu);
    reply.resumed = resumed ? 1u : 0u;
    reply.last_req_id = session->last_req;
    // A reply the client never received. It is safe to send again and it is
    // the only way that request can be answered, because running it a second
    // time would not be the same thing.
    if (resumed &&
        !req_at_or_before(session->last_reply_id, hello.last_req_id) &&
        !session->last_reply.empty()) {
      unanswered = session->last_reply;
      unanswered_id = session->last_reply_id;
    }
  }
  if (!resumed) {
    // A test hook, off by default: holds a new session's handshake reply
    // back, so that a test can be gone before it is written.
    static const int reply_delay_ms = [] {
      const char* v = std::getenv("RGPU_TEST_HANDSHAKE_DELAY_MS");
      return v ? std::atoi(v) : 0;
    }();
    if (reply_delay_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(reply_delay_ms));
    }
  }
  if (!write_exact(fd, &reply, sizeof(reply))) {
    if (!resumed) forget_unserved(session, key);
    ::close(fd);
    return;
  }

  if (resumed) {
    logf("session %llx resumed; the last request it completed was %u",
         (unsigned long long)key.first, reply.last_req_id);
    if (!unanswered.empty()) {
      if (!write_exact(fd, unanswered.data(), unanswered.size())) {
        ::close(fd);
        return;
      }
      logf("  resent the reply to request %u", unanswered_id);
    }
    // A test hook, off by default: holds the hand-over back, so that a test
    // can have the session's grace period run out in the middle of it.
    static const int handoff_delay_ms = [] {
      const char* v = std::getenv("RGPU_TEST_HANDOFF_DELAY_MS");
      return v ? std::atoi(v) : 0;
    }();
    if (handoff_delay_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(handoff_delay_ms));
    }
    // The session was live when it was looked up, but its grace period may
    // have run out since: the check above and this hand-over are separate
    // holds of the lock, with network writes between them. A connection handed
    // to a session that has finished is never read, and the client, told it
    // resumed, would wait on it forever. So the check is made again in the same
    // hold as the hand-over, and a finished session gets the connection
    // closed: the client comes back, and is told the session is gone.
    {
      std::lock_guard<std::mutex> lk(session->mu);
      if (!session->finished) {
        // A connection handed over before this one and not yet taken up is
        // this one's predecessor: its client is the same client, and it is
        // here again. Closed rather than dropped, so that a client waiting on
        // it learns at once instead of waiting for a reply nobody will send.
        if (session->pending_fd >= 0) {
          logf("session %llx: closing the connection handed over before this "
               "one, which its serving thread had not taken up yet",
               (unsigned long long)key.first);
          ::close(session->pending_fd);
        }
        session->pending_fd = fd;
        session->cv.notify_all();
        logf("  handed it over on descriptor %d", fd);
        return;
      }
    }
    logf("session %llx expired while a connection to it was being handed "
         "over; closing that connection",
         (unsigned long long)key.first);
    ::close(fd);
    return;
  }

  logf("session %llx started", (unsigned long long)key.first);
  {
    std::lock_guard<std::mutex> lk(session->mu);
    // A racing second connection with the same key can be treated as a resume
    // of this just-created session and hand its own connection over first.
    // Overwriting pending_fd would leak that fd and leave its client waiting
    // forever on a connection nobody reads, so it is closed first - the same
    // close-if-present guard the resumed hand-over above uses.
    if (session->pending_fd >= 0) {
      logf("session %llx: closing a connection handed over before this one "
           "began serving it",
           (unsigned long long)key.first);
      ::close(session->pending_fd);
    }
    session->pending_fd = fd;
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
