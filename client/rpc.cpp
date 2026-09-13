#include "client/rpc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <chrono>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <set>
#include <string>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"

namespace rgpu {
namespace {

// Guards the socket, the counter and the send queue. Never destroyed: threads
// keep calling in while the process exits, and locking a mutex that static
// destruction has already taken down is undefined (on libc++ it throws, from
// places that cannot).
std::mutex& g_mu = *new std::mutex();
int g_fd = -1;
uint32_t g_next_req = 1;
bool g_connect_failed = false;
uint32_t g_last_reply = 0;   // last request id we have seen a reply for
bool g_had_session = false;  // we have talked to this server before

// Frames queued by call_async and not yet written. Holding them lets a run of
// launches leave the client in one write instead of one per call, and lets
// none of them wait for a reply.
std::vector<uint8_t> g_queued;

// What the remoting layer actually cost, printed at exit when RGPU_STATS is
// set. Round trips are the number that matters: on a real network each one
// costs a full latency, so round trips per inference times the link's RTT is
// the added time, without having to run over that link to find out.
// Deliberately never destroyed: PyTorch's threads keep calling into the shim
// while the process is exiting, and a std::map that has already run its
// destructor is heap corruption rather than a wrong count. Printed from an
// atexit handler instead, which runs before static destruction.
struct Stats {
  uint64_t round_trips = 0;
  uint64_t async_calls = 0;
  uint64_t bytes_out = 0;
  uint64_t bytes_in = 0;
  // Which calls the round trips went to, so the ones worth making
  // asynchronous can be picked by evidence rather than by guess.
  std::map<uint32_t, uint64_t> by_api;

  void report() {
    std::fprintf(stderr,
                 "[rgpu] %llu round trips, %llu one-way calls, "
                 "%.1f MiB out, %.1f MiB in\n",
                 (unsigned long long)round_trips, (unsigned long long)async_calls,
                 bytes_out / 1048576.0, bytes_in / 1048576.0);
    std::vector<std::pair<uint64_t, uint32_t>> top;
    for (const auto& kv : by_api) top.emplace_back(kv.second, kv.first);
    std::sort(top.rbegin(), top.rend());
    for (size_t i = 0; i < top.size() && i < 15; i++) {
      std::fprintf(stderr, "[rgpu]   %8llu  %s (0x%x)\n",
                   (unsigned long long)top[i].first, api_name(top[i].second),
                   top[i].second);
    }
  }
};
Stats& g_stats = *new Stats();

// Registered on first use rather than unconditionally, so a process that never
// talks to us installs nothing.
void arm_stats_report() {
  if (!std::getenv("RGPU_STATS")) return;
  static std::once_flag once;
  std::call_once(once, [] { std::atexit([] { g_stats.report(); }); });
}

// Flushed automatically once the queue reaches this size, so a long stretch of
// asynchronous work cannot grow it without bound. Otherwise it goes out with
// the next call that needs a reply.
constexpr size_t kQueueFlushBytes = 256 * 1024;

int env_int(const char* k, int dflt) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : dflt;
}

bool verbose() {
  static bool v = env_int("RGPU_VERBOSE", 0) != 0;
  return v;
}

// This client process, as the server knows it. A connection is one attempt to
// reach the server; the session is what holds the GPU state, and it outlives
// any particular connection.
struct SessionId {
  uint64_t hi = 0;
  uint64_t lo = 0;
};

SessionId make_session_id() {
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> dist;
  std::mt19937_64 gen(((uint64_t)rd() << 32) ^ rd() ^
                      (uint64_t)::getpid() ^
                      (uint64_t)std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count());
  return SessionId{dist(gen), dist(gen)};
}

const SessionId g_session = make_session_id();

// --- client threads ---------------------------------------------------------
//
// Every request says which client thread issued it, so that the server can
// keep CUDA's per-thread state - the current context above all - for each one.
// The id is ours rather than the OS's: minted from 1 in the order threads first
// call, not reused (a recycled OS tid would inherit a dead thread's context),
// and the same size on every platform.

std::atomic<uint32_t> g_next_thread{1};
thread_local uint32_t t_thread_id = 0;  // 0 until this thread first calls
// Set once this thread's id has been retired. It can still call after that:
// see queue_frame_locked.
thread_local bool t_retired = false;

// Ids of threads that have exited, for the server to forget. Never destroyed,
// for the same reason as the stats: threads keep exiting while the process
// does. Its own lock, so a thread can exit without waiting behind a call that
// is blocked on the network, and a flag so that the common case - nothing to
// report - costs a load rather than a lock.
std::mutex& g_retired_mu = *new std::mutex();
std::vector<uint32_t>& g_retired = *new std::vector<uint32_t>();
std::atomic<bool> g_have_retired{false};

// Gives the thread's id back when the thread exits. Only records it: no I/O
// in a thread-exit path, which can run while the process is being torn down.
struct ThreadRetirer {
  ~ThreadRetirer() {
    std::lock_guard<std::mutex> lk(g_retired_mu);
    g_retired.push_back(t_thread_id);
    g_have_retired.store(true, std::memory_order_relaxed);
    t_retired = true;
  }
};

// ponytail: the counter is 32 bits and is not reclaimed. After about 4.3
// billion threads that ever called, it wraps and hands out ids that live
// threads - the main thread's 1 among them - still hold, and two threads would
// share one context on the server. Skipping zero only keeps "no thread"
// meaningful; it does not make the wrap safe. No process gets near it.
__attribute__((noinline)) uint32_t mint_thread_id() {
  uint32_t id;
  do {
    id = g_next_thread.fetch_add(1, std::memory_order_relaxed);
  } while (id == 0);
  t_thread_id = id;
  // Constructed here, on first use, so its destructor runs when this thread
  // exits and threads that never call cost nothing.
  static thread_local ThreadRetirer retirer;
  (void)retirer;
  return id;
}

// The calling thread's id. A plain thread-local read on every call after the
// first.
inline uint32_t this_thread_id() {
  const uint32_t id = t_thread_id;
  return id ? id : mint_thread_id();
}

// Frames written but not yet known to have reached the server. A reply
// acknowledges every frame up to its own id, because the server works through
// them in order, so this holds the batch since the last reply and nothing
// more. After a reconnect it is what gets sent again.
struct SentFrame {
  uint32_t req_id;
  std::vector<uint8_t> bytes;
};
std::deque<SentFrame> g_unacked;
size_t g_unacked_bytes = 0;

// A module image is megabytes, and holding several of them to replay would
// cost more than the recovery is worth. Past this the session is declared
// unrecoverable rather than quietly using unbounded memory.
constexpr size_t kMaxUnackedBytes = 64u << 20;
bool g_replay_possible = true;

void forget_acked_locked(uint32_t up_to) {
  while (!g_unacked.empty() && g_unacked.front().req_id <= up_to) {
    g_unacked_bytes -= g_unacked.front().bytes.size();
    g_unacked.pop_front();
  }
}

// Batching is on by default. RGPU_BATCH=0 makes every call a round trip,
// which is slower but makes a failing call report itself where it happened.
bool batching() {
  static bool v = env_int("RGPU_BATCH", 1) != 0;
  return v;
}

// Connects to RGPU_SERVER, "host:port", defaulting to 127.0.0.1:9713.
bool ensure_connected_locked() {
  arm_stats_report();
  if (g_fd >= 0) return true;
  if (g_connect_failed) return false;

  std::string spec = std::getenv("RGPU_SERVER") ? std::getenv("RGPU_SERVER")
                                                : "127.0.0.1:9713";
  std::string host = spec;
  std::string port = "9713";
  // Split on the last colon so a bare IPv6 literal still parses sensibly.
  size_t c = spec.rfind(':');
  if (c != std::string::npos) {
    host = spec.substr(0, c);
    port = spec.substr(c + 1);
  }

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  int e = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
  if (e != 0) {
    log("cannot resolve RGPU_SERVER=%s: %s", spec.c_str(), gai_strerror(e));
    g_connect_failed = true;
    return false;
  }

  int fd = -1;
  for (addrinfo* ai = res; ai; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);

  if (fd < 0) {
    log("cannot connect to %s: %s", spec.c_str(), std::strerror(errno));
    g_connect_failed = true;
    return false;
  }
  tune_socket(fd);

  // Say who we are. The server either starts a session or gives us back the
  // one we had, with everything in it still alive.
  Handshake hello{};
  hello.magic = kMagicHello;
  hello.version = kProtocolVersion;
  hello.session_hi = g_session.hi;
  hello.session_lo = g_session.lo;
  hello.last_req_id = g_last_reply;
  HandshakeReply reply{};
  if (!write_exact(fd, &hello, sizeof(hello)) ||
      !read_exact(fd, &reply, sizeof(reply)) ||
      reply.magic != kMagicHello) {
    log("handshake with %s failed", spec.c_str());
    ::close(fd);
    g_connect_failed = true;
    return false;
  }
  if (reply.version != kProtocolVersion) {
    log("server speaks protocol %u, this client speaks %u", reply.version,
        kProtocolVersion);
    ::close(fd);
    g_connect_failed = true;
    return false;
  }

  if (g_had_session && !reply.resumed) {
    // The session is gone rather than merely unreachable: every device
    // pointer and handle the application is holding refers to nothing.
    log("the server no longer has our session; GPU state is gone");
    ::close(fd);
    g_connect_failed = true;
    return false;
  }

  g_fd = fd;
  g_had_session = true;
  if (reply.resumed) {
    // Anything the server already finished must not be sent twice, and
    // everything still queued to be written is in the replay buffer already.
    forget_acked_locked(reply.last_req_id);
    g_queued.clear();
    log("reconnected to %s and resumed; %zu call(s) to send again",
        spec.c_str(), g_unacked.size());
    for (const auto& f : g_unacked) {
      if (!write_exact(g_fd, f.bytes.data(), f.bytes.size())) {
        ::close(g_fd);
        g_fd = -1;
        return false;
      }
    }
  } else if (verbose()) {
    log("connected to %s", spec.c_str());
  }
  return true;
}

// Waits for the server to come back, for as long as the session is likely to
// be kept there. Reconnecting is worth trying because the state is not on this
// side: if the session is still alive the application never learns anything
// happened.
bool reconnect_locked() {
  if (!g_replay_possible) {
    log("too much unacknowledged work to replay; not reconnecting");
    return false;
  }
  const int seconds = env_int("RGPU_RECONNECT_SECONDS", 60);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  int attempt = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const int wait_ms = attempt < 5 ? 100 * (1 << attempt) : 3000;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    attempt++;
    g_connect_failed = false;
    if (ensure_connected_locked()) return true;
    if (g_connect_failed && g_had_session && g_fd < 0 && !g_replay_possible) {
      break;
    }
  }
  log("could not reach the server again within %ds", seconds);
  g_connect_failed = true;
  return false;
}

// The connection is gone; the session on the other side may not be. Closing
// the socket here says nothing about whether the GPU state survived - that is
// decided when we try to reconnect and the server says whether it still has
// us.
void drop_connection_locked(const char* why) {
  if (g_fd >= 0) {
    ::close(g_fd);
    g_fd = -1;
  }
  log("connection lost (%s); trying to reach the server again", why);
}

}  // namespace

void log(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[rgpu] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

void queue_one_locked(uint32_t api_id, const Buffer& req, uint32_t flags,
                      uint32_t thread_id);

// Appends one frame to the queue rather than writing it. `thread_id` is the
// thread that issued the call, which is not necessarily the thread that will
// write it: a batch queued by one thread goes out with the next call any
// thread makes, and each frame has to keep its own issuer.
void queue_frame_locked(uint32_t api_id, const Buffer& req, uint32_t flags,
                        uint32_t thread_id) {
  if (g_have_retired.load(std::memory_order_relaxed)) {
    // Threads have exited since the last frame. Tell the server first, so it
    // can drop what it keeps for them; no reply, and nothing lost but a little
    // memory on the server if it never arrives.
    std::vector<uint32_t> gone;
    {
      std::lock_guard<std::mutex> lk(g_retired_mu);
      gone.swap(g_retired);
      // Never this thread's own id. A thread still calling is one whose
      // thread-local destructors are running - any built before its first
      // call run after the retirement - and announcing it ahead of those
      // calls would have the server forget the context they are made in.
      // It stays listed for the next frame another thread queues.
      auto self = std::find(gone.begin(), gone.end(), thread_id);
      if (self != gone.end()) {
        gone.erase(self);
        g_retired.push_back(thread_id);
      }
      g_have_retired.store(!g_retired.empty(), std::memory_order_relaxed);
    }
    if (!gone.empty()) {
      Buffer notice;
      notice.put<uint32_t>(static_cast<uint32_t>(gone.size()));
      notice.put_bytes(gone.data(), gone.size() * sizeof(uint32_t));
      queue_one_locked(API_rgpu_thread_gone, notice, kFlagNoReply, thread_id);
    }
  }
  queue_one_locked(api_id, req, flags, thread_id);

  if (t_retired) {
    // A call from a thread already retired. Its notice may well have gone out
    // already - any other thread's frame carries it, and this call may have
    // waited for the lock behind several - so the server has been told to
    // forget an id that is still in use. List it again, so that another
    // notice follows this frame; the last call the thread makes is then
    // always followed by one.
    std::lock_guard<std::mutex> lk(g_retired_mu);
    if (std::find(g_retired.begin(), g_retired.end(), thread_id) ==
        g_retired.end()) {
      g_retired.push_back(thread_id);
    }
    g_have_retired.store(true, std::memory_order_relaxed);
  }
}

void queue_one_locked(uint32_t api_id, const Buffer& req, uint32_t flags,
                      uint32_t thread_id) {
  ReqHeader h{};
  h.magic = kMagicReq;
  h.api_id = api_id;
  h.req_id = g_next_req++;
  h.flags = flags;
  h.thread_id = thread_id;
  h.payload_len = static_cast<uint32_t>(req.size());
  const auto* hb = reinterpret_cast<const uint8_t*>(&h);
  g_queued.insert(g_queued.end(), hb, hb + sizeof(h));
  g_queued.insert(g_queued.end(), req.data().begin(), req.data().end());
  g_stats.bytes_out += sizeof(h) + req.size();

  // Kept until a reply proves the server has it. Everything still in the send
  // queue is in here too, so a reconnect replays from here and starts the
  // queue empty.
  if (g_replay_possible) {
    SentFrame f;
    f.req_id = h.req_id;
    f.bytes.reserve(sizeof(h) + req.size());
    f.bytes.insert(f.bytes.end(), hb, hb + sizeof(h));
    f.bytes.insert(f.bytes.end(), req.data().begin(), req.data().end());
    g_unacked_bytes += f.bytes.size();
    g_unacked.push_back(std::move(f));
    if (g_unacked_bytes > kMaxUnackedBytes) {
      g_replay_possible = false;
      g_unacked.clear();
      g_unacked_bytes = 0;
    }
  }
}

// Writes everything queued as a single write. Returns false if the connection
// died, in which case it has already been dropped.
bool flush_locked() {
  if (g_fd < 0) return false;
  if (g_queued.empty()) return true;
  const bool ok = write_exact(g_fd, g_queued.data(), g_queued.size());
  g_queued.clear();
  if (!ok) drop_connection_locked("send failed");
  return ok;
}

CUresult call_async(uint32_t api_id, const Buffer& req) {
  if (!batching()) {
    // Wait for the reply, so a failure surfaces at the call that caused it.
    Buffer rsp;
    return call(api_id, req, &rsp);
  }

  const uint32_t thread_id = this_thread_id();
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ensure_connected_locked()) return CUDA_ERROR_NOT_INITIALIZED;

  if (verbose()) log("~> %s (%zu bytes, no reply)", api_name(api_id), req.size());

  queue_frame_locked(api_id, req, kFlagNoReply, thread_id);
  g_stats.async_calls++;
  if (g_queued.size() >= kQueueFlushBytes && !flush_locked()) {
    return CUDA_ERROR_UNKNOWN;
  }
  // The call has not run yet. CUDA says the same of any asynchronous call.
  return CUDA_SUCCESS;
}

CUresult call(uint32_t api_id, const Buffer& req, Buffer* rsp) {
  // ponytail: one connection under a global lock. Per-thread connections only
  // if profiling shows contention; correctness first.
  const uint32_t thread_id = this_thread_id();
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ensure_connected_locked()) return CUDA_ERROR_NOT_INITIALIZED;

  if (verbose()) log("-> %s (%zu bytes)", api_name(api_id), req.size());

  // Anything queued goes out ahead of this call, in one write, so the server
  // sees the same order the application issued.
  queue_frame_locked(api_id, req, 0, thread_id);
  const uint32_t expect_id = g_next_req - 1;

  // Two goes: one on the connection we have, and if that breaks, one on a
  // connection to the same session. The replay makes the second attempt the
  // same request, not a new one.
  for (int attempt = 0; attempt < 2; attempt++) {
    if (!flush_locked()) {
      if (attempt == 0 && reconnect_locked()) continue;
      return CUDA_ERROR_UNKNOWN;
    }

    RspHeader rh{};
    std::vector<uint8_t> payload;
    if (!recv_frame(g_fd, kMagicRsp, &rh, &payload)) {
      drop_connection_locked("recv failed");
      if (attempt == 0 && reconnect_locked()) continue;
      return CUDA_ERROR_UNKNOWN;
    }
    if (rh.req_id != expect_id) {
      // Under the lock this cannot happen unless the stream desynced.
      drop_connection_locked("response id mismatch");
      return CUDA_ERROR_UNKNOWN;
    }

    g_last_reply = rh.req_id;
    forget_acked_locked(rh.req_id);
    g_stats.round_trips++;
    g_stats.by_api[api_id]++;
    g_stats.bytes_in += sizeof(rh) + payload.size();
    *rsp = Buffer(std::move(payload));
    if (verbose()) log("<- %s result=%d", api_name(api_id), rh.result);
    return static_cast<CUresult>(rh.result);
  }
  return CUDA_ERROR_UNKNOWN;
}

CUresult unimplemented(const char* name, const char* why) {
  static std::mutex mu;
  static std::set<std::string> seen;
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!seen.insert(name).second) return CUDA_ERROR_NOT_SUPPORTED;
  }
  log("UNIMPLEMENTED %s (%s)", name, why ? why : "not generated");
  return CUDA_ERROR_NOT_SUPPORTED;
}

void unimplemented_rt(const char* name) {
  static std::mutex mu;
  static std::set<std::string> seen;
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!seen.insert(name).second) return;
  }
  log("UNTRANSLATED runtime call %s", name);
}

size_t image_size(const void* image) {
  if (!image) return 0;
  const auto* p = static_cast<const uint8_t*>(image);

  // Fatbin: a 16-byte header whose fatSize covers everything after it.
  struct FatbinHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t fat_size;
  };
  uint32_t magic = 0;
  std::memcpy(&magic, p, sizeof(magic));
  if (magic == 0xBA55ED50u) {
    FatbinHeader h{};
    std::memcpy(&h, p, sizeof(h));
    return static_cast<size_t>(h.header_size) + static_cast<size_t>(h.fat_size);
  }

  // Raw cubin: an ELF. Its length is the end of the section header table,
  // which for a cubin is the last thing in the file.
  if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
    const bool elf64 = p[4] == 2;
    if (elf64) {
      uint64_t shoff = 0;
      uint16_t shentsize = 0, shnum = 0;
      std::memcpy(&shoff, p + 0x28, sizeof(shoff));
      std::memcpy(&shentsize, p + 0x3a, sizeof(shentsize));
      std::memcpy(&shnum, p + 0x3c, sizeof(shnum));
      return static_cast<size_t>(shoff) +
             static_cast<size_t>(shentsize) * shnum;
    }
    uint32_t shoff = 0;
    uint16_t shentsize = 0, shnum = 0;
    std::memcpy(&shoff, p + 0x20, sizeof(shoff));
    std::memcpy(&shentsize, p + 0x2e, sizeof(shentsize));
    std::memcpy(&shnum, p + 0x30, sizeof(shnum));
    return static_cast<size_t>(shoff) +
           static_cast<size_t>(shentsize) * shnum;
  }

  // PTX arrives as NUL-terminated source text.
  if (p[0] == '/' || p[0] == '\n' || p[0] == '.' || p[0] == ' ') {
    return std::strlen(static_cast<const char*>(image)) + 1;
  }

  log("unrecognized module image (first bytes %02x %02x %02x %02x)",
      p[0], p[1], p[2], p[3]);
  return 0;
}

}  // namespace rgpu
