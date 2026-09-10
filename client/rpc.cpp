#include "client/rpc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "common/generated/api_ids.h"
#include "common/net.h"

namespace rgpu {
namespace {

std::mutex g_mu;          // guards the socket, the counter and the send queue
int g_fd = -1;
uint32_t g_next_req = 1;
bool g_connect_failed = false;

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
  g_fd = fd;
  if (verbose()) log("connected to %s", spec.c_str());
  return true;
}

void drop_connection_locked(const char* why) {
  if (g_fd >= 0) {
    ::close(g_fd);
    g_fd = -1;
  }
  // A dropped connection loses all server-side state (contexts, allocations,
  // loaded modules), so silently reconnecting would hand the application a
  // GPU that has forgotten everything. Fail instead.
  g_connect_failed = true;
  log("connection lost (%s); GPU state is gone, failing subsequent calls", why);
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

// Appends one frame to the queue rather than writing it.
void queue_frame_locked(uint32_t api_id, const Buffer& req, uint32_t flags) {
  ReqHeader h{};
  h.magic = kMagicReq;
  h.api_id = api_id;
  h.req_id = g_next_req++;
  h.flags = flags;
  h.payload_len = static_cast<uint32_t>(req.size());
  const auto* hb = reinterpret_cast<const uint8_t*>(&h);
  g_queued.insert(g_queued.end(), hb, hb + sizeof(h));
  g_queued.insert(g_queued.end(), req.data().begin(), req.data().end());
  g_stats.bytes_out += sizeof(h) + req.size();
}

// Writes everything queued as a single write. Returns false if the connection
// died, in which case it has already been dropped.
bool flush_locked() {
  if (g_queued.empty()) return true;
  const bool ok = write_exact(g_fd, g_queued.data(), g_queued.size());
  g_queued.clear();
  if (!ok) drop_connection_locked("send failed");
  return ok;
}

CUresult call_async(uint32_t api_id, const Buffer& req) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ensure_connected_locked()) return CUDA_ERROR_NOT_INITIALIZED;

  if (verbose()) log("~> %s (%zu bytes, no reply)", api_name(api_id), req.size());

  if (!batching()) {
    // Same frame, but wait for the reply so a failure surfaces here.
    ReqHeader h{};
    h.magic = kMagicReq;
    h.api_id = api_id;
    h.req_id = g_next_req++;
    h.flags = 0;
    h.payload_len = static_cast<uint32_t>(req.size());
    if (!flush_locked()) return CUDA_ERROR_UNKNOWN;
    if (!send_frame(g_fd, h, req)) {
      drop_connection_locked("send failed");
      return CUDA_ERROR_UNKNOWN;
    }
    g_stats.bytes_out += sizeof(h) + req.size();
    RspHeader rh{};
    std::vector<uint8_t> payload;
    if (!recv_frame(g_fd, kMagicRsp, &rh, &payload)) {
      drop_connection_locked("recv failed");
      return CUDA_ERROR_UNKNOWN;
    }
    g_stats.round_trips++;
    g_stats.by_api[api_id]++;
    g_stats.bytes_in += sizeof(rh) + payload.size();
    return static_cast<CUresult>(rh.result);
  }

  queue_frame_locked(api_id, req, kFlagNoReply);
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
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ensure_connected_locked()) return CUDA_ERROR_NOT_INITIALIZED;

  if (verbose()) log("-> %s (%zu bytes)", api_name(api_id), req.size());

  // Anything queued goes out ahead of this call, in one write, so the server
  // sees the same order the application issued.
  queue_frame_locked(api_id, req, 0);
  const uint32_t expect_id = g_next_req - 1;
  if (!flush_locked()) return CUDA_ERROR_UNKNOWN;

  RspHeader rh{};
  std::vector<uint8_t> payload;
  if (!recv_frame(g_fd, kMagicRsp, &rh, &payload)) {
    drop_connection_locked("recv failed");
    return CUDA_ERROR_UNKNOWN;
  }
  if (rh.req_id != expect_id) {
    // Under the lock this cannot happen unless the stream desynced.
    drop_connection_locked("response id mismatch");
    return CUDA_ERROR_UNKNOWN;
  }

  g_stats.round_trips++;
  g_stats.by_api[api_id]++;
  g_stats.bytes_in += sizeof(rh) + payload.size();
  *rsp = Buffer(std::move(payload));
  if (verbose()) log("<- %s result=%d", api_name(api_id), rh.result);
  return static_cast<CUresult>(rh.result);
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

size_t pointer_attr_size(unsigned int attribute) {
  switch (attribute) {
    // Handles and addresses: one pointer-sized value.
    case CU_POINTER_ATTRIBUTE_CONTEXT:
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
      return sizeof(void*);
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
    case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
      return sizeof(size_t);
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
      return sizeof(unsigned int);
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_MAPPED:
      return sizeof(bool);
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
    case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
      return sizeof(unsigned long long);
    default:
      // P2P tokens and mempool handles among others. Reporting zero makes the
      // call fail rather than reading a wrong number of bytes.
      log("unknown pointer attribute %u; refusing rather than guessing a size",
          attribute);
      return 0;
  }
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
