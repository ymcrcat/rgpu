#include "client/rpc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

#include "common/generated/api_ids.h"
#include "common/net.h"

namespace rgpu {
namespace {

std::mutex g_mu;          // guards the socket and the request counter
int g_fd = -1;
uint32_t g_next_req = 1;
bool g_connect_failed = false;

int env_int(const char* k, int dflt) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : dflt;
}

bool verbose() {
  static bool v = env_int("RGPU_VERBOSE", 0) != 0;
  return v;
}

// Connects to RGPU_SERVER, "host:port", defaulting to 127.0.0.1:9713.
bool ensure_connected_locked() {
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

CUresult call(uint32_t api_id, const Buffer& req, Buffer* rsp) {
  // ponytail: one connection under a global lock. Per-thread connections only
  // if profiling shows contention; correctness first.
  std::lock_guard<std::mutex> lk(g_mu);
  if (!ensure_connected_locked()) return CUDA_ERROR_NOT_INITIALIZED;

  ReqHeader h{};
  h.magic = kMagicReq;
  h.api_id = api_id;
  h.req_id = g_next_req++;
  h.flags = 0;
  h.payload_len = static_cast<uint32_t>(req.size());

  if (verbose()) log("-> %s (%zu bytes)", api_name(api_id), req.size());

  if (!send_frame(g_fd, h, req)) {
    drop_connection_locked("send failed");
    return CUDA_ERROR_UNKNOWN;
  }

  RspHeader rh{};
  std::vector<uint8_t> payload;
  if (!recv_frame(g_fd, kMagicRsp, &rh, &payload)) {
    drop_connection_locked("recv failed");
    return CUDA_ERROR_UNKNOWN;
  }
  if (rh.req_id != h.req_id) {
    // Under the lock this cannot happen unless the stream desynced.
    drop_connection_locked("response id mismatch");
    return CUDA_ERROR_UNKNOWN;
  }

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
