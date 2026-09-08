// Blocking stream-socket helpers shared by the shim and the server.
#pragma once

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "common/wire.h"

namespace rgpu {

// Loops over partial transfers and retries EINTR. Returns false on error or a
// peer that closed early.
inline bool read_exact(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = ::recv(fd, p, n, 0);
    if (r > 0) {
      p += r;
      n -= static_cast<size_t>(r);
      continue;
    }
    if (r == 0) return false;              // peer closed
    if (errno == EINTR) continue;
    return false;
  }
  return true;
}

inline bool write_exact(int fd, const void* buf, size_t n) {
  const auto* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    // MSG_NOSIGNAL: a dead peer must surface as EPIPE, not kill the process.
#ifdef MSG_NOSIGNAL
    ssize_t r = ::send(fd, p, n, MSG_NOSIGNAL);
#else
    ssize_t r = ::send(fd, p, n, 0);
#endif
    if (r > 0) {
      p += r;
      n -= static_cast<size_t>(r);
      continue;
    }
    if (r < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

// Kernel launches are small and latency-bound, so Nagle would be actively
// harmful here: it would sit on a launch waiting for more bytes.
inline void tune_socket(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
}

inline bool send_frame(int fd, const ReqHeader& h, const Buffer& payload) {
  if (!write_exact(fd, &h, sizeof(h))) return false;
  if (payload.size() == 0) return true;
  return write_exact(fd, payload.data().data(), payload.size());
}

inline bool send_frame(int fd, const RspHeader& h, const Buffer& payload) {
  if (!write_exact(fd, &h, sizeof(h))) return false;
  if (payload.size() == 0) return true;
  return write_exact(fd, payload.data().data(), payload.size());
}

// Reads a header plus its payload. `max_payload` bounds a corrupt or hostile
// length so a desync cannot make us allocate wildly.
template <typename Header>
bool recv_frame(int fd, uint32_t magic, Header* h, std::vector<uint8_t>* payload,
                size_t max_payload = 1ull << 32) {
  if (!read_exact(fd, h, sizeof(*h))) return false;
  if (h->magic != magic) return false;
  if (h->payload_len > max_payload) return false;
  payload->resize(h->payload_len);
  if (h->payload_len == 0) return true;
  return read_exact(fd, payload->data(), payload->size());
}

}  // namespace rgpu
