// rgpu wire format: length-prefixed frames over a stream socket.
//
// Both ends are little-endian (x86_64 and aarch64), so scalars go on the wire
// in native byte order. If a big-endian client ever matters, byteswap here.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace rgpu {

constexpr uint32_t kMagicReq = 0x52475155;  // "RGQU"
constexpr uint32_t kMagicRsp = 0x52475250;  // "RGRP"
constexpr uint32_t kMagicHello = 0x52474845;  // "RGHE"

// --- request ids -------------------------------------------------------------
//
// A client numbers its requests 1, 2, 3, ... in the order it sends them, and
// the numbers are 32 bits, so a client that lives long enough wraps. 0 is never
// a request's id: it means "no request" - a session that has completed none, a
// client that has had no reply - and minting skips it, so after 0xFFFFFFFF
// comes 1.
//
// Ordering ids therefore uses sequence-number arithmetic (RFC 1982): b is at
// or after a when the distance from a forward to b, modulo 2^32, is under half
// the id space. That is right for any two ids less than 2^31 requests apart.
// Past that an old id reads as a new one, so what bounds each comparison:
//
//   - The server's check for a request that already ran compares a frame the
//     client sent again with the last request completed. The client only
//     sends again the frames since its last reply, and it caps those at
//     64 MiB - under three million frames - so the two are that close.
//   - The handshake compares the last reply the client received with the
//     last reply the server kept. Nothing on the wire bounds that: any number
//     of calls without a reply can come between two replies. But they are
//     also frames since the client's last reply, all held for replay, so the
//     two ids can only drift more than the cap apart once the client has
//     passed the cap and thrown those frames away. Replay is already
//     impossible then: frames the server may never have received are gone,
//     and no ordering of ids at the handshake can bring the session back in
//     step.
//
// Nothing checks these bounds. A client that breaks them - a buggy or hostile
// one - can have its ids misordered.
//
// Every ordering of request ids, on either side, goes through this. Equality
// needs nothing special.

// Whether request `a` is at or before request `b`. No request (0) is before
// every request, and no request but itself is before it.
constexpr bool req_at_or_before(uint32_t a, uint32_t b) {
  if (a == 0) return true;
  if (b == 0) return false;
  return static_cast<uint32_t>(b - a) < 0x80000000u;
}

// Sent once, before any frames. The session id is the client process, not the
// connection: a connection that drops takes no state with it, because the
// server keeps the session alive for a while and hands the next connection
// carrying the same id back to the very thread that was serving it. That
// thread still holds the CUDA context, so device memory and every handle the
// client is holding stay valid.
//
// 3: every request carries the client thread that issued it (see ReqHeader).
// A peer speaking another version is refused at the handshake, whose layout
// does not change between versions so that the refusal can say which one each
// side speaks.
constexpr uint32_t kProtocolVersion = 3;

struct Handshake {
  uint32_t magic;
  uint32_t version;
  uint64_t session_hi;
  uint64_t session_lo;
  uint32_t last_req_id;  // last reply the client received; 0 if none yet
  uint32_t reserved;
};

struct HandshakeReply {
  uint32_t magic;
  uint32_t version;
  uint32_t resumed;      // 1 if this attached to a session that already existed
  uint32_t last_req_id;  // last request that session completed; 0 if none
};

// Flags on a request frame.
enum : uint32_t {
  // Server must not send a response. Set for calls in the async class once
  // batching is enabled; errors surface at the next synchronization point.
  kFlagNoReply = 1u << 0,
};

struct ReqHeader {
  uint32_t magic;
  uint32_t api_id;
  uint32_t req_id;
  uint32_t flags;
  // The client thread that issued this request, as the client numbers them:
  // an opaque id minted from 1 and never reused, not an OS thread id. CUDA's
  // current context is per thread, and one server thread serves every thread
  // of a client, so this is what lets it put each request back under the
  // context its own thread selected. It travels with the frame rather than
  // with the write, because a batch issued by one thread can be flushed by
  // another's call. Zero is never a valid id.
  uint32_t thread_id;
  uint32_t payload_len;
};
// Written and read as a memcpy of the struct, so the size is the protocol.
static_assert(sizeof(ReqHeader) == 24, "ReqHeader is 24 bytes on the wire");

struct RspHeader {
  uint32_t magic;
  uint32_t req_id;
  int32_t result;  // CUresult
  uint32_t payload_len;
};

// Append-only byte buffer with a read cursor. Serialization is a flat
// concatenation of fields; both sides walk it in the same order, so there are
// no tags and no length prefixes on individual fields.
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::vector<uint8_t> data) : data_(std::move(data)) {}

  template <typename T>
  void put(const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "put needs a POD type");
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    data_.insert(data_.end(), p, p + sizeof(T));
  }

  // Raw bytes, length known from context (a size parameter in the API).
  void put_bytes(const void* p, size_t n) {
    if (n == 0) return;
    const auto* b = static_cast<const uint8_t*>(p);
    data_.insert(data_.end(), b, b + n);
  }

  // Bytes with an explicit length, for when the reader cannot know it.
  void put_sized(const void* p, size_t n) {
    put<uint64_t>(n);
    put_bytes(p, n);
  }

  void put_str(const char* s) {
    size_t n = s ? std::strlen(s) : 0;
    put<uint8_t>(s ? 1 : 0);
    if (s) put_sized(s, n);
  }

  // Returns false once anything has overrun; callers check ok() at the end
  // rather than after every field.
  template <typename T>
  bool get(T* out) {
    static_assert(std::is_trivially_copyable<T>::value, "get needs a POD type");
    if (!take(sizeof(T))) return false;
    std::memcpy(out, data_.data() + read_ - sizeof(T), sizeof(T));
    return true;
  }

  // Borrows a view into the buffer; valid while the Buffer lives.
  bool get_bytes(size_t n, const uint8_t** out) {
    if (!take(n)) return false;
    *out = data_.data() + read_ - n;
    return true;
  }

  bool get_sized(const uint8_t** out, size_t* n) {
    uint64_t len = 0;
    if (!get(&len)) return false;
    *n = static_cast<size_t>(len);
    if (len == 0) {
      *out = nullptr;
      return true;
    }
    return get_bytes(*n, out);
  }

  // Returns a copy because the caller needs a NUL terminator.
  bool get_str(std::string* out, bool* present) {
    uint8_t has = 0;
    if (!get(&has)) return false;
    *present = has != 0;
    if (!has) {
      out->clear();
      return true;
    }
    const uint8_t* p = nullptr;
    size_t n = 0;
    if (!get_sized(&p, &n)) return false;
    out->assign(reinterpret_cast<const char*>(p), n);
    return true;
  }

  const std::vector<uint8_t>& data() const { return data_; }
  std::vector<uint8_t>& data() { return data_; }
  size_t size() const { return data_.size(); }
  void clear() {
    data_.clear();
    read_ = 0;
    ok_ = true;
  }
  bool ok() const { return ok_; }

 private:
  bool take(size_t n) {
    // Compared against what is left rather than added to the read position:
    // a length near 2^64 from the wire would wrap that sum and pass.
    if (!ok_ || n > data_.size() - read_) {
      ok_ = false;
      return false;
    }
    read_ += n;
    return true;
  }

  std::vector<uint8_t> data_;
  size_t read_ = 0;
  bool ok_ = true;
};

}  // namespace rgpu
