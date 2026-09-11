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

// Sent once, before any frames. The session id is the client process, not the
// connection: a connection that drops takes no state with it, because the
// server keeps the session alive for a while and hands the next connection
// carrying the same id back to the very thread that was serving it. That
// thread still holds the CUDA context, so device memory and every handle the
// client is holding stay valid.
constexpr uint32_t kProtocolVersion = 2;

struct Handshake {
  uint32_t magic;
  uint32_t version;
  uint64_t session_hi;
  uint64_t session_lo;
  uint32_t last_req_id;  // last reply the client received; 0 for a new session
  uint32_t reserved;
};

struct HandshakeReply {
  uint32_t magic;
  uint32_t version;
  uint32_t resumed;      // 1 if this attached to a session that already existed
  uint32_t last_req_id;  // last request that session actually completed
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
  uint32_t payload_len;
};

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
    if (!ok_ || read_ + n > data_.size()) {
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
