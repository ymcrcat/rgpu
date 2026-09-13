// Round-trip check for the wire serializer. Runs anywhere, no CUDA needed.
//   c++ -std=c++17 -I.. tests/test_wire.cpp -o /tmp/test_wire && /tmp/test_wire
// The build is RelWithDebInfo, which defines NDEBUG and so turns every
// assert into nothing. Undefined here, or this test checks nothing at all.
#undef NDEBUG
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "common/wire.h"

using rgpu::Buffer;
using rgpu::Handshake;
using rgpu::HandshakeReply;
using rgpu::ReqHeader;

int main() {
  // Headers go on the wire as a memcpy of the struct, so their layout is the
  // protocol. Protocol 3 added the issuing client thread to every request.
  assert(rgpu::kProtocolVersion == 3);
  assert(sizeof(ReqHeader) == 24);
  assert(offsetof(ReqHeader, magic) == 0);
  assert(offsetof(ReqHeader, api_id) == 4);
  assert(offsetof(ReqHeader, req_id) == 8);
  assert(offsetof(ReqHeader, flags) == 12);
  assert(offsetof(ReqHeader, thread_id) == 16);
  assert(offsetof(ReqHeader, payload_len) == 20);

  // The handshake is how two versions find out they disagree, so it must not
  // change between them: a peer from another version has to be able to read
  // the other side's version to say which one it is.
  assert(sizeof(Handshake) == 32);
  assert(offsetof(Handshake, version) == 4);
  assert(sizeof(HandshakeReply) == 16);
  assert(offsetof(HandshakeReply, version) == 4);

  Buffer b;
  b.put<uint32_t>(0xdeadbeef);
  b.put<uint64_t>(0x1122334455667788ull);
  b.put<int32_t>(-7);
  const char payload[] = {1, 2, 3, 4, 5};
  b.put_sized(payload, sizeof(payload));
  b.put_str("cuLaunchKernel");
  b.put_str(nullptr);
  b.put<double>(2.5);

  Buffer r(b.data());
  uint32_t u32 = 0;
  uint64_t u64 = 0;
  int32_t i32 = 0;
  assert(r.get(&u32) && u32 == 0xdeadbeef);
  assert(r.get(&u64) && u64 == 0x1122334455667788ull);
  assert(r.get(&i32) && i32 == -7);

  const uint8_t* bytes = nullptr;
  size_t n = 0;
  assert(r.get_sized(&bytes, &n) && n == sizeof(payload));
  for (size_t i = 0; i < n; i++) assert(bytes[i] == payload[i]);

  std::string s;
  bool present = false;
  assert(r.get_str(&s, &present) && present && s == "cuLaunchKernel");
  assert(r.get_str(&s, &present) && !present && s.empty());

  double d = 0;
  assert(r.get(&d) && d == 2.5);
  assert(r.ok());

  // Reading past the end must fail rather than return garbage.
  uint32_t overrun = 0;
  assert(!r.get(&overrun));
  assert(!r.ok());

  // A truncated buffer must not read out of bounds.
  std::vector<uint8_t> truncated(b.data().begin(), b.data().begin() + 3);
  Buffer t(truncated);
  uint64_t big = 0;
  assert(!t.get(&big));
  assert(!t.ok());

  // An empty buffer is safely readable and immediately not-ok.
  Buffer empty;
  assert(!empty.get(&u32));

  // A length field near 2^64 must be refused. The bounds check used to add
  // it to the read position, which wraps around, so the check passed and the
  // caller got a pointer with a length far past the end of the frame.
  Buffer hostile;
  hostile.put<uint64_t>(UINT64_MAX - 4);
  hostile.put<uint32_t>(0);
  Buffer h(hostile.data());
  const uint8_t* evil = nullptr;
  size_t evil_n = 0;
  assert(!h.get_sized(&evil, &evil_n));
  assert(!h.ok());

  // Request-id order, across the 32-bit wrap. 0 is no request.
  assert(rgpu::req_at_or_before(5, 5));
  assert(rgpu::req_at_or_before(4, 5) && !rgpu::req_at_or_before(5, 4));
  assert(rgpu::req_at_or_before(0xFFFFFFFFu, 1));
  assert(!rgpu::req_at_or_before(1, 0xFFFFFFFFu));
  assert(rgpu::req_at_or_before(0xFFFFFFF7u, 5));
  assert(rgpu::req_at_or_before(0, 1) && rgpu::req_at_or_before(0, 0xFFFFFFFFu));
  assert(!rgpu::req_at_or_before(1, 0) && !rgpu::req_at_or_before(0xFFFFFFFFu, 0));
  assert(rgpu::req_at_or_before(0, 0));

  std::printf("wire round-trip OK\n");
  return 0;
}
