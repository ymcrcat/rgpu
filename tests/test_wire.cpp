// Round-trip check for the wire serializer. Runs anywhere, no CUDA needed.
//   c++ -std=c++17 -I.. tests/test_wire.cpp -o /tmp/test_wire && /tmp/test_wire
// The build is RelWithDebInfo, which defines NDEBUG and so turns every
// assert into nothing. Undefined here, or this test checks nothing at all.
#undef NDEBUG
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "common/net.h"
#include "common/sizes.h"
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

  // The byte length of an array whose element count came off the wire. A
  // count chosen so the product wraps to a plausible length must not come out
  // as that length: 2^62 + 1 twelve-byte elements wraps to exactly 12.
  assert(rgpu::array_bytes(3, 12) == 36);
  assert(rgpu::array_bytes(0, 12) == 0);
  assert(rgpu::array_bytes((size_t(1) << 62) + 1, 12) == SIZE_MAX);
  assert(rgpu::array_bytes(SIZE_MAX, 2) == SIZE_MAX);

  // A frame goes out as one write, header and payload together. Sent as two,
  // it is two packets, and anything between the two ends that holds a small
  // packet until the one before it is acknowledged - an ssh tunnel does -
  // turns every reply into two round trips of latency instead of one. A
  // SEQPACKET socket keeps each write as its own message, so one receive
  // shows how the frame was written.
  {
    int pair[2];
    assert(::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, pair) == 0);
    rgpu::RspHeader head{};
    Buffer payload;
    payload.put<uint64_t>(0x1122334455667788ull);
    payload.put<uint32_t>(42);
    assert(rgpu::send_frame(pair[0], head, payload));
    uint8_t got[256];
    const ssize_t n = ::recv(pair[1], got, sizeof(got), 0);
    assert(n == static_cast<ssize_t>(sizeof(head) + payload.size()));
    rgpu::ReqHeader request{};
    assert(rgpu::send_frame(pair[0], request, payload));
    const ssize_t m = ::recv(pair[1], got, sizeof(got), 0);
    assert(m == static_cast<ssize_t>(sizeof(request) + payload.size()));
    ::close(pair[0]);
    ::close(pair[1]);
  }

  // The server reads frames in large pieces, not two receives per frame: a
  // client queues hundreds of small calls and sends them as one write, and a
  // receive each for every header and payload was about a thousand system
  // calls per generated token. So the first frame read takes everything that
  // was sent with it off the socket.
  {
    int pair[2];
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    Buffer first, second;
    first.put<uint32_t>(7);
    second.put<uint64_t>(0x0102030405060708ull);
    ReqHeader h1{}, h2{};
    h1.magic = h2.magic = rgpu::kMagicReq;
    h1.payload_len = static_cast<uint32_t>(first.size());
    h2.payload_len = static_cast<uint32_t>(second.size());
    std::vector<uint8_t> both;
    for (const auto& [h, b] : {std::pair{&h1, &first}, std::pair{&h2, &second}}) {
      const auto* hb = reinterpret_cast<const uint8_t*>(h);
      both.insert(both.end(), hb, hb + sizeof(*h));
      both.insert(both.end(), b->data().begin(), b->data().end());
    }
    assert(rgpu::write_exact(pair[0], both.data(), both.size()));

    rgpu::FrameReader reader(pair[1]);
    ReqHeader got{};
    std::vector<uint8_t> payload;
    assert(reader.recv_frame(rgpu::kMagicReq, &got, &payload));
    assert(payload == first.data());
    uint8_t byte;
    assert(::recv(pair[1], &byte, 1, MSG_PEEK | MSG_DONTWAIT) < 0);  // drained
    assert(reader.recv_frame(rgpu::kMagicReq, &got, &payload));
    assert(payload == second.data());

    // A payload larger than the reader's own buffer, arriving in pieces.
    std::vector<uint8_t> big(300 * 1024);
    for (size_t i = 0; i < big.size(); i++) big[i] = static_cast<uint8_t>(i * 31);
    ReqHeader h3{};
    h3.magic = rgpu::kMagicReq;
    h3.payload_len = static_cast<uint32_t>(big.size());
    std::thread writer([&] {
      assert(rgpu::write_exact(pair[0], &h3, sizeof(h3)));
      assert(rgpu::write_exact(pair[0], big.data(), big.size()));
      assert(rgpu::write_exact(pair[0], both.data(), both.size()));
      ::close(pair[0]);
    });
    assert(reader.recv_frame(rgpu::kMagicReq, &got, &payload));
    assert(payload == big);
    assert(reader.recv_frame(rgpu::kMagicReq, &got, &payload));
    assert(payload == first.data());
    assert(reader.recv_frame(rgpu::kMagicReq, &got, &payload));
    assert(payload == second.data());
    assert(!reader.recv_frame(rgpu::kMagicReq, &got, &payload));  // peer closed
    writer.join();
    ::close(pair[1]);
  }

  std::printf("wire round-trip OK\n");
  return 0;
}
