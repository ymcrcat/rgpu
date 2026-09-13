// Checks that a request is never run twice, however a reconnect lines up with
// it, without a GPU.
//
// After a connection drops the client reconnects and sends again every frame
// the server has not acknowledged. The handshake tells it the last request the
// session completed, but that is a snapshot: a request can still be running
// on the thread serving the session while the handshake is answered, and then
// the client, told it has not run, sends it again. The server has to notice
// when the copy arrives, because the driver calls behind a request - an
// allocation, a launch, a free - are not idempotent.
//
// A real client cannot be made to reconnect in the middle of a slow call on
// cue, so this speaks the protocol directly. The fake driver's
// cuDeviceTotalMem is slowed down (RGPU_FAKE_SLOW_TOTALMEM_MS) and counted
// ("totalmem" in RGPU_FAKE_STATS), which gives a call that is still running
// when the client comes back, and a count of how many times it ran.
//
//   RGPU_FAKE_SLOW_TOTALMEM_MS=1500 RGPU_FAKE_STATS=/tmp/s rgpu-server-fake 9726 &
//   RGPU_SERVER=127.0.0.1:9726 RGPU_FAKE_STATS=/tmp/s ./replay_smoke

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"

namespace {

int g_failures = 0;

#define EXPECT(cond, what)                                                \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what); \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

constexpr uint32_t kThread = 1;
// What the fake driver reports for every device.
constexpr size_t kFakeTotalMem = size_t(24) << 30;
constexpr uint64_t kSessionHi = 0x5e91a7000000000ull;

// The fake driver's count of cuDeviceTotalMem calls, or -1 if there is no
// stats file to read. The fake publishes only once something is counted, so
// until then the count is zero.
long totalmem_runs() {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  if (!path) return -1;
  std::FILE* f = std::fopen(path, "r");
  if (!f) return 0;
  char buf[512] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  const std::string line = std::string(" ") + buf;
  const std::string key = " totalmem=";
  const size_t at = line.find(key);
  if (at == std::string::npos) return 0;
  return std::strtol(line.c_str() + at + key.size(), nullptr, 10);
}

// Waits until the count reaches `want`: the slow call has started running.
bool await_runs(long want) {
  for (int i = 0; i < 100; i++) {
    if (totalmem_runs() >= want) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

int dial() {
  const char* env = std::getenv("RGPU_SERVER");
  std::string spec = env ? env : "127.0.0.1:9713";
  const size_t c = spec.rfind(':');
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (::getaddrinfo(spec.substr(0, c).c_str(), spec.substr(c + 1).c_str(),
                    &hints, &res) != 0) {
    return -1;
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
  if (fd >= 0) {
    // A server that neither answers nor closes would hang the test otherwise.
    timeval tv{10, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  return fd;
}

// Connects as session `lo`, saying the last reply received was `last_reply`.
// Returns the fd, or -1; fills in the server's answer.
int connect_session(uint64_t lo, uint32_t last_reply,
                    rgpu::HandshakeReply* reply) {
  int fd = dial();
  if (fd < 0) return -1;
  rgpu::Handshake hello{};
  hello.magic = rgpu::kMagicHello;
  hello.version = rgpu::kProtocolVersion;
  hello.session_hi = kSessionHi ^ static_cast<uint64_t>(::getpid());
  hello.session_lo = lo;
  hello.last_req_id = last_reply;
  if (!rgpu::write_exact(fd, &hello, sizeof(hello)) ||
      !rgpu::read_exact(fd, reply, sizeof(*reply)) ||
      reply->magic != rgpu::kMagicHello ||
      reply->version != rgpu::kProtocolVersion) {
    ::close(fd);
    return -1;
  }
  return fd;
}

// A request frame exactly as the client builds one, so it can be sent again
// byte for byte, as the client's replay does.
struct Frame {
  rgpu::ReqHeader h{};
  rgpu::Buffer payload;
};

Frame total_mem(uint32_t req_id, int dev, bool no_reply) {
  Frame f;
  f.payload.put<uint8_t>(1);  // the out-parameter is wanted
  f.payload.put<int>(dev);
  f.h.magic = rgpu::kMagicReq;
  f.h.api_id = rgpu::API_cuDeviceTotalMem_v2;
  f.h.req_id = req_id;
  f.h.flags = no_reply ? static_cast<uint32_t>(rgpu::kFlagNoReply) : 0u;
  f.h.thread_id = kThread;
  f.h.payload_len = static_cast<uint32_t>(f.payload.size());
  return f;
}

Frame device_count(uint32_t req_id) {
  Frame f;
  f.payload.put<uint8_t>(1);
  f.h.magic = rgpu::kMagicReq;
  f.h.api_id = rgpu::API_cuDeviceGetCount;
  f.h.req_id = req_id;
  f.h.thread_id = kThread;
  f.h.payload_len = static_cast<uint32_t>(f.payload.size());
  return f;
}

bool send(int fd, const Frame& f) { return rgpu::send_frame(fd, f.h, f.payload); }

struct Reply {
  bool received = false;
  rgpu::RspHeader h{};
  rgpu::Buffer body;
};

Reply receive(int fd) {
  Reply r;
  std::vector<uint8_t> body;
  r.received = rgpu::recv_frame(fd, rgpu::kMagicRsp, &r.h, &body);
  r.body = rgpu::Buffer(std::move(body));
  return r;
}

// A call that replies, checked to be the answer to `f` with `want`.
void expect_answer(int fd, const Frame& f, CUresult want, const char* what) {
  Reply r = receive(fd);
  if (!r.received) {
    std::fprintf(stderr, "FAIL: no reply to request %u (%s)\n", f.h.req_id,
                 what);
    g_failures++;
    return;
  }
  if (r.h.req_id != f.h.req_id || r.h.result != want) {
    std::fprintf(stderr,
                 "FAIL: %s: expected the reply to request %u with result %d, "
                 "got the reply to request %u with result %d\n",
                 what, f.h.req_id, (int)want, r.h.req_id, r.h.result);
    g_failures++;
  }
}

// A request that replies, still running when its connection drops and the
// client comes back. The handshake cannot yet say it completed, so the client
// sends it again; it must run once and be answered once, with its real reply.
void reply_expected_request_in_flight() {
  std::printf("-- a request still running when the client reconnects runs "
              "once\n");
  const uint64_t session = 1;
  const long before = totalmem_runs();
  EXPECT(before >= 0, "RGPU_FAKE_STATS is not set");

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;

  const Frame warm = device_count(1);
  EXPECT(send(fd, warm), "could not send the first request");
  expect_answer(fd, warm, CUDA_SUCCESS, "the first request");

  const Frame slow = total_mem(2, 0, false);
  EXPECT(send(fd, slow), "could not send the slow request");
  // Dropped while the call runs, as a broken route would drop it.
  EXPECT(await_runs(before + 1), "the slow request never started running");
  ::close(fd);

  fd = connect_session(session, 1, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1, "the session did not resume");
  if (fd < 0) return;
  // Otherwise the handshake already answered, and the race this case is
  // about did not happen: the test proves nothing, so it says so.
  EXPECT(hs.last_req_id == 1,
         "the handshake did not report the slow request as still running; "
         "the reconnect missed the window this case needs");

  EXPECT(send(fd, slow), "could not replay the slow request");
  Reply r = receive(fd);
  size_t bytes = 0;
  EXPECT(r.received, "no reply to the replayed request");
  if (r.received) {
    EXPECT(r.h.req_id == slow.h.req_id,
           "the reply to the replayed request carries another id");
    EXPECT(r.h.result == CUDA_SUCCESS, "the replayed request failed");
    EXPECT(r.body.get(&bytes) && bytes == kFakeTotalMem,
           "the replayed request's reply does not carry the device's memory");
  }

  // The conversation is still in step: the next reply is this call's.
  const Frame next = device_count(3);
  EXPECT(send(fd, next), "could not send a request after the replay");
  expect_answer(fd, next, CUDA_SUCCESS, "the request after the replay");

  const long runs = totalmem_runs() - before;
  std::printf("   the slow request ran %ld time(s)\n", runs);
  EXPECT(runs == 1, "a request replayed after a reconnect ran again");
  ::close(fd);
}

// A call sent without a reply, failing, still running when the connection
// drops; a call that replies was sent behind it and runs on the old
// connection too, taking the failure as its result. The client, told neither
// completed, sends both again. Neither may run again, and the reply that
// comes back is the one that carries the failure - once.
void no_reply_request_in_flight() {
  std::printf("-- a call without a reply, replayed after a reconnect, runs "
              "once and its failure is reported once\n");
  const uint64_t session = 2;
  const long before = totalmem_runs();

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;

  const Frame warm = device_count(1);
  EXPECT(send(fd, warm), "could not send the first request");
  expect_answer(fd, warm, CUDA_SUCCESS, "the first request");

  const Frame failing = total_mem(2, 99, true);  // no such device
  const Frame behind = device_count(3);
  EXPECT(send(fd, failing) && send(fd, behind),
         "could not send the call without a reply and the one behind it");
  EXPECT(await_runs(before + 1), "the call without a reply never started");
  ::close(fd);

  fd = connect_session(session, 1, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1, "the session did not resume");
  if (fd < 0) return;
  EXPECT(hs.last_req_id == 1,
         "the handshake did not report the call without a reply as still "
         "running; the reconnect missed the window this case needs");

  EXPECT(send(fd, failing) && send(fd, behind), "could not replay");
  expect_answer(fd, behind, CUDA_ERROR_INVALID_DEVICE,
                "the call behind the one without a reply");

  // Nothing is left over for the next call to report.
  const Frame next = device_count(4);
  EXPECT(send(fd, next), "could not send a request after the replay");
  expect_answer(fd, next, CUDA_SUCCESS,
                "the call after the replay (a failure reported twice)");

  const long runs = totalmem_runs() - before;
  std::printf("   the call without a reply ran %ld time(s)\n", runs);
  EXPECT(runs == 1, "a call without a reply ran again when replayed");

  // A copy that turns up even after the failure was reported - a client
  // replaying more than it should - is still not run, so there is no second
  // failure waiting for the call after it.
  EXPECT(send(fd, failing), "could not send the stale copy");
  const Frame after = device_count(5);
  EXPECT(send(fd, after), "could not send a request after the stale copy");
  expect_answer(fd, after, CUDA_SUCCESS,
                "the call after a stale copy of a failed call without a reply");
  EXPECT(totalmem_runs() - before == 1,
         "a stale copy of a call without a reply ran again");
  ::close(fd);
}

// A copy of a request that replies, turning up after a later request has been
// answered. The client never sends one - it forgets a frame once its reply
// arrives - so there is nothing right to answer it with; what matters is that
// it does not run, and does not put a reply in the stream the client would
// take for the next call's.
void stale_reply_expected_copy() {
  std::printf("-- a stale copy of an answered request is neither run nor "
              "answered\n");
  const uint64_t session = 3;
  const long before = totalmem_runs();

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;

  const Frame slow = total_mem(1, 0, false);
  EXPECT(send(fd, slow), "could not send the request");
  expect_answer(fd, slow, CUDA_SUCCESS, "the request");
  const Frame later = device_count(2);
  EXPECT(send(fd, later), "could not send the later request");
  expect_answer(fd, later, CUDA_SUCCESS, "the later request");

  EXPECT(send(fd, slow), "could not send the stale copy");
  const Frame next = device_count(3);
  EXPECT(send(fd, next), "could not send the request after the stale copy");
  expect_answer(fd, next, CUDA_SUCCESS, "the request after a stale copy");
  EXPECT(totalmem_runs() - before == 1, "a stale copy of a request ran again");
  ::close(fd);
}

// Request ids are 32 bits, and a long-lived client wraps them: after
// 0xFFFFFFFF comes 1 (0 names no request and is never used). Every ordering of
// ids has to survive that.

// A reply lost to a broken connection, for a request just past the wrap. The
// client last heard the reply to 0xFFFFFFFF, and the session completed 1: 1 is
// the later of the two, so the handshake resends its reply.
void handshake_resends_across_the_wrap() {
  std::printf("-- a lost reply to the first request past the id wrap is "
              "resent at the handshake\n");
  const uint64_t session = 4;

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;

  // A new session whose first request is not 1 still runs it.
  const Frame a = device_count(0xFFFFFFFEu);
  EXPECT(send(fd, a), "could not send a request");
  expect_answer(fd, a, CUDA_SUCCESS, "a new session's first request, id 0xFFFFFFFE");
  const Frame b = device_count(0xFFFFFFFFu);
  EXPECT(send(fd, b), "could not send a request");
  expect_answer(fd, b, CUDA_SUCCESS, "request 0xFFFFFFFF");

  const Frame lost = device_count(1);
  EXPECT(send(fd, lost), "could not send the first request past the wrap");
  // Long enough for a call with no delay to run and its reply to be kept;
  // the reply is never read.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  ::close(fd);

  fd = connect_session(session, 0xFFFFFFFFu, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1, "the session did not resume");
  if (fd < 0) return;
  EXPECT(hs.last_req_id == 1,
         "the handshake did not report request 1, past the wrap, as completed");
  expect_answer(fd, lost, CUDA_SUCCESS,
                "the resent reply to the first request past the wrap");

  const Frame next = device_count(2);
  EXPECT(send(fd, next), "could not send a request after the resent reply");
  expect_answer(fd, next, CUDA_SUCCESS, "the request after the resent reply");
  ::close(fd);
}

// A replay that straddles the wrap: a failing call without a reply, id
// 0xFFFFFFFF, still running when the connection drops, and a call that replies
// behind it, id 1. The client is told 0xFFFFFFFE completed and sends both
// again; neither runs again.
void replay_across_the_wrap() {
  std::printf("-- a replay that straddles the id wrap runs nothing twice\n");
  const uint64_t session = 5;
  const long before = totalmem_runs();

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;

  const Frame warm = device_count(0xFFFFFFFEu);
  EXPECT(send(fd, warm), "could not send the first request");
  expect_answer(fd, warm, CUDA_SUCCESS, "the first request");

  const Frame failing = total_mem(0xFFFFFFFFu, 99, true);
  const Frame behind = device_count(1);
  EXPECT(send(fd, failing) && send(fd, behind),
         "could not send the calls either side of the wrap");
  EXPECT(await_runs(before + 1), "the call without a reply never started");
  ::close(fd);

  fd = connect_session(session, 0xFFFFFFFEu, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1, "the session did not resume");
  if (fd < 0) return;
  EXPECT(hs.last_req_id == 0xFFFFFFFEu,
         "the handshake did not report the call without a reply as still "
         "running; the reconnect missed the window this case needs");

  EXPECT(send(fd, failing) && send(fd, behind), "could not replay");
  expect_answer(fd, behind, CUDA_ERROR_INVALID_DEVICE,
                "the call past the wrap, behind the failed one");
  const Frame next = device_count(2);
  EXPECT(send(fd, next), "could not send a request after the replay");
  expect_answer(fd, next, CUDA_SUCCESS, "the call after the replay");

  const long runs = totalmem_runs() - before;
  std::printf("   the call without a reply ran %ld time(s)\n", runs);
  EXPECT(runs == 1, "a call replayed across the wrap ran again");
  ::close(fd);
}

// A request too short for its call, as a client built against another wire
// layout would send. The server drops the connection over it - but first
// records it as completed, with an error, like any other request. Its handler
// may have called the driver before it found the arguments short, so a
// request left unrecorded would be sent again by the client when it
// reconnects, and run again.
Frame malformed(uint32_t req_id, bool no_reply) {
  Frame f = total_mem(req_id, 0, no_reply);
  f.payload = rgpu::Buffer();
  f.payload.put<uint8_t>(1);  // the out-parameter flag, and no device after it
  f.h.payload_len = static_cast<uint32_t>(f.payload.size());
  return f;
}

bool closed_by_server(int fd) {
  char byte = 0;
  return ::recv(fd, &byte, 1, 0) == 0;
}

void malformed_request_is_recorded() {
  std::printf("-- a malformed request is answered with an error and recorded "
              "as completed before the connection drops\n");
  const uint64_t session = 6;

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;
  const Frame warm = device_count(1);
  EXPECT(send(fd, warm), "could not send the first request");
  expect_answer(fd, warm, CUDA_SUCCESS, "the first request");

  const Frame bad = malformed(2, false);
  EXPECT(send(fd, bad), "could not send the malformed request");
  expect_answer(fd, bad, CUDA_ERROR_INVALID_VALUE, "the malformed request");
  EXPECT(closed_by_server(fd),
         "the server kept the connection open after a malformed request");
  ::close(fd);

  // As if that error reply had been lost with the connection.
  fd = connect_session(session, 1, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1, "the session did not resume");
  if (fd < 0) return;
  EXPECT(hs.last_req_id == 2,
         "the handshake did not report the malformed request as completed");
  expect_answer(fd, bad, CUDA_ERROR_INVALID_VALUE,
                "the resent reply to the malformed request");
  const Frame next = device_count(3);
  EXPECT(send(fd, next), "could not send a request after the malformed one");
  expect_answer(fd, next, CUDA_SUCCESS, "the request after the malformed one");
  ::close(fd);
}

void malformed_request_without_reply_is_recorded() {
  std::printf("-- a malformed call without a reply is recorded as completed, "
              "and its failure reported by the next call\n");
  const uint64_t session = 7;

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;
  const Frame warm = device_count(1);
  EXPECT(send(fd, warm), "could not send the first request");
  expect_answer(fd, warm, CUDA_SUCCESS, "the first request");

  const Frame bad = malformed(2, true);
  const Frame behind = device_count(3);
  EXPECT(send(fd, bad) && send(fd, behind),
         "could not send the malformed call and the one behind it");
  EXPECT(!receive(fd).received,
         "the server answered on the connection it should have dropped");
  ::close(fd);

  fd = connect_session(session, 1, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1, "the session did not resume");
  if (fd < 0) return;
  EXPECT(hs.last_req_id == 2,
         "the handshake did not report the malformed call as completed");
  // What the client sends again: everything after what the handshake said.
  EXPECT(send(fd, behind), "could not replay the call behind it");
  expect_answer(fd, behind, CUDA_ERROR_INVALID_VALUE,
                "the call after the malformed call without a reply");
  const Frame next = device_count(4);
  EXPECT(send(fd, next), "could not send a request after that");
  expect_answer(fd, next, CUDA_SUCCESS, "the request after that");
  ::close(fd);
}

// Frames naming request id 0, which is no request: a zero-filled header, or a
// client that does not number its requests. None is run or answered, and the
// session goes on. The server says so once per session however many arrive;
// run_smoke.sh counts the lines.
void frames_without_a_request_id() {
  std::printf("-- frames naming no request id are skipped\n");
  const uint64_t session = 8;
  const long before = totalmem_runs();

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return;
  for (int i = 0; i < 5; i++) {
    EXPECT(send(fd, total_mem(0, 0, i % 2 == 0)),
           "could not send a frame naming no request id");
  }
  const Frame next = device_count(1);
  EXPECT(send(fd, next), "could not send a request after them");
  expect_answer(fd, next, CUDA_SUCCESS,
                "the request after frames naming no request id");
  EXPECT(totalmem_runs() == before, "a frame naming no request id ran");
  ::close(fd);
}

// A handshake saying the client has already had replies, for a session this
// server does not have, comes from a client whose session expired or whose
// server restarted. Every handle it holds names nothing here. The server says
// the session is gone and closes, rather than starting an empty session under
// that id: the client's next attempt would find that one and be told it
// resumed, and would send its unacknowledged calls into it.
void claimed_progress_in_an_unknown_session() {
  std::printf("-- a handshake claiming progress in a session the server does "
              "not have is refused\n");
  const uint64_t session = 9;

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 7, &hs);
  EXPECT(fd >= 0, "no answer to a handshake for an unknown session");
  if (fd >= 0) {
    EXPECT(hs.resumed == 0, "the server resumed a session it never had");
    EXPECT(closed_by_server(fd),
           "the server kept a connection open after saying its session was "
           "gone");
    ::close(fd);
  }

  // Nothing was left behind under that id.
  fd = connect_session(session, 7, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0,
         "a second attempt resumed an empty session made for the first");
  if (fd >= 0) ::close(fd);

  // A client starting afresh is still served, under the same id too.
  fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "a fresh handshake did not start a session");
  if (fd < 0) return;
  const Frame first = device_count(1);
  EXPECT(send(fd, first), "could not send the first request");
  expect_answer(fd, first, CUDA_SUCCESS, "the first request of a fresh session");
  ::close(fd);
}

// --- replay_smoke handoff -----------------------------------------------------
//
// The grace period running out while a reconnect is being handed over. The
// handshake has told the client its session resumed; if the session then
// expires before the connection reaches the thread that served it, nothing
// will ever read from that connection, and the client waits forever for a
// reply. The hand-over has to check, in the same step, that the session is
// still there, and otherwise close the connection - the client then comes back
// and is told the session is gone.
//
// The server runs with RGPU_SESSION_GRACE=1 and RGPU_TEST_HANDOFF_DELAY_MS
// holding the hand-over back well past it, which makes the window wide enough
// to land in every time.
int expiry_during_handoff() {
  std::printf("-- a session that expires while a reconnect is handed over "
              "closes the connection\n");
  const uint64_t session = 1;

  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return 1;
  const Frame first = device_count(1);
  EXPECT(send(fd, first), "could not send the first request");
  expect_answer(fd, first, CUDA_SUCCESS, "the first request");
  ::close(fd);

  fd = connect_session(session, 1, &hs);
  EXPECT(fd >= 0 && hs.resumed == 1,
         "the reconnect was not told it resumed; the case needs the session "
         "to expire after the handshake, not before");
  if (fd < 0) return 1;
  const auto start = std::chrono::steady_clock::now();
  const Frame next = device_count(2);
  send(fd, next);
  Reply r = receive(fd);
  const double waited = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start)
                            .count();
  std::printf("   the connection ended after %.1fs\n", waited);
  EXPECT(!r.received,
         "a session that expired during the hand-over answered a request");
  // The receive times out after 10s (see dial). A connection nobody serves
  // runs into that; one the server closed ends when the hand-over does.
  EXPECT(waited < 8.0,
         "the connection handed to an expired session was left open, and a "
         "client would wait on it forever");
  ::close(fd);

  fd = connect_session(session, 1, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0,
         "coming back after the hand-over failed resumed an expired session");
  if (fd >= 0) ::close(fd);

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: a session that expires during a hand-over is not "
              "handed a connection\n");
  return 0;
}

// --- replay_smoke flood -----------------------------------------------------
//
// A client can make failures of calls sent without a reply as fast as it can
// send frames: calls from threads refused at the session's cap are each
// parked in a new retired slot, which evicts the oldest. Each used to be a
// line or two in the server log. They are now said once per session and
// counted, and the count is reported at expiry; run_smoke.sh counts the lines.
//
// What is held and what is evicted does not change: a thread's failure still
// reaches its next call that replies, and a refused thread let in later still
// gets its own failure back. The server runs with RGPU_MAX_CLIENT_THREADS=2.
Frame device_get(uint32_t req_id, uint32_t thread, int ordinal, bool no_reply) {
  Frame f;
  f.payload.put<uint8_t>(1);
  f.payload.put<int>(ordinal);
  f.h.magic = rgpu::kMagicReq;
  f.h.api_id = rgpu::API_cuDeviceGet;
  f.h.req_id = req_id;
  f.h.flags = no_reply ? static_cast<uint32_t>(rgpu::kFlagNoReply) : 0u;
  f.h.thread_id = thread;
  f.h.payload_len = static_cast<uint32_t>(f.payload.size());
  return f;
}

constexpr int kRefused = 200;

int deferred_error_flood() {
  std::printf("-- failures of calls without a reply are logged once per "
              "session\n");
  const uint64_t session = 2;
  rgpu::HandshakeReply hs{};
  int fd = connect_session(session, 0, &hs);
  EXPECT(fd >= 0 && hs.resumed == 0, "could not start a session");
  if (fd < 0) return 1;

  uint32_t id = 1;
  Frame f = device_get(id++, 1, 0, false);
  EXPECT(send(fd, f), "send");
  expect_answer(fd, f, CUDA_SUCCESS, "thread 1's first call");
  f = device_get(id++, 2, 0, false);
  EXPECT(send(fd, f), "send");
  expect_answer(fd, f, CUDA_SUCCESS, "thread 2's first call");

  // Thread 1 fails twice without a reply: the first is held, the second is
  // dropped behind it.
  EXPECT(send(fd, device_get(id++, 1, 99, true)), "send");
  EXPECT(send(fd, device_get(id++, 1, 99, true)), "send");
  // Threads refused at the cap, each failing without a reply.
  constexpr uint32_t kFirstRefused = 1000;
  for (int i = 0; i < kRefused; i++) {
    EXPECT(send(fd, device_get(id++, kFirstRefused + i, 0, true)), "send");
  }

  f = device_count(id++);
  EXPECT(send(fd, f), "send");
  expect_answer(fd, f, CUDA_ERROR_INVALID_DEVICE,
                "thread 1's next call that replies, after its failures");

  // Thread 2 goes, making room; the last refused thread comes in and gets
  // back the failure its refused call left.
  Frame gone;
  gone.payload.put<uint32_t>(1);
  gone.payload.put<uint32_t>(2);
  gone.h.magic = rgpu::kMagicReq;
  gone.h.api_id = rgpu::API_rgpu_thread_gone;
  gone.h.req_id = id++;
  gone.h.flags = rgpu::kFlagNoReply;
  gone.h.thread_id = 1;
  gone.h.payload_len = static_cast<uint32_t>(gone.payload.size());
  EXPECT(send(fd, gone), "send");
  f = device_count(id++);
  f.h.thread_id = kFirstRefused + kRefused - 1;
  EXPECT(send(fd, f), "send");
  expect_answer(fd, f, CUDA_ERROR_INVALID_VALUE,
                "the last thread refused at the cap, once let in");
  ::close(fd);

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: failures of calls without a reply were held as before\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "handoff") return expiry_during_handoff();
  if (mode == "flood") return deferred_error_flood();

  reply_expected_request_in_flight();
  no_reply_request_in_flight();
  stale_reply_expected_copy();
  handshake_resends_across_the_wrap();
  replay_across_the_wrap();
  malformed_request_is_recorded();
  malformed_request_without_reply_is_recorded();
  frames_without_a_request_id();
  claimed_progress_in_an_unknown_session();

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: a request replayed after a reconnect runs at most once\n");
  return 0;
}
