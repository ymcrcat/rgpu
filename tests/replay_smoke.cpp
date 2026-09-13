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

}  // namespace

int main() {
  reply_expected_request_in_flight();
  no_reply_request_in_flight();
  stale_reply_expected_copy();

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: a request replayed after a reconnect runs at most once\n");
  return 0;
}
