// Checks that every request names the client thread that issued it, without a
// GPU.
//
// CUDA's current context is per thread, and one server thread serves every
// thread of a client process, so the only way the server can run a request
// under the context its thread selected is to be told which thread that was.
// Most of what can go wrong with that is on the client, and is visible in the
// bytes it writes, so most of this test is a scripted server of its own that
// reads exactly what the client sends:
//
//   wire     the scripted server. Each case forks a fresh client process,
//            because a client's connection state is per process and some
//            cases leave it refusing to connect. Needs nothing else.
//   runtime  against rgpu-server-fake at RGPU_SERVER, alone on it: a device
//            reset on one thread has to invalidate another thread's cached
//            context selection.
//
//   ./thread_id_smoke wire
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./thread_id_smoke runtime

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "client/rpc.h"
#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"

namespace {

using rgpu::Buffer;
using rgpu::Handshake;
using rgpu::HandshakeReply;
using rgpu::ReqHeader;
using rgpu::RspHeader;

// Failures in whichever process this is: the scripted server in the parent, or
// the client in a forked child, which reports them through its exit status.
std::atomic<int> g_failures{0};

#define EXPECT(cond, what)                                                 \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what);  \
      g_failures++;                                                        \
    }                                                                      \
  } while (0)

// Call ids the scripted server answers without caring what they mean.
constexpr uint32_t kApiA = 0x7e570a01;
constexpr uint32_t kApiB = 0x7e570b01;
constexpr uint32_t kApiC = 0x7e570c01;
constexpr uint32_t kApiD = 0x7e570d01;
constexpr uint32_t kApiMain = 0x7e570e01;
constexpr uint32_t kApiMarker = 0x7e570f01;

// --- the scripted server ---------------------------------------------------

struct Frame {
  ReqHeader h{};
  std::vector<uint8_t> payload;

  // The frame exactly as it was written, header and all.
  std::vector<uint8_t> bytes() const {
    std::vector<uint8_t> out(sizeof(h) + payload.size());
    std::memcpy(out.data(), &h, sizeof(h));
    if (!payload.empty()) {
      std::memcpy(out.data() + sizeof(h), payload.data(), payload.size());
    }
    return out;
  }
};

int listen_local(uint16_t* port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  socklen_t len = sizeof(addr);
  if (fd < 0 || ::bind(fd, reinterpret_cast<sockaddr*>(&addr), len) != 0 ||
      ::listen(fd, 4) != 0 ||
      ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    std::perror("listen");
    std::exit(2);
  }
  *port = ntohs(addr.sin_port);
  return fd;
}

// The next connection, or -1 if none arrives in time.
int accept_within(int lfd, int ms) {
  pollfd p{lfd, POLLIN, 0};
  if (::poll(&p, 1, ms) <= 0) return -1;
  int fd = ::accept(lfd, nullptr, nullptr);
  if (fd >= 0) {
    // A client that stops talking fails the case instead of hanging it.
    timeval tv{10, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  return fd;
}

bool read_hello(int fd, Handshake* hello) {
  return rgpu::read_exact(fd, hello, sizeof(*hello)) &&
         hello->magic == rgpu::kMagicHello;
}

bool send_hello(int fd, uint32_t version, bool resumed, uint32_t last_req) {
  HandshakeReply reply{};
  reply.magic = rgpu::kMagicHello;
  reply.version = version;
  reply.resumed = resumed ? 1 : 0;
  reply.last_req_id = last_req;
  return rgpu::write_exact(fd, &reply, sizeof(reply));
}

// Accepts a connection and completes a new-session handshake on it.
int accept_session(int lfd, Handshake* hello) {
  int fd = accept_within(lfd, 10000);
  EXPECT(fd >= 0, "the client never connected");
  if (fd < 0) return -1;
  Handshake ignored{};
  if (!hello) hello = &ignored;
  EXPECT(read_hello(fd, hello), "the client sent no handshake");
  EXPECT(hello->version == rgpu::kProtocolVersion,
         "the client's handshake named the wrong protocol");
  EXPECT(send_hello(fd, rgpu::kProtocolVersion, false, 0),
         "could not answer the handshake");
  return fd;
}

bool read_frame(int fd, Frame* f) {
  return rgpu::recv_frame(fd, rgpu::kMagicReq, &f->h, &f->payload);
}

bool reply(int fd, const Frame& f, int32_t result,
           const Buffer& payload = Buffer()) {
  RspHeader rh{};
  rh.magic = rgpu::kMagicRsp;
  rh.req_id = f.h.req_id;
  rh.result = result;
  rh.payload_len = static_cast<uint32_t>(payload.size());
  return rgpu::send_frame(fd, rh, payload);
}

// Reads everything the client sends until it goes away, answering each call
// that wants an answer with success.
std::vector<Frame> read_until_closed(int fd) {
  std::vector<Frame> frames;
  Frame f;
  while (read_frame(fd, &f)) {
    if (!(f.h.flags & rgpu::kFlagNoReply)) reply(fd, f, CUDA_SUCCESS);
    frames.push_back(f);
  }
  return frames;
}

// Runs `client` in a fresh process pointed at a listener, and `server` here
// against that listener. The child is forked before this process has any
// other thread, so it starts clean; `server` may close the listener, and
// sets it to -1 if it does.
void run_case(const char* name, const std::function<void()>& client,
              const std::function<void(int& lfd)>& server) {
  std::fprintf(stderr, "-- %s\n", name);
  const int before = g_failures.load();
  uint16_t port = 0;
  int lfd = listen_local(&port);

  const pid_t pid = ::fork();
  if (pid == 0) {
    g_failures = 0;  // only this process's own count decides its exit status
    ::close(lfd);
    ::alarm(60);
    const std::string spec = "127.0.0.1:" + std::to_string(port);
    ::setenv("RGPU_SERVER", spec.c_str(), 1);
    // Several cases depend on calls being queued rather than sent at once.
    ::setenv("RGPU_BATCH", "1", 1);
    ::setenv("RGPU_RECONNECT_SECONDS", "1", 1);
    client();
    std::fflush(stderr);
    ::_exit(g_failures.load() == 0 ? 0 : 1);
  }

  server(lfd);
  if (lfd >= 0) ::close(lfd);
  int status = 0;
  ::waitpid(pid, &status, 0);
  EXPECT(WIFEXITED(status) && WEXITSTATUS(status) == 0,
         "the client process reported a failure (see above)");
  std::fprintf(stderr, "   %s\n", g_failures.load() == before ? "ok" : "FAILED");
}

// Runs `fn` with this process's stderr sent to a file, and returns what was
// written. The shim reports connection failures only by logging them.
std::string capture_stderr(const std::function<void()>& fn) {
  char path[] = "/tmp/rgpu-thread-id-XXXXXX";
  int tfd = ::mkstemp(path);
  std::fflush(stderr);
  int saved = ::dup(2);
  ::dup2(tfd, 2);
  fn();
  std::fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  std::string out;
  char buf[4096];
  ::lseek(tfd, 0, SEEK_SET);
  for (ssize_t n; (n = ::read(tfd, buf, sizeof(buf))) > 0;) out.append(buf, n);
  ::close(tfd);
  ::unlink(path);
  std::fputs(out.c_str(), stderr);
  return out;
}

// --- the cases -------------------------------------------------------------

CUresult sync_call(uint32_t api) {
  Buffer req, rsp;
  return rgpu::call(api, req, &rsp);
}

// Thread A queues three calls that want no reply and returns without sending
// them. Thread B then makes a call that does want one, and its write carries
// A's three ahead of its own. Identity has to go with each frame, not with
// whoever wrote it. A stays alive until B is done, so that its exit - which
// the retirement case covers - adds nothing to the traffic here.
void client_a_then_b() {
  std::mutex mu;
  std::condition_variable cv;
  int stage = 0;
  auto advance = [&](int to) {
    std::lock_guard<std::mutex> lk(mu);
    stage = to;
    cv.notify_all();
  };
  auto await = [&](int at) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return stage >= at; });
  };
  std::thread a([&] {
    for (uint32_t i = 0; i < 3; i++) {
      Buffer req;
      req.put<uint32_t>(0xA0 + i);
      EXPECT(rgpu::call_async(kApiA, req) == CUDA_SUCCESS,
             "thread A could not queue a call");
    }
    advance(1);
    await(2);
  });
  await(1);
  std::thread b([] {
    EXPECT(sync_call(kApiB) == CUDA_SUCCESS, "thread B's call failed");
  });
  b.join();
  advance(2);
  a.join();
}

void check_a_then_b(const std::vector<Frame>& f) {
  for (int i = 0; i < 3; i++) {
    EXPECT(f[i].h.api_id == kApiA && (f[i].h.flags & rgpu::kFlagNoReply),
           "thread A's queued calls did not arrive first, unanswered");
  }
  EXPECT(f[3].h.api_id == kApiB && !(f[3].h.flags & rgpu::kFlagNoReply),
         "thread B's call did not arrive after A's");
  EXPECT(f[0].h.thread_id != 0, "thread A's frames carried no thread id");
  EXPECT(f[1].h.thread_id == f[0].h.thread_id &&
             f[2].h.thread_id == f[0].h.thread_id,
         "thread A's frames did not all carry the same thread id");
  EXPECT(f[3].h.thread_id != 0, "thread B's frame carried no thread id");
  EXPECT(f[3].h.thread_id != f[0].h.thread_id,
         "thread A's frames carried the id of thread B, which wrote them");
  // Minted from 1, in the order threads first call, so a server can keep them
  // in a small table.
  EXPECT(f[0].h.thread_id == 1 && f[3].h.thread_id == 2,
         "thread ids were not minted densely from 1");
}

void wire_cases() {
  run_case("a batch flushed by another thread keeps its own thread id",
           client_a_then_b, [](int& lfd) {
             int fd = accept_session(lfd, nullptr);
             if (fd < 0) return;
             std::vector<Frame> f = read_until_closed(fd);
             ::close(fd);
             EXPECT(f.size() == 4, "expected exactly four frames");
             if (f.size() == 4) check_a_then_b(f);
           });

  // The same traffic, but the connection breaks after the server has read all
  // four frames and before it answers. It says it completed the first; the
  // other three come again, and must be the frames that were sent, thread ids
  // included, or a replayed launch would run under another thread's context.
  run_case("a replayed frame carries the thread id it was sent with",
           client_a_then_b, [](int& lfd) {
             Handshake first{};
             int fd = accept_session(lfd, &first);
             if (fd < 0) return;
             std::vector<Frame> sent(4);
             bool got = true;
             for (auto& f : sent) got = got && read_frame(fd, &f);
             ::close(fd);
             EXPECT(got, "the client did not send four frames");
             if (!got) return;
             check_a_then_b(sent);

             fd = accept_within(lfd, 10000);
             EXPECT(fd >= 0, "the client did not come back after the break");
             if (fd < 0) return;
             Handshake again{};
             EXPECT(read_hello(fd, &again), "no handshake on reconnect");
             EXPECT(again.session_hi == first.session_hi &&
                        again.session_lo == first.session_lo,
                    "the client came back as a different session");
             send_hello(fd, rgpu::kProtocolVersion, true, sent[0].h.req_id);
             std::vector<Frame> replayed(3);
             got = true;
             for (auto& f : replayed) got = got && read_frame(fd, &f);
             EXPECT(got, "the client did not replay three frames");
             if (got) {
               for (int i = 0; i < 3; i++) {
                 EXPECT(replayed[i].h.thread_id == sent[i + 1].h.thread_id,
                        "a replayed frame carried a different thread id");
                 EXPECT(replayed[i].bytes() == sent[i + 1].bytes(),
                        "a replayed frame differs from the one first sent");
               }
               reply(fd, replayed[2], CUDA_SUCCESS);
             }
             EXPECT(read_until_closed(fd).empty(),
                    "the client sent more than the replay");
             ::close(fd);
           });

  // A thread that has exited gives its id back: the next frame anyone queues
  // is preceded by a notice naming it, so the server can drop what it kept for
  // that thread. The id itself is never handed out again.
  run_case("a finished thread's id is retired and never reused",
           [] {
             std::thread c([] {
               EXPECT(sync_call(kApiC) == CUDA_SUCCESS, "C's call failed");
             });
             c.join();
             std::thread d([] {
               EXPECT(sync_call(kApiD) == CUDA_SUCCESS, "D's call failed");
             });
             d.join();
             EXPECT(sync_call(kApiMain) == CUDA_SUCCESS, "main's call failed");
             // A thread that never called has no id and nothing to retire.
             std::thread quiet([] {});
             quiet.join();
             EXPECT(sync_call(kApiMain) == CUDA_SUCCESS, "main's call failed");
           },
           [](int& lfd) {
             int fd = accept_session(lfd, nullptr);
             if (fd < 0) return;
             std::vector<Frame> f = read_until_closed(fd);
             ::close(fd);
             const uint32_t want_api[] = {kApiC, rgpu::API_rgpu_thread_gone,
                                          kApiD, rgpu::API_rgpu_thread_gone,
                                          kApiMain, kApiMain};
             const uint32_t want_thread[] = {1, 2, 2, 3, 3, 3};
             EXPECT(f.size() == 6, "expected six frames: C, notice, D, "
                                   "notice, main, main");
             if (f.size() != 6) {
               for (const auto& x : f) {
                 std::fprintf(stderr, "   got api %#x thread %u\n", x.h.api_id,
                              x.h.thread_id);
               }
               return;
             }
             for (int i = 0; i < 6; i++) {
               EXPECT(f[i].h.api_id == want_api[i],
                      "frames arrived in the wrong order");
               EXPECT(f[i].h.thread_id == want_thread[i],
                      "a frame carried the wrong thread id");
             }
             // Each notice names exactly the thread that had just finished,
             // and wants no reply.
             const uint32_t gone_id[] = {1, 2};
             for (int k = 0; k < 2; k++) {
               const Frame& n = f[1 + 2 * k];
               EXPECT(n.h.flags & rgpu::kFlagNoReply,
                      "a retirement notice asked for a reply");
               Buffer b(n.payload);
               uint32_t count = 0, id = 0;
               EXPECT(b.get(&count) && count == 1 && b.get(&id) &&
                          id == gone_id[k] && b.size() == 8,
                      "a retirement notice did not name the finished thread");
             }
           });

  // What a client of this version sees from a server that speaks another.
  // A server from before protocol 3 closes on a mismatch without a word.
  run_case("a server that closes on the handshake is refused cleanly",
           [] {
             CUresult first = CUDA_SUCCESS, second = CUDA_SUCCESS;
             const std::string log = capture_stderr([&] {
               first = sync_call(kApiA);
               second = sync_call(kApiA);
             });
             EXPECT(first == CUDA_ERROR_NOT_INITIALIZED &&
                        second == CUDA_ERROR_NOT_INITIALIZED,
                    "calls against a refusing server did not fail as "
                    "not initialized");
             EXPECT(log.find("handshake with") != std::string::npos,
                    "the client did not report the failed handshake");
           },
           [](int& lfd) {
             int fd = accept_within(lfd, 10000);
             EXPECT(fd >= 0, "the client never connected");
             if (fd < 0) return;
             Handshake hello{};
             EXPECT(read_hello(fd, &hello) &&
                        hello.version == rgpu::kProtocolVersion,
                    "the client did not say it speaks this protocol");
             ::close(fd);
             EXPECT(accept_within(lfd, 1000) < 0,
                    "the client tried again after being refused");
           });

  // A server from protocol 3 on names its version before closing.
  run_case("a server that names another protocol is refused precisely",
           [] {
             CUresult r = CUDA_SUCCESS;
             const std::string log =
                 capture_stderr([&] { r = sync_call(kApiA); });
             EXPECT(r == CUDA_ERROR_NOT_INITIALIZED,
                    "a call against a mismatched server did not fail");
             const std::string want = "server speaks protocol 2, this client "
                                      "speaks " +
                                      std::to_string(rgpu::kProtocolVersion);
             EXPECT(log.find(want) != std::string::npos,
                    "the client did not name both protocol versions");
           },
           [](int& lfd) {
             int fd = accept_within(lfd, 10000);
             EXPECT(fd >= 0, "the client never connected");
             if (fd < 0) return;
             Handshake hello{};
             read_hello(fd, &hello);
             send_hello(fd, 2, false, 0);
             ::close(fd);
           });
}

// --- against the fake driver -----------------------------------------------

#define CHECK_RT(call)                                                    \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,  \
                   #call, e_);                                            \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

// The runtime caches, per thread, that the thread's context has been made
// current. cudaDeviceReset releases every primary context in the process, so
// it has to invalidate that cache on every thread, not only its own: a thread
// left believing in a released context sends its next call under it.
void runtime_cases() {
  std::mutex mu;
  std::condition_variable cv;
  int stage = 0;
  auto advance = [&](int to) {
    std::lock_guard<std::mutex> lk(mu);
    stage = to;
    cv.notify_all();
  };
  auto await = [&](int at) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return stage >= at; });
  };

  std::thread a([&] {
    void* p = nullptr;
    CHECK_RT(cudaSetDevice(0));
    CHECK_RT(cudaMalloc(&p, 64));
    CHECK_RT(cudaFree(p));
    advance(1);
    await(2);
    // Another thread has reset the device since this one selected it.
    void* q = nullptr;
    CHECK_RT(cudaMalloc(&q, 64));
    if (q) CHECK_RT(cudaFree(q));
  });
  await(1);
  CHECK_RT(cudaDeviceReset());
  advance(2);
  a.join();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "wire") {
    wire_cases();
  } else if (mode == "runtime") {
    runtime_cases();
  } else {
    std::fprintf(stderr, "usage: %s wire|runtime\n", argv[0]);
    return 2;
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures.load());
    return 1;
  }
  std::printf("\nPASS: thread ids (%s)\n", mode.c_str());
  return 0;
}
