// Checks that each client thread keeps its own CUDA context across the wire,
// without a GPU.
//
// CUDA's current context - and the stack it is the top of - belongs to the
// calling thread, but one server thread serves every thread of a client
// process. So the server keeps each client thread's context stack, puts its
// top back before running that thread's request, and reads it back
// afterwards. Without that, the last thread to
// select a context wins for everybody, silently: issue #2.
//
// Every case asserts what the driver documents - what cuCtxGetCurrent returns,
// which device an allocation is on - and never that a use under the wrong
// context fails, because on hardware with unified addressing it most likely
// does not.
//
// Runs against its own rgpu-server-fake with two devices, destroyed context
// handles reused, and batching on:
//
//   RGPU_FAKE_DEVICES=2 RGPU_FAKE_STATS=/tmp/s RGPU_MAX_CLIENT_THREADS=16
//     RGPU_FAKE_REUSE_CONTEXTS=1 rgpu-server-fake 9723 &
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9723 RGPU_BATCH=1
//     RGPU_FAKE_STATS=/tmp/s RGPU_MAX_CLIENT_THREADS=16 ./threadctx_smoke

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"
#include "common/wire.h"

namespace {

int g_failures = 0;
std::mutex g_fail_mu;

void fail_at(const char* file, int line, const char* what) {
  std::lock_guard<std::mutex> lk(g_fail_mu);
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
  g_failures++;
}

#define EXPECT(cond, what)                          \
  do {                                              \
    if (!(cond)) fail_at(__FILE__, __LINE__, what); \
  } while (0)

#define CHECK(call)                                                    \
  do {                                                                 \
    CUresult r_ = (call);                                              \
    if (r_ != CUDA_SUCCESS) {                                          \
      char msg_[256];                                                  \
      std::snprintf(msg_, sizeof(msg_), "%s -> %d", #call, (int)r_);   \
      fail_at(__FILE__, __LINE__, msg_);                               \
    }                                                                  \
  } while (0)

#define CHECK_RT(call)                                                 \
  do {                                                                 \
    cudaError_t e_ = (call);                                           \
    if (e_ != cudaSuccess) {                                           \
      char msg_[256];                                                  \
      std::snprintf(msg_, sizeof(msg_), "%s -> %d", #call, (int)e_);   \
      fail_at(__FILE__, __LINE__, msg_);                               \
    }                                                                  \
  } while (0)

// One of the fake driver's counters, as it last published them to
// RGPU_FAKE_STATS, or -1 if it cannot be read. The server publishes before it
// replies, so after a call has returned the file includes it.
long fake_counter(const char* name) {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  if (!path) return -1;
  std::FILE* f = std::fopen(path, "r");
  if (!f) return -1;
  char buf[512] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  const std::string line = std::string(" ") + buf;
  const std::string key = std::string(" ") + name + "=";
  const size_t at = line.find(key);
  if (at == std::string::npos) return -1;
  return std::strtol(line.c_str() + at + key.size(), nullptr, 10);
}

int ordinal_of(CUdeviceptr p) {
  int ordinal = -1;
  CHECK(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p));
  return ordinal;
}

// Lets two threads take turns: each waits for the stage it acts in. The
// point of every case is the order in which the threads' requests reach the
// server, so the order is forced rather than hoped for.
class Turns {
 public:
  void advance(int to) {
    std::lock_guard<std::mutex> lk(mu_);
    stage_ = to;
    cv_.notify_all();
  }
  void await(int at) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stage_ >= at; });
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int stage_ = 0;
};

// Test 2 of the design, and the smallest regression test for all of it: a
// context one thread makes current is still current on that thread after
// another thread has made a different one current.
void currency_is_read_back() {
  std::printf("-- a thread's current context survives another thread's\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  EXPECT(p0 && p1 && p0 != p1, "the two devices share a primary context");

  Turns turns;
  CUcontext a_sees = nullptr, b_sees = nullptr;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    turns.advance(1);
    turns.await(2);
    CHECK(cuCtxGetCurrent(&a_sees));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxGetCurrent(&b_sees));
  });
  a.join();
  b.join();
  EXPECT(a_sees == p0,
         "thread A's current context was changed by thread B's selection");
  EXPECT(b_sees == p1,
         "thread B's current context was changed by thread A's query");

  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// Test 1 of the design: issue #2's scenario, through the runtime API, and
// asserted by where the memory went. Thread A's second allocation used to land
// on device 1, because the runtime remembers per thread that its context is
// current and so sends no selection, while the one server thread serving both
// had been moved to device 1 by thread B.
void issue_scenario_by_placement() {
  std::printf("-- issue #2: an allocation lands on its own thread's device\n");
  Turns turns;
  void* a1 = nullptr;
  void* a2 = nullptr;
  void* b1 = nullptr;
  int a1_on = -1, a2_on = -1, b1_on = -1;
  std::thread a([&] {
    CHECK_RT(cudaSetDevice(0));
    CHECK_RT(cudaMalloc(&a1, 64));
    turns.advance(1);
    turns.await(2);
    CHECK_RT(cudaMalloc(&a2, 64));
    a1_on = ordinal_of(reinterpret_cast<CUdeviceptr>(a1));
    a2_on = ordinal_of(reinterpret_cast<CUdeviceptr>(a2));
    CHECK_RT(cudaFree(a1));
    CHECK_RT(cudaFree(a2));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK_RT(cudaSetDevice(1));
    CHECK_RT(cudaMalloc(&b1, 64));
    b1_on = ordinal_of(reinterpret_cast<CUdeviceptr>(b1));
    turns.advance(2);
    turns.await(3);
    CHECK_RT(cudaFree(b1));
  });
  a.join();
  b.join();
  EXPECT(a1_on == 0, "thread A's first allocation is not on device 0");
  EXPECT(b1_on == 1, "thread B's allocation is not on device 1");
  EXPECT(a2_on == 0,
         "thread A's allocation after thread B selected device 1 landed on "
         "device 1");
}

// Test 4 of the design: a run of calls queued without replies by one thread
// and flushed by another thread's call runs under the context of the thread
// that queued it. The memsets succeed under any context, because the driver
// infers placement from the pointer, so what is checked is what the fake saw:
// every device-memory write that ran under a context other than its memory's
// own is counted.
void batch_flushed_by_another_thread() {
  std::printf("-- a batch flushed by another thread runs under its own "
              "thread's context\n");
  constexpr int kQueued = 8;
  constexpr size_t kBytes = 64;
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  long before = -1, after = -1;
  CUdeviceptr mem = 0;
  CUcontext b_sees = nullptr;
  CUresult flushed_by = CUDA_ERROR_UNKNOWN;
  unsigned char last = 0;
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    CHECK(cuMemAlloc(&mem, kBytes));
    turns.advance(1);
    turns.await(2);
    // Every reply so far has been read, so nothing is still running.
    before = fake_counter("crossctx");
    for (int i = 1; i <= kQueued; i++) {
      CHECK(cuMemsetD8Async(mem, static_cast<unsigned char>(i), kBytes,
                            nullptr));
    }
    // Nothing from this thread until the other one has flushed the batch.
    turns.advance(3);
    turns.await(4);
    CHECK(cuMemcpyDtoH(&last, mem, 1));
    CHECK(cuMemFree(mem));
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(5);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    turns.advance(2);
    turns.await(3);
    flushed_by = cuCtxGetCurrent(&b_sees);
    after = fake_counter("crossctx");
    turns.advance(4);
    turns.await(5);
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();

  EXPECT(before >= 0 && after >= 0,
         "could not read the fake's counters (is RGPU_FAKE_STATS set?)");
  EXPECT(flushed_by == CUDA_SUCCESS,
         "the call that flushed the batch reported a failure");
  EXPECT(b_sees == p1, "the flushing thread's own context changed");
  EXPECT(last == kQueued, "the queued memsets did not all run");
  if (after != before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "%ld of the queued memsets ran under the flushing thread's "
                  "context instead of their own",
                  after - before);
    fail_at(__FILE__, __LINE__, msg);
  }

  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// The price of all this for a client with one thread: nothing. The server only
// makes a thread's context current when it differs from the one it last made
// current, so a thread calling on its own pays no cuCtxSetCurrent beyond the
// ones it asks for itself.
void one_thread_costs_no_switches() {
  std::printf("-- one thread's calls cost no context switches\n");
  CUcontext p0 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuCtxSetCurrent(p0));
  const long before = fake_counter("ctxsets");
  constexpr size_t kBytes = 256;
  for (int i = 0; i < 20; i++) {
    CUdeviceptr d = 0;
    CHECK(cuMemAlloc(&d, kBytes));
    CHECK(cuMemsetD8Async(d, 0x5a, kBytes, nullptr));
    unsigned char out = 0;
    CHECK(cuMemcpyDtoH(&out, d, 1));
    EXPECT(out == 0x5a, "a memset did not reach the memory");
    CUdevice dev = -1;
    CHECK(cuCtxGetDevice(&dev));
    CUcontext cur = nullptr;
    CHECK(cuCtxGetCurrent(&cur));
    CHECK(cuMemFree(d));
  }
  const long after = fake_counter("ctxsets");
  EXPECT(before >= 0 && after >= 0,
         "could not read the fake's counters (is RGPU_FAKE_STATS set?)");
  if (after != before) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "a single thread's 120 calls cost %ld context switches",
                  after - before);
    fail_at(__FILE__, __LINE__, msg);
  }
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(0));
}

// A context one thread made current and another thread then destroyed, and
// whose address the next context created then got. The first thread's next
// call must not run under that new context, nor under whatever the other
// thread left current: CUDA leaves a destroyed context current to the threads
// it was current to and fails their calls with
// CUDA_ERROR_CONTEXT_IS_DESTROYED, and the thread can select again.
//
// The server runs with RGPU_FAKE_REUSE_CONTEXTS=1, so the fake hands the
// destroyed context's handle to the next create, as a real driver may.
void context_destroyed_by_another_thread() {
  std::printf("-- a context destroyed by another thread is not replaced by "
              "the context that took its address\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, other = nullptr;
  CUresult after_destroy = CUDA_SUCCESS;
  int landed_on = -1;
  CUcontext a_sees = reinterpret_cast<CUcontext>(0xbadull);
  int reselected_on = -1;
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    turns.advance(1);
    turns.await(2);
    CUdeviceptr d = 0;
    after_destroy = cuMemAlloc(&d, 64);
    if (after_destroy == CUDA_SUCCESS) {
      landed_on = ordinal_of(d);
      cuMemFree(d);
    }
    CHECK(cuCtxGetCurrent(&a_sees));
    CHECK(cuCtxSetCurrent(p0));
    CUdeviceptr e = 0;
    CHECK(cuMemAlloc(&e, 64));
    if (e) {
      reselected_on = ordinal_of(e);
      CHECK(cuMemFree(e));
    }
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    CHECK(cuCtxDestroy(made));
    // Pushed on top of p1, on device 1, so that running under it shows.
    CHECK(cuCtxCreate(&other, 0, 1));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxDestroy(other));
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  EXPECT(other == made,
         "the fake did not reuse the destroyed context's handle (is "
         "RGPU_FAKE_REUSE_CONTEXTS=1 set on the server?)");
  if (after_destroy != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "a call after another thread destroyed this thread's context "
                  "returned %d (on device %d), not "
                  "CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)after_destroy, landed_on);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(a_sees != other && a_sees != p1,
         "a thread whose context was destroyed was bound to another thread's "
         "context");
  EXPECT(reselected_on == 0,
         "a thread whose context was destroyed could not select another");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// A thread whose context was destroyed under it recovers the way it would in
// CUDA: its very first call can be a selection, which succeeds, and what it
// pushes on top of the destroyed context can be popped to find the destroyed
// one current again.
void recovery_after_destroy() {
  std::printf("-- a thread recovers from a context destroyed under it\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, popped = nullptr;
  CUresult pushed = CUDA_ERROR_UNKNOWN;
  CUresult under_push = CUDA_ERROR_UNKNOWN, after_pop = CUDA_SUCCESS;
  int pushed_on = -1, set_on = -1;
  CUcontext at_end = reinterpret_cast<CUcontext>(0xbadull);
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    turns.advance(1);
    turns.await(2);
    pushed = cuCtxPushCurrent(p0);
    CUdeviceptr d = 0;
    under_push = cuMemAlloc(&d, 64);
    if (under_push == CUDA_SUCCESS) {
      pushed_on = ordinal_of(d);
      CHECK(cuMemFree(d));
    }
    CHECK(cuCtxPopCurrent(&popped));
    CUdeviceptr e = 0;
    after_pop = cuMemAlloc(&e, 64);
    if (after_pop == CUDA_SUCCESS) cuMemFree(e);
    // Replaces the destroyed context at the top of the stack.
    CHECK(cuCtxSetCurrent(p1));
    CUdeviceptr f = 0;
    CHECK(cuMemAlloc(&f, 64));
    if (f) {
      set_on = ordinal_of(f);
      CHECK(cuMemFree(f));
    }
    CHECK(cuCtxSetCurrent(nullptr));
    CHECK(cuCtxGetCurrent(&at_end));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxDestroy(made));
    turns.advance(2);
    turns.await(3);
  });
  a.join();
  b.join();
  if (pushed != CUDA_SUCCESS) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "the first call after this thread's context was destroyed, a "
                  "push, returned %d",
                  (int)pushed);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(under_push == CUDA_SUCCESS && pushed_on == 0,
         "an allocation under the pushed context did not land on its device");
  EXPECT(popped == p0, "the pop did not return the pushed context");
  if (after_pop != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "after popping back to the destroyed context an allocation "
                  "returned %d, not CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)after_pop);
    fail_at(__FILE__, __LINE__, msg);
  }
  EXPECT(set_on == 1,
         "selecting over a destroyed context did not take effect");
  EXPECT(at_end == nullptr,
         "the thread's stack was not empty after it popped everything");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// Issue #2's regression in C1, exactly: a thread that creates a context and
// pushes another on top of it, with a background thread's context-free call in
// between, pops back to the context it created. The background thread has no
// context of its own, and making that true on the one server thread must not
// throw away the first thread's stack.
void push_pop_across_a_background_call() {
  std::printf("-- a thread's context stack survives another thread's "
              "context-free call\n");
  CUcontext p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext made = nullptr, popped = nullptr;
  CUcontext after_pop = reinterpret_cast<CUcontext>(0xbadull);
  std::thread a([&] {
    CHECK(cuCtxCreate(&made, 0, 0));
    CHECK(cuCtxPushCurrent(p1));
    turns.advance(1);
    turns.await(2);
    CHECK(cuCtxPopCurrent(&popped));
    CHECK(cuCtxGetCurrent(&after_pop));
    CHECK(cuCtxDestroy(made));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    int count = 0;
    CHECK(cuDeviceGetCount(&count));
    turns.advance(2);
    turns.await(3);
  });
  a.join();
  b.join();
  EXPECT(popped == p1, "the pop did not return the pushed context");
  EXPECT(after_pop == made,
         "popping did not make the context underneath current again: another "
         "thread's call threw the stack away");
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// Test 3 of the design: context stacks are per thread. A pushes one context
// over its own; B, with a different context of its own, pushes and pops; A
// pops and is back where it started, and so is B.
void stacks_are_per_thread() {
  std::printf("-- each thread pops back to its own context\n");
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));

  Turns turns;
  CUcontext a_popped = nullptr, b_popped = nullptr;
  CUcontext a_after = reinterpret_cast<CUcontext>(0xbadull);
  CUcontext b_after = reinterpret_cast<CUcontext>(0xbadull);
  std::thread a([&] {
    CHECK(cuCtxSetCurrent(p0));
    CHECK(cuCtxPushCurrent(p1));
    turns.advance(1);
    turns.await(2);
    CHECK(cuCtxPopCurrent(&a_popped));
    CHECK(cuCtxGetCurrent(&a_after));
    CHECK(cuCtxSetCurrent(nullptr));
    turns.advance(3);
  });
  std::thread b([&] {
    turns.await(1);
    CHECK(cuCtxSetCurrent(p1));
    CHECK(cuCtxPushCurrent(p0));
    CHECK(cuCtxPopCurrent(&b_popped));
    CHECK(cuCtxGetCurrent(&b_after));
    turns.advance(2);
    turns.await(3);
    CHECK(cuCtxSetCurrent(nullptr));
  });
  a.join();
  b.join();
  EXPECT(b_popped == p0, "thread B's pop did not return what it pushed");
  EXPECT(b_after == p1, "thread B did not pop back to its own context");
  EXPECT(a_popped == p1, "thread A's pop did not return what it pushed");
  EXPECT(a_after == p0,
         "thread A did not pop back to its own context after thread B pushed "
         "and popped");
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));
}

// --- the slot table, from a connection of our own ---------------------------
//
// A real client mints thread ids and announces their end itself, and cannot be
// made to send a call after its thread was announced gone, or to have more
// threads than a server allows, on cue. These cases speak the protocol
// directly, as their own session, and choose the thread ids.

class RawSession {
 public:
  ~RawSession() {
    if (fd_ >= 0) ::close(fd_);
  }

  bool open() {
    const char* env = std::getenv("RGPU_SERVER");
    std::string spec = env ? env : "127.0.0.1:9713";
    const size_t c = spec.rfind(':');
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(spec.substr(0, c).c_str(), spec.substr(c + 1).c_str(),
                      &hints, &res) != 0) {
      return false;
    }
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
      fd_ = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd_ < 0) continue;
      if (::connect(fd_, ai->ai_addr, ai->ai_addrlen) == 0) break;
      ::close(fd_);
      fd_ = -1;
    }
    ::freeaddrinfo(res);
    if (fd_ < 0) return false;
    timeval tv{5, 0};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    static std::atomic<uint64_t> n{0};
    rgpu::Handshake hello{};
    hello.magic = rgpu::kMagicHello;
    hello.version = rgpu::kProtocolVersion;
    hello.session_hi = 0x7c7c000000000000ull ^ static_cast<uint64_t>(::getpid());
    hello.session_lo = ++n;
    rgpu::HandshakeReply reply{};
    return rgpu::write_exact(fd_, &hello, sizeof(hello)) &&
           rgpu::read_exact(fd_, &reply, sizeof(reply)) &&
           reply.version == rgpu::kProtocolVersion && reply.resumed == 0;
  }

  bool send(uint32_t api, uint32_t thread, const rgpu::Buffer& payload,
            bool no_reply) {
    rgpu::ReqHeader h{};
    h.magic = rgpu::kMagicReq;
    h.api_id = api;
    h.req_id = next_++;
    h.flags = no_reply ? static_cast<uint32_t>(rgpu::kFlagNoReply) : 0u;
    h.thread_id = thread;
    h.payload_len = static_cast<uint32_t>(payload.size());
    return rgpu::send_frame(fd_, h, payload);
  }

  // A call that replies. CUDA_ERROR_UNKNOWN if the connection failed.
  CUresult call(uint32_t api, uint32_t thread, const rgpu::Buffer& payload,
                rgpu::Buffer* rsp) {
    if (!send(api, thread, payload, false)) return CUDA_ERROR_UNKNOWN;
    rgpu::RspHeader rh{};
    std::vector<uint8_t> body;
    if (!rgpu::recv_frame(fd_, rgpu::kMagicRsp, &rh, &body)) {
      return CUDA_ERROR_UNKNOWN;
    }
    *rsp = rgpu::Buffer(std::move(body));
    return static_cast<CUresult>(rh.result);
  }

  CUresult set_current(uint32_t thread, CUcontext ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(ctx));
    return call(rgpu::API_cuCtxSetCurrent, thread, req, &rsp);
  }

  // The thread's current context, or a marker value if the call failed.
  CUcontext get_current(uint32_t thread, CUresult* r = nullptr) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    const CUresult got = call(rgpu::API_cuCtxGetCurrent, thread, req, &rsp);
    if (r) *r = got;
    uint64_t h = 0;
    if (got != CUDA_SUCCESS || !rsp.get(&h)) {
      return reinterpret_cast<CUcontext>(0xbadull);
    }
    return reinterpret_cast<CUcontext>(h);
  }

  CUresult retain(uint32_t thread, int dev, CUcontext* ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    req.put<int>(dev);
    const CUresult r = call(rgpu::API_cuDevicePrimaryCtxRetain, thread, req, &rsp);
    uint64_t h = 0;
    if (r == CUDA_SUCCESS && rsp.get(&h)) *ctx = reinterpret_cast<CUcontext>(h);
    return r;
  }

  CUresult release(uint32_t thread, int dev) {
    rgpu::Buffer req, rsp;
    req.put<int>(dev);
    return call(rgpu::API_cuDevicePrimaryCtxRelease_v2, thread, req, &rsp);
  }

  CUresult create(uint32_t thread, int dev, CUcontext* ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    req.put<unsigned int>(0);
    req.put<int>(dev);
    const CUresult r = call(rgpu::API_cuCtxCreate_v2, thread, req, &rsp);
    uint64_t h = 0;
    if (r == CUDA_SUCCESS && rsp.get(&h)) *ctx = reinterpret_cast<CUcontext>(h);
    return r;
  }

  CUresult destroy(uint32_t thread, CUcontext ctx) {
    rgpu::Buffer req, rsp;
    req.put<uint64_t>(reinterpret_cast<uint64_t>(ctx));
    return call(rgpu::API_cuCtxDestroy_v2, thread, req, &rsp);
  }

  CUresult alloc(uint32_t thread, size_t bytes, CUdeviceptr* ptr) {
    rgpu::Buffer req, rsp;
    req.put<uint8_t>(1);
    req.put<size_t>(bytes);
    const CUresult r = call(rgpu::API_cuMemAlloc_v2, thread, req, &rsp);
    if (r == CUDA_SUCCESS && !rsp.get(ptr)) return CUDA_ERROR_UNKNOWN;
    return r;
  }

  CUresult free(uint32_t thread, CUdeviceptr ptr) {
    rgpu::Buffer req, rsp;
    req.put<CUdeviceptr>(ptr);
    return call(rgpu::API_cuMemFree_v2, thread, req, &rsp);
  }

  // Announces threads gone, sent by `sender` without a reply, as the client
  // sends it.
  bool gone(uint32_t sender, const std::vector<uint32_t>& ids) {
    rgpu::Buffer req;
    req.put<uint32_t>(static_cast<uint32_t>(ids.size()));
    for (uint32_t id : ids) req.put<uint32_t>(id);
    return send(rgpu::API_rgpu_thread_gone, sender, req, true);
  }

 private:
  int fd_ = -1;
  uint32_t next_ = 1;
};

// A primary context one session's thread had current, destroyed by another
// session's last release, and brought back by a third session that then also
// creates a context of its own. The first thread never learns of any of it:
// its session saw neither the release nor the retain. Its next call must not
// run under whatever now answers to the handle it saved - on hardware that
// could be a new context at the old address, on the fake it is the revived
// primary context the third session holds - but fail the way a call on a
// destroyed context fails.
//
// Runs before anything in this process retains device 0, so that the second
// session's release really is the last one.
void primary_destroyed_by_another_session() {
  std::printf("-- a primary context destroyed by another session is not "
              "rebound by its old handle\n");
  constexpr uint32_t kHolder = 1, kOther = 2;
  RawSession s1, s2, s3;
  if (!s1.open() || !s2.open() || !s3.open()) {
    fail_at(__FILE__, __LINE__, "could not open sessions of our own");
    return;
  }
  CUcontext p0 = nullptr, s2_p0 = nullptr, s3_p0 = nullptr, s3_made = nullptr;
  CHECK(s1.retain(kHolder, 0, &p0));
  CHECK(s1.set_current(kHolder, p0));
  CHECK(s2.retain(kHolder, 0, &s2_p0));
  // Not the last release: the second session still holds one.
  CHECK(s1.release(kHolder, 0));
  // Another thread of the first session calls, so the holder's context is no
  // longer the one its session's server thread has current.
  CUresult r = CUDA_ERROR_UNKNOWN;
  s1.get_current(kOther, &r);
  CHECK(r);
  // The last release in the process: device 0's primary context is destroyed.
  CHECK(s2.release(kHolder, 0));
  CHECK(s3.retain(kHolder, 0, &s3_p0));
  CHECK(s3.create(kHolder, 0, &s3_made));

  CUdeviceptr d = 0;
  const CUresult alloc = s1.alloc(kHolder, 64, &d);
  if (alloc == CUDA_SUCCESS) s1.free(kHolder, d);
  if (alloc != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "a call from a thread whose primary context another session "
                  "destroyed returned %d, not CUDA_ERROR_CONTEXT_IS_DESTROYED",
                  (int)alloc);
    fail_at(__FILE__, __LINE__, msg);
  }
  const CUcontext sees = s1.get_current(kHolder, &r);
  CHECK(r);
  EXPECT(sees != p0 && sees != s3_made,
         "a thread whose primary context another session destroyed was bound "
         "to a context by its old handle");

  CHECK(s3.destroy(kHolder, s3_made));
  CHECK(s3.release(kHolder, 0));
}

// Late thread-local destructors on a client thread can call after the client
// has announced the thread gone. Such a call finds the context its thread had,
// not an empty slot and not another thread's context.
void late_call_after_notice(CUcontext p0, CUcontext p1) {
  std::printf("-- a call after its thread was announced gone keeps its "
              "context\n");
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  CHECK(s.set_current(7, p0));
  CHECK(s.set_current(8, p1));
  EXPECT(s.gone(8, {7}), "could not send the notice");
  CUresult r = CUDA_SUCCESS;
  EXPECT(s.get_current(7, &r) == p0,
         "a late call from a thread announced gone lost its context");
  CHECK(r);
  EXPECT(s.gone(8, {7}), "could not send the notice again");
  EXPECT(s.get_current(8) == p1, "the other thread's context changed");
}

// The retired slots are bounded. A late call from a thread retired longer ago
// than the most recent 64 finds an empty slot - no context - and never another
// thread's. The bound is kRetiredSlots in server/main.cpp.
void retired_slots_are_bounded(CUcontext p0, CUcontext p1) {
  std::printf("-- only the most recently retired threads keep their "
              "context\n");
  constexpr uint32_t kRetiredSlots = 64;
  constexpr uint32_t kFirst = 100;
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  // One more than the bound, each retired straight after it selects, so the
  // live table never holds more than two and the cap is not what is tested.
  for (uint32_t id = kFirst; id <= kFirst + kRetiredSlots; id++) {
    CHECK(s.set_current(id, p0));
    EXPECT(s.gone(id, {id}), "could not send a notice");
  }
  // Each late call below brings its thread back to the live table, which
  // leaves the retired set as it was, and no notice follows.
  CHECK(s.set_current(99, p1));
  EXPECT(s.get_current(kFirst + 1) == p0,
         "the oldest thread still within the bound lost its context");
  EXPECT(s.get_current(kFirst + kRetiredSlots) == p0,
         "the most recently retired thread lost its context");
  CUresult r = CUDA_SUCCESS;
  EXPECT(s.get_current(kFirst, &r) == nullptr,
         "a late call from a thread retired past the bound did not get an "
         "empty slot");
  CHECK(r);
}

// A client mints thread ids and nothing authenticates it, so the live slots a
// session may hold are capped (RGPU_MAX_CLIENT_THREADS). A thread past the cap
// is refused, and a notice still gets through and makes room.
void live_slots_are_capped() {
  std::printf("-- a session's live client threads are capped\n");
  const char* env = std::getenv("RGPU_MAX_CLIENT_THREADS");
  const uint32_t cap = env ? static_cast<uint32_t>(std::atol(env)) : 0;
  if (cap == 0 || cap > 1024) {
    fail_at(__FILE__, __LINE__,
            "RGPU_MAX_CLIENT_THREADS must name the server's cap, and a small "
            "one");
    return;
  }
  RawSession s;
  if (!s.open()) {
    fail_at(__FILE__, __LINE__, "could not open a session of our own");
    return;
  }
  for (uint32_t id = 1; id <= cap; id++) {
    CUresult r = CUDA_ERROR_UNKNOWN;
    s.get_current(id, &r);
    if (r != CUDA_SUCCESS) {
      fail_at(__FILE__, __LINE__, "a thread within the cap was refused");
      return;
    }
  }
  CUresult r = CUDA_SUCCESS;
  s.get_current(cap + 1, &r);
  EXPECT(r == CUDA_ERROR_INVALID_VALUE,
         "a thread past the cap was not refused with CUDA_ERROR_INVALID_VALUE");
  // Sent by the refused thread itself, which is how a client whose new thread
  // has just announced an old one would send it.
  EXPECT(s.gone(cap + 1, {1}), "could not send a notice");
  s.get_current(cap + 1, &r);
  EXPECT(r == CUDA_SUCCESS,
         "a notice from a thread past the cap did not make room for it");
}

}  // namespace

int main() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    std::fprintf(stderr, "FAIL: needs a server with two devices "
                         "(RGPU_FAKE_DEVICES=2)\n");
    return 1;
  }

  primary_destroyed_by_another_session();
  currency_is_read_back();
  issue_scenario_by_placement();
  batch_flushed_by_another_thread();
  one_thread_costs_no_switches();
  context_destroyed_by_another_thread();
  recovery_after_destroy();
  push_pop_across_a_background_call();
  stacks_are_per_thread();

  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  late_call_after_notice(p0, p1);
  retired_slots_are_bounded(p0, p1);
  live_slots_are_capped();
  CHECK(cuDevicePrimaryCtxRelease(0));
  CHECK(cuDevicePrimaryCtxRelease(1));

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: each client thread keeps its own context\n");
  return 0;
}
