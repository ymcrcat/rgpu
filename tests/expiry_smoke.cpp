// Checks that a client which dies takes its GPU resources with it, and that
// a client which is still there keeps its own.
//
// The resources belong to the server process, not to the client, so nothing
// about a client exiting reclaims them: the session record goes, the serving
// thread goes, and the device memory, the retained primary context and the
// library handles all stay exactly where they were. This test is the one that
// can see that, because the fake driver underneath the server counts what it
// has handed out and not got back (tests/fake_stats.h).
//
// Two sessions, one server:
//
//   - this process takes some resources and keeps its connection open;
//   - a child process takes more and is killed without freeing anything;
//   - after the grace period the counts must be back to exactly what this
//     process holds, and this process must still work.
//
// The second half matters as much as the first. A primary context is shared
// between sessions, so a cleanup that releases one retain too many would take
// the context away from a session still using it, which is worse than the leak
// it was fixing.
//
//   RGPU_SESSION_GRACE=3 RGPU_FAKE_STATS=/tmp/s rgpu-server-fake &
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./expiry_smoke

#include <signal.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <sys/wait.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#ifdef RGPU_EXPIRY_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef RGPU_EXPIRY_CUBLASLT
#include <cublasLt.h>
#endif
#ifdef RGPU_EXPIRY_CUDNN
#include <cudnn.h>
#endif

namespace {

int g_failures = 0;

#define CHECK(call)                                                      \
  do {                                                                   \
    CUresult r_ = (call);                                                \
    if (r_ != CUDA_SUCCESS) {                                            \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, \
                   #call, r_);                                           \
      g_failures++;                                                      \
    }                                                                    \
  } while (0)

const size_t kBytes = 4096;

// A fatbin header the shim can size, with a filler body: enough to ship, and
// what the fake driver accepts. The same shape as the one in launch_smoke.
std::vector<unsigned char> fake_fatbin() {
  std::vector<unsigned char> img(16 + 256, 0xAB);
  const unsigned int magic = 0xBA55ED50u;
  const unsigned short version = 1, header_size = 16;
  const unsigned long long body = 256;
  std::memcpy(img.data() + 0, &magic, sizeof(magic));
  std::memcpy(img.data() + 4, &version, sizeof(version));
  std::memcpy(img.data() + 6, &header_size, sizeof(header_size));
  std::memcpy(img.data() + 8, &body, sizeof(body));
  return img;
}

// The counters as the fake driver last published them. One line of
// name=value pairs; compared whole, so a kind this test never thought about
// still has to come back to where it started.
std::string read_stats() {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  if (!path) return "";
  std::FILE* f = std::fopen(path, "r");
  if (!f) return "";
  char buf[512] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  std::string s(buf);
  // The line ends with counters of calls rather than of resources, which move
  // with every context switch and are no part of what has to come back.
  const size_t calls = s.find(" ctxsets=");
  if (calls != std::string::npos) s.erase(calls);
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  return s;
}

long field(const std::string& stats, const char* name) {
  const std::string key = std::string(name) + "=";
  size_t at = stats.find(key);
  if (at == std::string::npos) return -1;
  return std::strtol(stats.c_str() + at + key.size(), nullptr, 10);
}

// A counter of calls rather than of resources, which read_stats leaves out.
long call_count(const char* name) {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  std::FILE* f = path ? std::fopen(path, "r") : nullptr;
  if (!f) return -1;
  char buf[512] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  return field(std::string(" ") + buf, (std::string(" ") + name).c_str());
}

// Called in a freshly forked child, before it execs. The children wait in
// pause() for the parent to kill them, so a parent that crashes instead would
// leave them running, holding sessions and pipe ends open. On Linux, where
// these tests run, they are made to die with it: the death signal survives
// the exec, and the parent is checked again in case it died before the signal
// was armed.
void die_with_parent(pid_t parent) {
#ifdef __linux__
  if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
    std::perror("prctl");
    ::_exit(127);
  }
  if (::getppid() != parent) ::_exit(127);
#else
  (void)parent;
#endif
}

// Everything a client can take and then die holding. Called in both processes:
// the child's copy is what has to come back, this process's copy is what must
// not.
struct Held {
  CUdeviceptr a = 0, b = 0;
  CUstream stream = nullptr;
  CUevent event = nullptr;
  // A captured graph, a clone of it and an executable made from it. These
  // belong to no context (real-GPU probe, check 6), so a context destroy, a
  // reset or a last release does not take them with it: a session that tears
  // its context down while still holding them has to destroy them itself
  // (destroy_held_graphs), or they leak. A session that simply holds them until
  // it expires has the server release them like anything else.
  CUgraph graph = nullptr;
  CUgraphExec exec = nullptr;
  CUgraph clone = nullptr;
};

// What the driver hands out, in whatever context is current.
void take_driver_resources(Held* held) {
  CHECK(cuMemAlloc(&held->a, kBytes));
  CHECK(cuMemAlloc(&held->b, kBytes));
  CHECK(cuStreamCreate(&held->stream, 0));
  CHECK(cuEventCreate(&held->event, 0));

  // A module, and a graph with an executable made from it. These are the
  // release paths that nothing else here would exercise, and a graph exec in
  // particular can own a great deal of device memory.
  const std::vector<unsigned char> image = fake_fatbin();
  CUmodule mod = nullptr;
  CHECK(cuModuleLoadData(&mod, image.data()));
  CHECK(cuStreamBeginCapture(held->stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(held->stream, &held->graph));
  CHECK(cuGraphInstantiateWithFlags(&held->exec, held->graph, 0));
  CHECK(cuGraphClone(&held->clone, held->graph));
}

// Destroys the context-less graph objects a Held carries. No context teardown
// takes them, so a session that is about to destroy, reset or release its
// context while holding them destroys them here first. A context must be
// current.
void destroy_held_graphs(const Held& held) {
  CHECK(cuGraphExecDestroy(held.exec));
  CHECK(cuGraphDestroy(held.clone));
  CHECK(cuGraphDestroy(held.graph));
}

void take_resources(CUdevice dev, Held* held, bool own_context) {
  CUcontext ctx = nullptr;
  if (own_context) {
    // A context of its own, on top of the shared one, so the cleanup has to
    // free what is inside it before destroying it.
    CHECK(cuCtxCreate(&ctx, 0, dev));
  }
  take_driver_resources(held);
#ifdef RGPU_EXPIRY_CUBLAS
  cublasHandle_t blas = nullptr;
  if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: cublasCreate\n");
    g_failures++;
  }
#endif
#ifdef RGPU_EXPIRY_CUBLASLT
  cublasLtHandle_t lt = nullptr;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: cublasLtCreate\n");
    g_failures++;
  }
#endif
#ifdef RGPU_EXPIRY_CUDNN
  cudnnHandle_t dnn = nullptr;
  if (cudnnCreate(&dnn) != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: cudnnCreate\n");
    g_failures++;
  }
#endif
}

// The other half of the reset ruling: alone on a server, a client may reset
// the device. Everything taken here is in the primary context, so the reset
// destroys all of it, and a server that went on to free any of it at expiry
// would show as stale. The two retains are another matter: a reset does not
// release a primary context, so the session still holds both, and expiry has
// to release exactly those two - a server that forgot them would leak them,
// and one that released a third would show as an over-release.
//
// Not only memory. The server forgets every kind of thing it recorded in the
// primary context when the device is reset, so every kind the driver hands
// out is taken here: a driver whose reset left the streams, events and modules
// behind would show them as held at the end, and a server that went on to
// release them would show as stale. The captured graph, its clone and the
// executable are the exception: they belong to no context (real-GPU probe,
// check 6), so no reset takes them, and this session destroys them itself
// before resetting. Library handles are not taken: they are the libraries'
// objects, not the driver's, and the fake libraries do not model a reset.
int reset_alone() {
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CHECK(cuCtxSetCurrent(primary));
  CUdeviceptr a = 0, b = 0;
  CHECK(cuMemAlloc(&a, kBytes));
  CHECK(cuMemAlloc(&b, kBytes));
  Held held;
  take_driver_resources(&held);
  // The graph objects belong to no context, so the reset will not take them;
  // destroy them here, while the primary context is still current.
  destroy_held_graphs(held);

  const CUresult r = cuDevicePrimaryCtxReset(dev);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "FAIL: the only session on the server was refused a device "
                 "reset (%d)\n",
                 r);
    g_failures++;
  }

  // The driver's side of it, which the server has to agree with: everything
  // in the context is gone, and both retains are still held. The header is
  // explicit that "resetting the primary context does not release it", so the
  // session still owes its two releases, and run_smoke.sh checks that expiry
  // pays them - no more, no fewer.
  const std::string after = read_stats();
  if (!after.empty()) {
    for (const char* kind :
         {"allocs", "modules", "streams", "events", "graphs", "execs"}) {
      if (field(after, kind) != 0) {
        std::fprintf(stderr, "FAIL: the reset left %s behind: %s\n", kind,
                     after.c_str());
        g_failures++;
      }
    }
    if (field(after, "retains") != 2) {
      std::fprintf(stderr,
                   "FAIL: a reset should leave both retains held: %s\n",
                   after.c_str());
      g_failures++;
    }
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  // Left exactly as it is: the server has to be right about what it owes.
  std::printf("PASS: the only session on a server may reset the device\n");
  return 0;
}

// Releasing the last retain on a primary context resets it, in the driver's
// words "automatically ... once the last reference to it is released", so
// everything in it is destroyed without the client freeing any of it. Alone
// on a server this session's release is the last one, and the server has to
// notice: if it still thought it held the memory, stream and module, it would
// free them at expiry, and the fake driver would count every one as stale.
int release_alone() {
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CHECK(cuCtxSetCurrent(primary));
  Held held;
  take_driver_resources(&held);
  // The graph objects belong to no context, so the last release will not take
  // them; destroy them here, while the primary context is still current.
  destroy_held_graphs(held);
  CHECK(cuDevicePrimaryCtxRelease(dev));

  const std::string after = read_stats();
  if (!after.empty()) {
    for (const char* kind : {"allocs", "retains", "modules", "streams",
                             "events", "graphs", "execs"}) {
      if (field(after, kind) != 0) {
        std::fprintf(stderr,
                     "FAIL: releasing the last retain left %s behind: %s\n",
                     kind, after.c_str());
        g_failures++;
      }
    }
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: releasing the last retain destroys what was in it\n");
  return 0;
}

// cuCtxDetach destroys a created context, whose usage count is 1, and
// everything in it, as cuCtxDestroy does. The server has to forget the context
// and what was in it: if it did not, it would free the memory and destroy the
// context again at expiry - stale calls, and once the driver hands the
// context's address to another session's cuCtxCreate, a destroy of that
// session's context.
int detach_alone() {
  CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, 0, 0));
  CUdeviceptr d = 0;
  CHECK(cuMemAlloc(&d, kBytes));
  CHECK(cuCtxDetach(ctx));

  const std::string after = read_stats();
  if (!after.empty()) {
    for (const char* kind : {"allocs", "contexts"}) {
      if (field(after, kind) != 0) {
        std::fprintf(stderr,
                     "FAIL: detaching the context left %s behind: %s\n", kind,
                     after.c_str());
        g_failures++;
      }
    }
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: detaching a created context destroys what was in it\n");
  return 0;
}

// A graph captured on a stream, a clone of a graph and an executable made
// from one belong to no context at all (real-GPU probe, check 6): they outlive
// the capture context and every other. Made here on device 0's objects while
// device 1 is current, they survive this session's last release on device 0 -
// which does destroy device 0's stream - so this session destroys them itself
// rather than leak them. The server records them under device 0 as a heuristic
// for when to give up on them; it forgets them on the release and never frees
// them again, so nothing is stale either way.
//
// Needs a server with two devices (RGPU_FAKE_DEVICES=2).
int derived_alone() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    std::fprintf(stderr, "FAIL: derived needs a server with two devices\n");
    return 1;
  }
  CUcontext p0 = nullptr, p1 = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
  CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
  CHECK(cuCtxSetCurrent(p0));
  CUstream s = nullptr;
  CUgraph g = nullptr;
  CHECK(cuStreamCreate(&s, 0));
  CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(s, &g));

  CHECK(cuCtxSetCurrent(p1));
  CUgraph clone = nullptr, captured = nullptr;
  CUgraphExec exec = nullptr;
  CHECK(cuGraphClone(&clone, g));
  CHECK(cuGraphInstantiateWithFlags(&exec, clone, 0));
  CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(s, &captured));

  // The graph objects belong to no context, so releasing device 0 leaves them;
  // destroy them here (device 1 is current), so only device 0's stream goes
  // with the release.
  CHECK(cuGraphExecDestroy(exec));
  CHECK(cuGraphDestroy(clone));
  CHECK(cuGraphDestroy(g));
  CHECK(cuGraphDestroy(captured));

  CHECK(cuDevicePrimaryCtxRelease(0));
  // The premise: device 0's stream went with device 0, and freeing the
  // context-less graph objects above was neither stale nor left anything.
  const std::string after = read_stats();
  for (const char* kind : {"streams", "graphs", "execs", "stale"}) {
    if (field(after, kind) != 0) {
      std::fprintf(stderr,
                   "FAIL: releasing device 0 left %s behind: %s\n", kind,
                   after.c_str());
      g_failures++;
    }
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  // Device 1's retain is left for expiry to release.
  std::printf("PASS: graph objects made on device 0 belong to no context and "
              "are freed by their maker\n");
  return 0;
}

// Two client threads, each with its own device's primary context current and
// one with a created context pushed on top, each take everything a client can
// take and then the process ends holding all of it. The server keeps a slot
// for each thread naming the contexts, and the serving thread has the last
// thread's context current when the session expires; neither may stand in
// the way of the release, which has to give back both devices' retains and
// free everything once. The slots own nothing, so nothing may be released
// twice either.
//
// Alone on a server with two devices (RGPU_FAKE_DEVICES=2). run_smoke.sh
// checks the counters once the session has expired: retains is both devices'
// together, and a device released more times than it was retained would show
// as an over-release rather than hide in the sum.
int threads_alone() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    std::fprintf(stderr, "FAIL: threads needs a server with two devices\n");
    return 1;
  }

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

  // The threads take turns, so their CHECKs never run at once.
  Held a_held, b_held;
  CUdeviceptr a_extra = 0, b_extra = 0;
  std::thread a([&] {
    CUcontext p0 = nullptr;
    CHECK(cuDevicePrimaryCtxRetain(&p0, 0));
    CHECK(cuCtxSetCurrent(p0));
    take_driver_resources(&a_held);
    Held in_own;
    take_resources(0, &in_own, /*own_context=*/true);
    advance(1);
    await(2);
    CHECK(cuMemAlloc(&a_extra, kBytes));
    advance(3);
    await(4);
  });
  std::thread b([&] {
    await(1);
    CUcontext p1 = nullptr;
    CHECK(cuDevicePrimaryCtxRetain(&p1, 1));
    CHECK(cuCtxSetCurrent(p1));
    take_resources(1, &b_held, /*own_context=*/false);
    advance(2);
    await(3);
    // The last call of the session: the serving thread is left with this
    // thread's context current, not the one that made the created context.
    CHECK(cuMemAlloc(&b_extra, kBytes));
    CHECK(cuCtxSynchronize());
    advance(4);
  });
  a.join();
  b.join();

  // The premise: both devices are retained, and the created context is live.
  const std::string held = read_stats();
  if (field(held, "retains") != 2 || field(held, "contexts") != 1) {
    std::fprintf(stderr,
                 "FAIL: expected both devices retained and one created "
                 "context before exiting: %s\n",
                 held.c_str());
    g_failures++;
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  // Exits holding everything, for expiry to give back.
  std::printf("PASS: two threads on two devices exit holding everything\n");
  return 0;
}

// --- three tenants on one primary context -----------------------------------
//
// A session may release its retain while still holding what it made in the
// primary context: that is legal, and while anybody else holds a retain the
// context lives on and so does everything in it. But when somebody else then
// makes the last release - by calling it, or by expiring - the context is
// destroyed, and only that session's record is corrected. The first session's
// record still lists memory that no longer exists, and if a third session has
// retained and allocated in between, on a real driver those addresses can be
// the third session's. So the first session's expiry must free nothing.
//
//   B: retains, and holds.
//   A: retains, takes resources in the primary context, releases (not the
//      last: B holds one), and waits.
//   B: releases (`tenants release`) or is killed and expires
//      (`tenants expire`) - the last release either way.
//   C (this process): retains, takes resources, writes memory.
//   A: killed, expires. C's counts must not move, nothing may be stale, and
//      C must read its memory back.
//
// Whether a session has expired is read from the server's log, which
// run_smoke.sh names in RGPU_SERVER_LOG: the counters cannot say, because
// what is being checked is that the expiry changes nothing.

// Waits for a byte on `fd`; false if the other end went away first.
bool await_byte(int fd, char want) {
  char c = 0;
  return ::read(fd, &c, 1) == 1 && c == want;
}

int tenant_b(int ready_fd, int cmd_fd) {
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));
  const char ok = g_failures ? 'x' : 'k';
  if (::write(ready_fd, &ok, 1) != 1) return 1;
  if (!await_byte(cmd_fd, 'r')) return 1;
  CHECK(cuDevicePrimaryCtxRelease(0));
  const char done = g_failures ? 'x' : 'k';
  if (::write(ready_fd, &done, 1) != 1) return 1;
  for (;;) ::pause();
}

int tenant_a(int ready_fd) {
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));
  CHECK(cuCtxSetCurrent(primary));
  Held held;
  take_driver_resources(&held);
  // The graph objects belong to no context, so another tenant's last release
  // will not take them with the primary context; destroy them here, while this
  // tenant still has a current context, so they do not leak.
  destroy_held_graphs(held);
  CHECK(cuDevicePrimaryCtxRelease(0));
  const char ok = g_failures ? 'x' : 'k';
  if (::write(ready_fd, &ok, 1) != 1) return 1;
  for (;;) ::pause();
}

// Starts this binary again as `role`, with the given descriptors passed on
// the command line.
pid_t spawn(const char* self, const char* role, std::vector<int> fds) {
  const pid_t parent = ::getpid();
  const pid_t pid = ::fork();
  if (pid != 0) return pid;
  die_with_parent(parent);
  std::vector<std::string> strs;
  for (int fd : fds) strs.push_back(std::to_string(fd));
  std::vector<char*> args = {const_cast<char*>(self), const_cast<char*>(role)};
  for (auto& str : strs) args.push_back(&str[0]);
  args.push_back(nullptr);
  ::execv("/proc/self/exe", args.data());
  ::execv(self, args.data());
  std::perror("execv");
  ::_exit(127);
}

int expired_sessions() {
  const char* path = std::getenv("RGPU_SERVER_LOG");
  if (!path) return -1;
  std::FILE* f = std::fopen(path, "r");
  if (!f) return -1;
  int n = 0;
  char line[1024];
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strstr(line, " expired;")) n++;
  }
  std::fclose(f);
  return n;
}

bool await_expired(int n) {
  const int grace = std::getenv("RGPU_SESSION_GRACE")
                        ? std::atoi(std::getenv("RGPU_SESSION_GRACE"))
                        : 120;
  for (int waited = 0; waited < (grace + 20) * 1000; waited += 100) {
    if (expired_sessions() >= n) return true;
    ::usleep(100 * 1000);
  }
  return false;
}

int tenants(const char* self, bool b_expires) {
  if (!std::getenv("RGPU_FAKE_STATS") || expired_sessions() < 0) {
    std::fprintf(stderr, "FAIL: tenants needs RGPU_FAKE_STATS and "
                         "RGPU_SERVER_LOG; run it from run_smoke.sh\n");
    return 1;
  }
  int ready[2], cmd[2];
  if (::pipe(ready) != 0 || ::pipe(cmd) != 0) return 1;
  std::vector<pid_t> children;
  auto fail = [&](const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    for (pid_t c : children) {
      ::kill(c, SIGKILL);
      ::waitpid(c, nullptr, 0);
    }
    return 1;
  };

  const pid_t b = spawn(self, "tenant-b", {ready[1], cmd[0]});
  children.push_back(b);
  if (!await_byte(ready[0], 'k')) return fail("session B never retained");
  const pid_t a = spawn(self, "tenant-a", {ready[1]});
  children.push_back(a);
  if (!await_byte(ready[0], 'k')) return fail("session A never got going");

  const std::string with_a = read_stats();
  if (field(with_a, "allocs") < 2 || field(with_a, "retains") != 1) {
    return fail("A's resources should be live, and only B's retain held");
  }

  if (b_expires) {
    ::kill(b, SIGKILL);
    ::waitpid(b, nullptr, 0);
    children.erase(children.begin());
    if (!await_expired(1)) return fail("session B never expired");
  } else {
    if (::write(cmd[1], "r", 1) != 1 || !await_byte(ready[0], 'k')) {
      return fail("session B never released");
    }
  }

  // The premise: that was the last release, and the context went with it.
  const std::string gone = read_stats();
  if (field(gone, "allocs") != 0 || field(gone, "retains") != 0 ||
      field(gone, "modules") != 0 || field(gone, "stale") != 0) {
    std::fprintf(stderr, "  found: %s\n", gone.c_str());
    return fail("B's release should have been the last one and destroyed "
                "everything in the primary context");
  }

  // C, which is this process, moves into the recreated context.
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));
  CHECK(cuCtxSetCurrent(primary));
  Held mine;
  take_driver_resources(&mine);
  std::vector<unsigned char> written(kBytes);
  for (size_t i = 0; i < kBytes; i++) written[i] = (unsigned char)(i * 7 + 1);
  CHECK(cuMemcpyHtoD(mine.a, written.data(), kBytes));
  CHECK(cuCtxSynchronize());
  const std::string baseline = read_stats();
  const int expired_before = expired_sessions();

  ::kill(a, SIGKILL);
  ::waitpid(a, nullptr, 0);
  children.pop_back();
  if (!await_expired(expired_before + 1)) {
    return fail("session A never expired");
  }

  const std::string after = read_stats();
  if (after != baseline) {
    std::fprintf(stderr,
                 "FAIL: A's expiry touched what it no longer owned\n"
                 "  expected: %s\n     found: %s\n",
                 baseline.c_str(), after.c_str());
    g_failures++;
  }
  std::vector<unsigned char> read_back(kBytes, 0);
  CHECK(cuMemcpyDtoH(read_back.data(), mine.a, kBytes));
  if (std::memcmp(read_back.data(), written.data(), kBytes) != 0) {
    std::fprintf(stderr, "FAIL: C's memory did not survive A's expiry\n");
    g_failures++;
  }
  CHECK(cuStreamSynchronize(mine.stream));

  for (pid_t c : children) {
    ::kill(c, SIGKILL);
    ::waitpid(c, nullptr, 0);
  }
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: a session whose primary context was destroyed by "
              "another's %s frees nothing at expiry\n",
              b_expires ? "expiry" : "release");
  return 0;
}

// --- an object made from another session's object ---------------------------
//
// A session that captures a graph on a stream it did not make - a stream
// handle from another session, which nothing stops a client sending - cannot
// know which context the graph lives in: the stream's own record is in the
// other session. If it recorded the graph in its current context, its expiry
// would free the graph even after the graph's real context had been destroyed
// and the graph with it, which is a stale free. So it records the graph as of
// unknown placement, and its expiry leaves it alone: at worst a leak.
//
//   This process: retains device 0's primary context, makes a stream.
//   M: retains the same context, captures a graph on this process's stream,
//      and is killed. Its expiry must not free the graph.
//   This process: destroys the graph and the stream itself, and exits.
//
// What is checked is the server's rule, not where a driver puts the graph:
// the graph must survive M's expiry wherever it lives.

int foreign_capturer(int ready_fd, const char* stream_arg) {
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));
  CHECK(cuCtxSetCurrent(primary));
  const auto stream = reinterpret_cast<CUstream>(
      static_cast<uint64_t>(std::strtoull(stream_arg, nullptr, 16)));
  CUgraph graph = nullptr;
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(stream, &graph));
  char msg[32];
  const int n = std::snprintf(msg, sizeof(msg), "%c%016llx", g_failures ? 'x' : 'k',
                              (unsigned long long)reinterpret_cast<uint64_t>(graph));
  if (::write(ready_fd, msg, n) != n) return 1;
  for (;;) ::pause();
}

int foreign(const char* self) {
  if (!std::getenv("RGPU_FAKE_STATS") || expired_sessions() < 0) {
    std::fprintf(stderr, "FAIL: foreign needs RGPU_FAKE_STATS and "
                         "RGPU_SERVER_LOG; run it from run_smoke.sh\n");
    return 1;
  }
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));
  CHECK(cuCtxSetCurrent(primary));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, 0));

  int ready[2];
  if (::pipe(ready) != 0) return 1;
  char stream_hex[32];
  std::snprintf(stream_hex, sizeof(stream_hex), "%llx",
                (unsigned long long)reinterpret_cast<uint64_t>(stream));
  const std::string ready_fd = std::to_string(ready[1]);
  const pid_t parent = ::getpid();
  const pid_t m = ::fork();
  if (m == 0) {
    die_with_parent(parent);
    char* args[] = {const_cast<char*>(self), const_cast<char*>("foreign-m"),
                    const_cast<char*>(ready_fd.c_str()), stream_hex, nullptr};
    ::execv("/proc/self/exe", args);
    ::execv(self, args);
    std::perror("execv");
    ::_exit(127);
  }
  char msg[17] = {0};
  if (!await_byte(ready[0], 'k') || ::read(ready[0], msg, 16) != 16) {
    std::fprintf(stderr, "FAIL: session M never captured a graph\n");
    ::kill(m, SIGKILL);
    ::waitpid(m, nullptr, 0);
    return 1;
  }
  const auto graph = reinterpret_cast<CUgraph>(
      static_cast<uint64_t>(std::strtoull(msg, nullptr, 16)));

  const std::string with_m = read_stats();
  const int expired_before = expired_sessions();
  ::kill(m, SIGKILL);
  ::waitpid(m, nullptr, 0);
  if (!await_expired(expired_before + 1)) {
    std::fprintf(stderr, "FAIL: session M never expired\n");
    return 1;
  }
  const std::string after = read_stats();
  if (field(with_m, "graphs") != 1 || field(after, "graphs") != 1 ||
      field(after, "stale") != 0) {
    std::fprintf(stderr,
                 "FAIL: M's expiry freed a graph it made from a stream it did "
                 "not make\n  before: %s\n   after: %s\n",
                 with_m.c_str(), after.c_str());
    g_failures++;
  }

  // What M's expiry left alone is still this server's to clean up.
  CHECK(cuGraphDestroy(graph));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuDevicePrimaryCtxRelease(0));
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: a graph made from another session's stream survives "
              "that session's expiry\n");
  return 0;
}

// A session that dies in the middle of a stream capture. Everything it made
// has to come back at expiry, and the captures it left open have to be ended:
// an open capture holds its stream in capture, and while one begun other than
// RELAXED is open, the thread that began it - to the driver, the thread that
// served the session, which is the one that releases it - is prohibited from
// potentially unsafe calls, freeing memory among them, unless its own mode is
// RELAXED. The fake driver enforces that much (tests/fake_cuda.cpp).
//
//   C: retains device 0's primary context and allocates; begins a capture on
//      the context's default stream (GLOBAL) and on a stream of its own
//      (THREAD_LOCAL); allocates again in RELAXED mode, the way PyTorch makes
//      an allocation in the middle of a capture; leaves its thread in GLOBAL
//      mode ("strict") or in RELAXED mode ("relaxed"); and is killed.
//   This process: holds a retain on the same context throughout, so C's expiry
//      is not the last release - which would destroy everything in it and
//      hide whatever C's own release left behind - and checks what is left
//      once C has expired.
int capture_holder(int ready_fd, bool relaxed) {
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));
  CHECK(cuCtxSetCurrent(primary));
  CUdeviceptr before = 0, during = 0;
  CHECK(cuMemAlloc(&before, kBytes));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, 0));
  CHECK(cuStreamBeginCapture(nullptr, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));

  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
  CHECK(cuThreadExchangeStreamCaptureMode(&mode));
  CHECK(cuMemAlloc(&during, kBytes));
  if (!relaxed) {
    CHECK(cuThreadExchangeStreamCaptureMode(&mode));
    // The restriction is real, or this case would prove nothing.
    CUdeviceptr refused = 0;
    if (cuMemAlloc(&refused, kBytes) == CUDA_SUCCESS) {
      std::fprintf(stderr, "FAIL: an allocation in GLOBAL mode during this "
                           "thread's own capture was allowed\n");
      g_failures++;
    }
  }
  const char ok = g_failures ? 'x' : 'k';
  if (::write(ready_fd, &ok, 1) != 1) return 1;
  for (;;) ::pause();
}

int capture(const char* self, bool relaxed) {
  if (!std::getenv("RGPU_FAKE_STATS") || expired_sessions() < 0) {
    std::fprintf(stderr, "FAIL: capture needs RGPU_FAKE_STATS and "
                         "RGPU_SERVER_LOG; run it from run_smoke.sh\n");
    return 1;
  }
  CHECK(cuInit(0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, 0));

  int ready[2];
  if (::pipe(ready) != 0) return 1;
  const pid_t c = spawn(self, relaxed ? "capture-relaxed" : "capture-strict",
                        {ready[1]});
  if (!await_byte(ready[0], 'k')) {
    std::fprintf(stderr, "FAIL: session C never got into its captures\n");
    ::kill(c, SIGKILL);
    ::waitpid(c, nullptr, 0);
    return 1;
  }
  const std::string with_c = read_stats();
  if (field(with_c, "captures") != 2 || field(with_c, "allocs") != 2) {
    std::fprintf(stderr, "FAIL: session C should hold two captures and two "
                         "allocations: %s\n", with_c.c_str());
    g_failures++;
  }

  const long swaps_with_c = call_count("modeswaps");
  const int expired_before = expired_sessions();
  ::kill(c, SIGKILL);
  ::waitpid(c, nullptr, 0);
  if (!await_expired(expired_before + 1)) {
    std::fprintf(stderr, "FAIL: session C never expired\n");
    return 1;
  }
  const std::string after = read_stats();
  // The serving thread is put back in the default mode before the release:
  // one exchange if C left it RELAXED, none if C left it in the default.
  const long swaps = call_count("modeswaps") - swaps_with_c;
  if (swaps != (relaxed ? 1 : 0)) {
    std::fprintf(stderr, "FAIL: C's expiry made %ld capture-mode exchange(s); "
                         "expected %d, putting the serving thread back in the "
                         "default mode only if C left it elsewhere\n",
                 swaps, relaxed ? 1 : 0);
    g_failures++;
  }
  for (const char* kind : {"allocs", "streams", "graphs", "execs", "captures",
                           "overreleases", "stale"}) {
    if (field(after, kind) != 0) {
      std::fprintf(stderr,
                   "FAIL: session C expired in the middle of its captures and "
                   "left %s behind\n  with C: %s\n   after: %s\n",
                   kind, with_c.c_str(), after.c_str());
      g_failures++;
    }
  }
  if (field(after, "retains") != 1) {
    std::fprintf(stderr, "FAIL: only this session's retain should be left: "
                         "%s\n", after.c_str());
    g_failures++;
  }

  CHECK(cuDevicePrimaryCtxRelease(0));
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS: a session that expires in the middle of a capture (%s) "
              "gives everything back\n", relaxed ? "relaxed" : "strict");
  return 0;
}

// The client that dies. Takes rather more than the parent, says so down the
// pipe, and then waits to be killed: no exit handler, no frees, nothing the
// server can read as a goodbye. That is what a crash looks like from here.
int hold_and_wait(int ready_fd) {
  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));

  // Two retains and one release: the session owns one, and the server has to
  // release exactly that many at the end.
  //
  // The release comes after resources are made in the primary context, and it
  // is not the last release in the process - the parent holds a retain too -
  // so it destroys nothing, and the server must go on owing every one of
  // them. A server that forgot a primary context's contents on any release,
  // rather than only the last, would leave them behind at expiry, and the
  // counts would never come back to the parent's.
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CHECK(cuCtxSetCurrent(primary));
  Held in_primary;
  take_driver_resources(&in_primary);
  CHECK(cuDevicePrimaryCtxRelease(dev));

  Held held;
  take_resources(dev, &held, /*own_context=*/true);
  CUdeviceptr extra = 0;
  CHECK(cuMemAlloc(&extra, kBytes));

  // Everything above returns a value, so it has already reached the server;
  // this is belt and braces for anything that might be batched.
  CHECK(cuCtxSynchronize());

  const char msg = g_failures ? 'x' : 'k';
  if (::write(ready_fd, &msg, 1) != 1) return 1;
  for (;;) ::pause();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2 && std::strcmp(argv[1], "hold") == 0) {
    return hold_and_wait(std::atoi(argv[2]));
  }
  if (argc > 1 && std::strcmp(argv[1], "reset") == 0) return reset_alone();
  if (argc > 1 && std::strcmp(argv[1], "release") == 0) return release_alone();
  if (argc > 1 && std::strcmp(argv[1], "detach") == 0) return detach_alone();
  if (argc > 1 && std::strcmp(argv[1], "derived") == 0) return derived_alone();
  if (argc > 1 && std::strcmp(argv[1], "threads") == 0) return threads_alone();
  if (argc > 3 && std::strcmp(argv[1], "tenant-b") == 0) {
    return tenant_b(std::atoi(argv[2]), std::atoi(argv[3]));
  }
  if (argc > 2 && std::strcmp(argv[1], "tenant-a") == 0) {
    return tenant_a(std::atoi(argv[2]));
  }
  if (argc > 3 && std::strcmp(argv[1], "foreign-m") == 0) {
    return foreign_capturer(std::atoi(argv[2]), argv[3]);
  }
  if (argc > 1 && std::strcmp(argv[1], "foreign") == 0) return foreign(argv[0]);
  if (argc > 2 && std::strcmp(argv[1], "capture-strict") == 0) {
    return capture_holder(std::atoi(argv[2]), false);
  }
  if (argc > 2 && std::strcmp(argv[1], "capture-relaxed") == 0) {
    return capture_holder(std::atoi(argv[2]), true);
  }
  if (argc > 2 && std::strcmp(argv[1], "capture") == 0) {
    return capture(argv[0], std::strcmp(argv[2], "relaxed") == 0);
  }
  if (argc > 2 && std::strcmp(argv[1], "tenants") == 0) {
    return tenants(argv[0], std::strcmp(argv[2], "expire") == 0);
  }

  if (!std::getenv("RGPU_FAKE_STATS")) {
    std::fprintf(stderr,
                 "FAIL: RGPU_FAKE_STATS is not set, so there is no way to see "
                 "what the server is holding; run this from run_smoke.sh\n");
    return 1;
  }

  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CHECK(cuCtxSetCurrent(primary));

  Held mine;
  take_resources(dev, &mine, /*own_context=*/false);
  std::vector<unsigned char> written(kBytes);
  for (size_t i = 0; i < kBytes; i++) written[i] = (unsigned char)(i * 11 + 3);
  CHECK(cuMemcpyHtoD(mine.a, written.data(), kBytes));
  CHECK(cuCtxSynchronize());
  if (g_failures) {
    std::printf("\nFAILED: %d check(s) before the test even started\n",
                g_failures);
    return 1;
  }

  // What this process holds, and the only thing that should be left at the
  // end.
  const std::string baseline = read_stats();
  if (baseline.empty()) {
    std::fprintf(stderr, "FAIL: the server published no counters\n");
    return 1;
  }
  std::printf("this session holds: %s\n", baseline.c_str());

  int ready[2] = {-1, -1};
  if (::pipe(ready) != 0) {
    std::perror("pipe");
    return 1;
  }
  const pid_t parent = ::getpid();
  const pid_t child = ::fork();
  if (child < 0) {
    std::perror("fork");
    return 1;
  }
  if (child == 0) {
    die_with_parent(parent);
    ::close(ready[0]);
    char fd[16];
    std::snprintf(fd, sizeof(fd), "%d", ready[1]);
    char* args[] = {argv[0], const_cast<char*>("hold"), fd, nullptr};
    ::execv("/proc/self/exe", args);
    ::execv(argv[0], args);
    std::perror("execv");
    ::_exit(127);
  }
  ::close(ready[1]);

  char ack = 0;
  if (::read(ready[0], &ack, 1) != 1 || ack != 'k') {
    std::fprintf(stderr, "FAIL: the second session never got going\n");
    ::kill(child, SIGKILL);
    ::waitpid(child, nullptr, 0);
    return 1;
  }

  // A device reset would destroy what the other session is holding, and this
  // one cannot be trusted with that decision: while anybody else is connected
  // it has to be refused. Checked here, with the child live and its resources
  // counted, so a reset that got through would show up twice over - once as
  // the wrong return code, and once as the counts never coming back.
  const CUresult refused = cuDevicePrimaryCtxReset(dev);
  if (refused != CUDA_ERROR_NOT_SUPPORTED) {
    std::fprintf(stderr,
                 "FAIL: resetting the device with another session live "
                 "returned %d, expected %d\n",
                 refused, CUDA_ERROR_NOT_SUPPORTED);
    g_failures++;
  }

  const std::string both = read_stats();
  std::printf("with the second session: %s\n", both.c_str());
  // The premise: the child's resources really are outstanding on the server.
  // Without this the rest would pass against a server that had never heard of
  // the child at all.
  if (field(both, "allocs") < field(baseline, "allocs") + 3 ||
      field(both, "retains") < field(baseline, "retains") + 1 ||
      field(both, "contexts") < field(baseline, "contexts") + 1) {
    std::fprintf(stderr,
                 "FAIL: the second session's resources never showed up; this "
                 "test proves nothing\n");
    ::kill(child, SIGKILL);
    ::waitpid(child, nullptr, 0);
    return 1;
  }

  // Killed, not asked to stop. Nothing on the client side runs after this.
  ::kill(child, SIGKILL);
  ::waitpid(child, nullptr, 0);

  // The grace period, then the cleanup. Polled rather than slept through, so a
  // slow machine does not turn into a false failure, and a fast one does not
  // make this test slow.
  const int grace = std::getenv("RGPU_SESSION_GRACE")
                        ? std::atoi(std::getenv("RGPU_SESSION_GRACE"))
                        : 120;
  const int deadline_ms = (grace + 20) * 1000;
  std::string now = both;
  for (int waited = 0; waited < deadline_ms; waited += 100) {
    now = read_stats();
    if (now == baseline) break;
    ::usleep(100 * 1000);
  }
  if (now != baseline) {
    std::fprintf(stderr,
                 "FAIL: %ds after the client died its resources are still "
                 "held\n  expected: %s\n     found: %s\n",
                 deadline_ms / 1000, baseline.c_str(), now.c_str());
    g_failures++;
  } else {
    std::printf("after the dead session expired: %s\n", now.c_str());
  }

  // And this session, which never went anywhere, still has everything it had.
  // A cleanup that released one retain too many would have taken the primary
  // context out from under it.
  std::vector<unsigned char> read_back(kBytes, 0);
  CHECK(cuMemcpyDtoH(read_back.data(), mine.a, kBytes));
  if (std::memcmp(read_back.data(), written.data(), kBytes) != 0) {
    std::fprintf(stderr,
                 "FAIL: this session's memory did not survive the other "
                 "session's cleanup\n");
    g_failures++;
  }
  CUdeviceptr after = 0;
  CHECK(cuMemAlloc(&after, kBytes));
  CHECK(cuMemsetD8(after, 0x5a, kBytes));
  CHECK(cuStreamSynchronize(mine.stream));
  CHECK(cuMemFree(after));

  // With the other session gone this one is alone, so a device reset that was
  // refused earlier must now be allowed. Last, because it destroys this
  // session's own memory. It pins the other half of the session count: a
  // server that never counted a session out would refuse resets for good after
  // the first client came and went. The count drops when the dead session's
  // thread finishes, a moment after its resources come back, so this waits
  // for that rather than racing it.
  CUresult reset = CUDA_ERROR_NOT_SUPPORTED;
  for (int waited = 0; waited < 10000 && reset == CUDA_ERROR_NOT_SUPPORTED;
       waited += 100) {
    reset = cuDevicePrimaryCtxReset(dev);
    if (reset == CUDA_ERROR_NOT_SUPPORTED) ::usleep(100 * 1000);
  }
  if (reset != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "FAIL: alone on the server after the other session expired, "
                 "a device reset still returned %d\n",
                 reset);
    g_failures++;
  }

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf(
      "\nPASS: a dead session's resources came back and a live one's did "
      "not\n");
  return 0;
}
