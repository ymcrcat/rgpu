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
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  return s;
}

long field(const std::string& stats, const char* name) {
  const std::string key = std::string(name) + "=";
  size_t at = stats.find(key);
  if (at == std::string::npos) return -1;
  return std::strtol(stats.c_str() + at + key.size(), nullptr, 10);
}

// Everything a client can take and then die holding. Called in both processes:
// the child's copy is what has to come back, this process's copy is what must
// not.
struct Held {
  CUdeviceptr a = 0, b = 0;
  CUstream stream = nullptr;
  CUevent event = nullptr;
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
  CUgraph graph = nullptr;
  CUgraphExec exec = nullptr;
  CHECK(cuStreamBeginCapture(held->stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuStreamEndCapture(held->stream, &graph));
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CUgraph clone = nullptr;
  CHECK(cuGraphClone(&clone, graph));
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
// out is taken here: a driver whose reset left the streams, events, modules and
// graphs behind would show them as held at the end, and a server that went on
// to release them would show as stale. Library handles are not taken: they are
// the libraries' objects, not the driver's, and the fake libraries do not
// model a reset.
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
  const pid_t child = ::fork();
  if (child < 0) {
    std::perror("fork");
    return 1;
  }
  if (child == 0) {
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
