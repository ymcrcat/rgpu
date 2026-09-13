// The client half of tests/hw/e2e_check.sh: CUDA driver-API programs that run
// through the rgpu client shim (build/libcuda.so.1) against a real
// rgpu-server on a GPU host.
//
// The existing smoke tests (tests/reconnect_smoke.cpp, expiry_smoke.cpp,
// threadctx_smoke.cpp) do much the same against the fake driver, but they read
// the fake's counters (RGPU_FAKE_STATS) and depend on its two fake devices, so
// they cannot say anything on hardware. Everything here is checked through
// what a real driver reports: memory, pointer attributes, kernel results.
//
//   hold DIR          keep a session: retain the primary context, hold 64 MiB
//                     with a known pattern, and answer commands written to
//                     DIR/cmd (free, verify, reset, after-reset, exit) in
//                     DIR/reply
//   hog MIB           allocate MIB MiB and _exit without freeing anything
//   reset             retain the primary context, ask for a device reset and
//                     print RESET_RC <code>
//   threads           two client threads, each with its own context on the
//                     one GPU
//   drop IMAGE K      K accumulating kernel launches (a += b) across a broken
//                     connection; the result says whether any ran twice or
//                     not at all
//   drop-memset K     the same with memsets, when there is no kernel image:
//                     catches lost work, cannot catch duplicated work
//   lost DIR          hold a session, wait for DIR/go, then check that every
//                     call fails promptly (the server was restarted meanwhile)
//
// Every mode first checks that it really is talking through the shim, since a
// GPU host also has the real libcuda.so.1 on the loader's path.

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

namespace {

int g_failures = 0;
std::mutex g_mu;

void line(const char* verdict, const char* fmt, ...) {
  char msg[2048];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lk(g_mu);
  std::printf("%s %s\n", verdict, msg);
  std::fflush(stdout);
  if (std::strcmp(verdict, "FAIL") == 0) g_failures++;
}

#define PASS(...) line("PASS", __VA_ARGS__)
#define FAIL(...) line("FAIL", __VA_ARGS__)
#define NOTE(...) line("    ", __VA_ARGS__)

#define CHECK(call)                                                        \
  do {                                                                     \
    CUresult r_ = (call);                                                  \
    if (r_ != CUDA_SUCCESS) {                                              \
      FAIL("%s:%d: %s -> %d", __FILE__, __LINE__, #call, static_cast<int>(r_)); \
    }                                                                      \
  } while (0)

// Stops the mode: nothing after a failed setup step would mean anything.
#define NEED(call)                                                         \
  do {                                                                     \
    CUresult r_ = (call);                                                  \
    if (r_ != CUDA_SUCCESS) {                                              \
      FAIL("setup %s:%d: %s -> %d", __FILE__, __LINE__, #call,             \
           static_cast<int>(r_));                                          \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

void ensure_shim() {
  // rgpu::log, exported by the shim and by nothing NVIDIA ships.
  if (::dlsym(RTLD_DEFAULT, "_ZN4rgpu3logEPKcz")) return;
  Dl_info info{};
  ::dladdr(reinterpret_cast<void*>(&cuInit), &info);
  FAIL("cuInit comes from %s, not the rgpu shim; set LD_LIBRARY_PATH to the "
       "build directory",
       info.dli_fname ? info.dli_fname : "?");
  std::exit(3);
}

bool exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

void write_file(const std::string& path, const std::string& text) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp);
    f << text << "\n";
  }
  std::rename(tmp.c_str(), path.c_str());
}

std::string read_word(const std::string& path) {
  std::ifstream f(path);
  std::string w;
  f >> w;
  return w;
}

void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::vector<unsigned char> pattern(size_t n, unsigned seed) {
  std::vector<unsigned char> v(n);
  for (size_t i = 0; i < n; i++) {
    v[i] = static_cast<unsigned char>((i * 131 + seed) >> 3);
  }
  return v;
}

// A small allocation in the current context that has to round-trip and name
// `ctx` as its owner.
bool fresh_allocation_works(CUcontext ctx, std::string* why) {
  CUdeviceptr p = 0;
  CUresult r = cuMemAlloc(&p, 1u << 20);
  if (r != CUDA_SUCCESS) {
    *why = "cuMemAlloc -> " + std::to_string(r);
    return false;
  }
  const std::vector<unsigned char> out = pattern(4096, 99);
  std::vector<unsigned char> back(4096, 0);
  CUresult w = cuMemcpyHtoD(p, out.data(), out.size());
  CUresult rd = cuMemcpyDtoH(back.data(), p, back.size());
  CUcontext owner = nullptr;
  CUresult a = cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, p);
  cuMemFree(p);
  if (w != CUDA_SUCCESS || rd != CUDA_SUCCESS || back != out) {
    *why = "round trip HtoD " + std::to_string(w) + " DtoH " +
           std::to_string(rd) + (back == out ? "" : " bytes differ");
    return false;
  }
  if (a != CUDA_SUCCESS || owner != ctx) {
    *why = "owner attribute -> " + std::to_string(a) + " names another context";
    return false;
  }
  return true;
}

// --- hold --------------------------------------------------------------------

int mode_hold(const std::string& dir) {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  CUcontext primary = nullptr;
  NEED(cuDevicePrimaryCtxRetain(&primary, dev));
  NEED(cuCtxSetCurrent(primary));
  const size_t n = 64u << 20;
  const std::vector<unsigned char> written = pattern(n, 7);
  CUdeviceptr held = 0;
  NEED(cuMemAlloc(&held, n));
  NEED(cuMemcpyHtoD(held, written.data(), n));
  NEED(cuCtxSynchronize());
  write_file(dir + "/ready", "ready");
  bool reset_done = false;

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::minutes(30);
  while (std::chrono::steady_clock::now() < deadline) {
    const std::string cmd_path = dir + "/cmd";
    if (!exists(cmd_path)) {
      sleep_ms(50);
      continue;
    }
    const std::string cmd = read_word(cmd_path);
    std::remove(cmd_path.c_str());
    std::ostringstream reply;
    if (cmd == "free") {
      size_t free = 0, total = 0;
      const CUresult r = cuMemGetInfo(&free, &total);
      reply << "rc=" << r << " free=" << free << " total=" << total;
    } else if (cmd == "verify") {
      std::vector<unsigned char> back(n, 0);
      const CUresult r = cuMemcpyDtoH(back.data(), held, n);
      std::string why;
      if (r != CUDA_SUCCESS) {
        reply << "bad: reading the held memory back -> " << r;
      } else if (back != written) {
        reply << "bad: the held memory no longer holds what was written";
      } else if (!fresh_allocation_works(primary, &why)) {
        reply << "bad: " << why;
      } else {
        reply << "ok";
      }
    } else if (cmd == "reset") {
      const CUresult r = cuDevicePrimaryCtxReset(dev);
      if (r == CUDA_SUCCESS) reset_done = true;
      reply << "rc=" << r;
    } else if (cmd == "after-reset") {
      CUcontext cur = nullptr;
      const CUresult g = cuCtxGetCurrent(&cur);
      std::string why;
      if (g != CUDA_SUCCESS || cur != primary) {
        reply << "bad: cuCtxGetCurrent -> " << g
              << (cur == primary ? "" : ", not the primary context");
      } else {
        // A reset leaves the primary context's handle valid but *inactive*
        // (driver_probe check 7 measured this on hardware: an allocation under
        // it returns CUDA_ERROR_CONTEXT_IS_DESTROYED until it is retained
        // again). So a driver-API client recovers the way cudart does on its
        // next runtime call - by retaining the primary context once more - and
        // that is what this checks: the same handle comes back and the session
        // can allocate in it again. The extra retain is balanced by a release
        // so the exit accounting is unchanged.
        CUcontext again = nullptr;
        const CUresult re = cuDevicePrimaryCtxRetain(&again, dev);
        if (re != CUDA_SUCCESS || again != primary) {
          reply << "bad: re-retain after reset -> " << re
                << (again == primary ? "" : ", a different handle");
        } else {
          cuCtxSetCurrent(primary);
          if (!fresh_allocation_works(primary, &why)) {
            reply << "bad: " << why;
          } else {
            reply << "ok";
          }
          cuDevicePrimaryCtxRelease(dev);
        }
      }
    } else if (cmd == "exit") {
      CUresult f = CUDA_SUCCESS;
      if (!reset_done) f = cuMemFree(held);
      cuCtxSetCurrent(nullptr);
      const CUresult r = cuDevicePrimaryCtxRelease(dev);
      reply << "free=" << f << " release=" << r;
      write_file(dir + "/reply", reply.str());
      return 0;
    } else {
      reply << "bad: unknown command " << cmd;
    }
    write_file(dir + "/reply", reply.str());
  }
  std::fprintf(stderr, "hold: no exit command within 30 minutes\n");
  return 1;
}

// --- hog ---------------------------------------------------------------------

int mode_hog(size_t mib) {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  CUcontext primary = nullptr;
  NEED(cuDevicePrimaryCtxRetain(&primary, dev));
  NEED(cuCtxSetCurrent(primary));
  CUdeviceptr p = 0;
  NEED(cuMemAlloc(&p, mib << 20));
  NEED(cuMemsetD8(p, 0xAB, mib << 20));
  NEED(cuCtxSynchronize());
  std::printf("ALLOCATED %zu MiB\n", mib);
  std::fflush(stdout);
  // No free, no release, no destructors: the connection just goes away.
  std::_Exit(0);
}

// --- reset -------------------------------------------------------------------

int mode_reset() {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  CUcontext primary = nullptr;
  NEED(cuDevicePrimaryCtxRetain(&primary, dev));
  NEED(cuCtxSetCurrent(primary));
  const CUresult r = cuDevicePrimaryCtxReset(dev);
  std::printf("RESET_RC %d\n", static_cast<int>(r));
  std::fflush(stdout);
  cuCtxSetCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}

// --- threads -----------------------------------------------------------------

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

CUcontext owner_of(CUdeviceptr p, CUresult* r) {
  CUcontext c = nullptr;
  *r = cuPointerGetAttribute(&c, CU_POINTER_ATTRIBUTE_CONTEXT, p);
  return c;
}

// Two client threads on one GPU. On one device, DEVICE_ORDINAL cannot tell
// contexts apart, so placement is read with CU_POINTER_ATTRIBUTE_CONTEXT;
// tests/hw/driver_probe check 2 shows the real driver answers that with the
// allocating context. Each step runs only after the other thread has moved
// its own context, which is the order issue #2 went wrong in.
int mode_threads() {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  Turns turns;
  CUcontext c1 = nullptr, c2 = nullptr, p = nullptr;
  const size_t n = 1u << 20;

  std::thread t1([&] {
    NEED(cuCtxCreate(&c1, 0, dev));
    CUdeviceptr m1 = 0;
    NEED(cuMemAlloc(&m1, n));
    const std::vector<unsigned char> w = pattern(n, 0x11);
    NEED(cuMemcpyHtoD(m1, w.data(), n));
    turns.advance(1);
    turns.await(2);  // thread 2 has created and made current its own context

    CUcontext cur = nullptr;
    CHECK(cuCtxGetCurrent(&cur));
    if (cur == c1) {
      PASS("threads: thread 1 still sees its own context after thread 2 "
           "created one");
    } else {
      FAIL("threads: thread 1's cuCtxGetCurrent names %p after thread 2 "
           "created its context (own %p, thread 2's %p)",
           (void*)cur, (void*)c1, (void*)c2);
    }
    CUdeviceptr m1b = 0;
    CHECK(cuMemAlloc(&m1b, n));
    CUresult r = CUDA_SUCCESS;
    const CUcontext owner = owner_of(m1b, &r);
    if (r == CUDA_SUCCESS && owner == c1) {
      PASS("threads: thread 1's allocation after thread 2's switch belongs to "
           "thread 1's context");
    } else {
      FAIL("threads: thread 1's allocation belongs to %p (-> %d); own %p, "
           "thread 2's %p",
           (void*)owner, static_cast<int>(r), (void*)c1, (void*)c2);
    }
    std::vector<unsigned char> back(n, 0);
    CHECK(cuMemcpyDtoH(back.data(), m1, n));
    if (back != w) FAIL("threads: thread 1's memory did not read back");
    turns.advance(3);
    turns.await(4);  // thread 2 has switched to the primary context

    cur = nullptr;
    CHECK(cuCtxGetCurrent(&cur));
    CUdeviceptr m1c = 0;
    CHECK(cuMemAlloc(&m1c, n));
    const CUcontext owner_c = owner_of(m1c, &r);
    if (cur == c1 && r == CUDA_SUCCESS && owner_c == c1) {
      PASS("threads: after thread 2 moved to the primary context, thread 1 "
           "still sees and allocates in its own");
    } else {
      FAIL("threads: after thread 2's move to the primary context, thread 1 "
           "sees %p and allocates in %p (own %p, primary %p)",
           (void*)cur, (void*)owner_c, (void*)c1, (void*)p);
    }
    CHECK(cuMemFree(m1c));
    CHECK(cuMemFree(m1b));
    CHECK(cuMemFree(m1));
    CHECK(cuCtxDestroy(c1));
    turns.advance(5);
  });

  std::thread t2([&] {
    turns.await(1);
    NEED(cuCtxCreate(&c2, 0, dev));
    CUdeviceptr m2 = 0;
    NEED(cuMemAlloc(&m2, n));
    const std::vector<unsigned char> w = pattern(n, 0x22);
    NEED(cuMemcpyHtoD(m2, w.data(), n));
    turns.advance(2);
    turns.await(3);  // thread 1 has run under its own context

    CUcontext cur = nullptr;
    CHECK(cuCtxGetCurrent(&cur));
    CUresult r = CUDA_SUCCESS;
    const CUcontext owner = owner_of(m2, &r);
    if (cur == c2 && r == CUDA_SUCCESS && owner == c2) {
      PASS("threads: thread 2 still sees its own context after thread 1's "
           "calls, and its allocation belongs to it");
    } else {
      FAIL("threads: thread 2 sees %p, its allocation belongs to %p (-> %d); "
           "own %p, thread 1's %p",
           (void*)cur, (void*)owner, static_cast<int>(r), (void*)c2, (void*)c1);
    }
    NEED(cuDevicePrimaryCtxRetain(&p, dev));
    NEED(cuCtxSetCurrent(p));
    CUdeviceptr mp = 0;
    CHECK(cuMemAlloc(&mp, n));
    const CUcontext owner_p = owner_of(mp, &r);
    if (r == CUDA_SUCCESS && owner_p == p) {
      PASS("threads: thread 2's allocation after its move belongs to the "
           "primary context");
    } else {
      FAIL("threads: thread 2's allocation after its move belongs to %p "
           "(primary %p)",
           (void*)owner_p, (void*)p);
    }
    turns.advance(4);
    turns.await(5);  // thread 1 has checked, freed and destroyed its context

    cur = nullptr;
    CHECK(cuCtxGetCurrent(&cur));
    if (cur == p) {
      PASS("threads: thread 2 still sees the primary context after thread 1 "
           "destroyed its own");
    } else {
      FAIL("threads: thread 2 sees %p after thread 1's destroy (primary %p)",
           (void*)cur, (void*)p);
    }
    std::vector<unsigned char> back(n, 0);
    CHECK(cuMemcpyDtoH(back.data(), m2, n));
    if (back != w) FAIL("threads: thread 2's first memory did not read back");
    CHECK(cuMemFree(mp));
    CHECK(cuMemFree(m2));
    CHECK(cuCtxSetCurrent(c2));
    CHECK(cuCtxDestroy(c2));
    CHECK(cuDevicePrimaryCtxRelease(dev));
  });
  t1.join();
  t2.join();
  if (c1 == c2 || c1 == p || c2 == p) {
    FAIL("threads: two of the three contexts share a handle (%p %p %p)",
         (void*)c1, (void*)c2, (void*)p);
  }
  return g_failures ? 1 : 0;
}

// --- drop --------------------------------------------------------------------

std::vector<char> read_image(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> img((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  img.push_back(0);
  return img;
}

// a starts at 0 and every launch adds b, so after K launches a == K * b
// exactly (small integers are exact in float). A launch that ran twice makes
// it bigger, one that was lost makes it smaller. Launches go without replies,
// so the batch the server breaks in the middle of is a batch of them, with a
// synchronising call every 50 so the break can also fall between batches.
int mode_drop(const std::string& image_path, int k) {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  NEED(cuCtxCreate(&ctx, 0, dev));
  std::vector<char> image = read_image(image_path);
  if (image.size() < 2) {
    FAIL("cannot read %s", image_path.c_str());
    return 1;
  }
  CUmodule mod = nullptr;
  NEED(cuModuleLoadData(&mod, image.data()));
  CUfunction fn = nullptr;
  NEED(cuModuleGetFunction(&fn, mod, "vecadd"));
  int n = 1 << 16;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUdeviceptr a = 0, b = 0;
  NEED(cuMemAlloc(&a, bytes));
  NEED(cuMemAlloc(&b, bytes));
  std::vector<float> ha(n, 0.0f), hb(n);
  for (int i = 0; i < n; i++) hb[i] = static_cast<float>(i % 7 + 1);
  NEED(cuMemcpyHtoD(a, ha.data(), bytes));
  NEED(cuMemcpyHtoD(b, hb.data(), bytes));
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  void* args[] = {&a, &b, &a, &n};
  for (int i = 1; i <= k; i++) {
    CHECK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args,
                         nullptr));
    if (i % 50 == 0) CHECK(cuCtxSynchronize());
  }
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(ha.data(), a, bytes));
  int wrong = 0;
  for (int i = 0; i < n; i++) wrong += ha[i] != static_cast<float>(k) * hb[i];
  // Element 6 has b == 7, so a / 7 counts the launches that ran there.
  std::printf("RESULT launches=%d ran=%g wrong_elements=%d\n", k,
              static_cast<double>(ha[6] / hb[6]), wrong);
  if (wrong == 0) {
    PASS("drop: all %d accumulating launches ran exactly once", k);
  } else {
    FAIL("drop: %d of %d elements wrong; element 6 says %g launches ran, not "
         "%d (more: a launch ran twice; fewer: one was lost)",
         wrong, n, static_cast<double>(ha[6] / hb[6]), k);
  }
  CHECK(cuMemFree(a));
  CHECK(cuMemFree(b));
  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));
  return g_failures ? 1 : 0;
}

int mode_drop_memset(int k) {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  NEED(cuCtxCreate(&ctx, 0, dev));
  const size_t n = 4096;
  CUdeviceptr m = 0;
  NEED(cuMemAlloc(&m, n * static_cast<size_t>(k)));
  // Memset i writes value i into its own slot, so every memset that was lost
  // leaves a zero behind. A duplicated one writes the same bytes again and
  // cannot be seen: this mode proves nothing about at-most-once.
  NEED(cuMemsetD8(m, 0, n * static_cast<size_t>(k)));
  for (int i = 0; i < k; i++) {
    CHECK(cuMemsetD8Async(m + n * i, static_cast<unsigned char>(i % 255 + 1),
                          n, nullptr));
    if ((i + 1) % 50 == 0) CHECK(cuCtxSynchronize());
  }
  CHECK(cuCtxSynchronize());
  std::vector<unsigned char> back(n * static_cast<size_t>(k), 0);
  CHECK(cuMemcpyDtoH(back.data(), m, back.size()));
  int lost = 0;
  for (int i = 0; i < k; i++) {
    lost += back[n * i] != static_cast<unsigned char>(i % 255 + 1) ||
            back[n * (i + 1) - 1] != static_cast<unsigned char>(i % 255 + 1);
  }
  if (lost == 0) {
    PASS("drop-memset: all %d memsets landed (duplicates are not detectable "
         "this way)",
         k);
  } else {
    FAIL("drop-memset: %d of %d memsets were lost", lost, k);
  }
  CHECK(cuMemFree(m));
  CHECK(cuCtxDestroy(ctx));
  return g_failures ? 1 : 0;
}

// --- lost --------------------------------------------------------------------

// The session is lost while this client is idle: the script kills the server
// and starts a new one on the same port, then writes DIR/go. Work queued
// before the kill, without replies, must not be replayed into the new server
// (the script reads its log), and every call from here on must fail, quickly.
int mode_lost(const std::string& dir) {
  NEED(cuInit(0));
  CUdevice dev = 0;
  NEED(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  NEED(cuCtxCreate(&ctx, 0, dev));
  const size_t n = 1u << 20;
  CUdeviceptr d = 0;
  NEED(cuMemAlloc(&d, n));
  NEED(cuCtxSynchronize());
  // Without replies, so they may still be unacknowledged when the server dies.
  for (int i = 0; i < 8; i++) CHECK(cuMemsetD8Async(d, 0x5a, n, nullptr));
  write_file(dir + "/ready", "ready");
  for (int waited = 0; !exists(dir + "/go"); waited += 50) {
    if (waited > 300000) {
      FAIL("lost: never told to go on");
      return 1;
    }
    sleep_ms(50);
  }
  const auto start = std::chrono::steady_clock::now();
  size_t total = 0;
  int count = 0;
  const CUresult r1 = cuDeviceTotalMem(&total, dev);
  const CUresult r2 = cuCtxSynchronize();
  const CUresult r3 = cuDeviceGetCount(&count);
  const CUresult r4 = cuMemFree(d);
  const double took = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  sleep_ms(1000);
  const CUresult r5 = cuDeviceGetCount(&count);
  std::printf("LOST_RCS total=%d sync=%d count=%d free=%d later_count=%d "
              "seconds=%.1f\n",
              r1, r2, r3, r4, r5, took);
  if (r1 != CUDA_SUCCESS && r2 != CUDA_SUCCESS && r3 != CUDA_SUCCESS &&
      r4 != CUDA_SUCCESS && r5 != CUDA_SUCCESS) {
    PASS("lost: every call after the server restart failed (%d %d %d %d, and "
         "%d a second later)",
         r1, r2, r3, r4, r5);
  } else {
    FAIL("lost: a call succeeded in a session that is gone (%d %d %d %d %d)",
         r1, r2, r3, r4, r5);
  }
  if (took < 15.0) {
    PASS("lost: the four calls failed within %.1fs", took);
  } else {
    FAIL("lost: failing four calls took %.1fs; the client kept trying to "
         "reach a session the server said was gone",
         took);
  }
  return g_failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  auto arg = [&](int i) -> std::string {
    if (argc <= i) {
      std::fprintf(stderr, "%s needs more arguments\n", mode.c_str());
      std::exit(64);
    }
    return argv[i];
  };
  if (mode.empty()) {
    std::fprintf(stderr,
                 "usage: %s hold DIR | hog MIB | reset | threads | "
                 "drop IMAGE K | drop-memset K | lost DIR\n",
                 argv[0]);
    return 64;
  }
  ensure_shim();
  if (mode == "hold") return mode_hold(arg(2));
  if (mode == "hog") return mode_hog(std::strtoull(arg(2).c_str(), nullptr, 10));
  if (mode == "reset") return mode_reset();
  if (mode == "threads") return mode_threads();
  if (mode == "drop") return mode_drop(arg(2), std::atoi(arg(3).c_str()));
  if (mode == "drop-memset") return mode_drop_memset(std::atoi(arg(2).c_str()));
  if (mode == "lost") return mode_lost(arg(2));
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 64;
}
