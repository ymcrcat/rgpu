// Real-hardware probe of the CUDA driver behaviours that rgpu's fixes for
// issues #2 (per-thread contexts) and #4 (expired sessions release their GPU
// resources) rely on.
//
// Those fixes were built and tested against tests/fake_cuda.cpp, and several
// of the behaviours the fake models were read off the CUDA header, or inferred
// from it, rather than observed. This program links the REAL libcuda.so.1 - no
// rgpu at all - and looks.
//
//   g++ -std=c++17 -I third_party/cuda_include tests/hw/driver_probe.cpp
//       -o build/hw/driver_probe -L<dir with libcuda.so.1> -l:libcuda.so.1
//       -ldl -lpthread
//   build/hw/driver_probe [--image build/vecadd.fatbin] [--only 1,7] [--timeout 120]
//
// (scripts/hw_check.sh does all of this on the pod.)
//
// Output: one line per behaviour, starting with
//   PASS      the driver does what rgpu (the server, the client or the fake
//             driver its tests run on) assumes;
//   FAIL      it does not - the line says what that means for the code;
//   OBSERVED  a value we only needed to know;
//   SKIP      the check could not run (missing API, no kernel image, ...);
//   INFO      context.
// Each FAIL/PASS is tagged with what depends on it:
//   [server]  server/ or client/ code relies on the answer;
//   [fake]    only tests/fake_cuda.cpp models it, so a FAIL means the fake is
//             not faithful and tests built on it may assert false contracts.
//
// Exit status: 1 if any FAIL, else 2 if any SKIP, else 0.
//
// Every check runs in its own child process (this binary re-executed with
// --check ID). Several checks deliberately use handles whose context has been
// destroyed, and a driver is free to crash or leave a sticky error on that; a
// fresh process per check keeps one check's damage out of the next one's
// answer, and lets a crash itself be reported as an observation. The parent
// never initialises CUDA, so nothing is forked after cuInit.
//
// Where the assumptions live is cited as file:line at commit 400ce99 (branch
// reviewer-issues); the files are being edited, so search for the quoted
// function or comment if a line has moved.

#include <dlfcn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

// The graph and stream-capture calls (checks 6 and 12) are in the 12.8 driver
// header this repo builds against. A header older than that lacks them, and
// those checks are compiled out and reported as SKIP rather than breaking the
// build. Even when compiled in, they are resolved with dlsym at run time,
// because the pod's driver may be older than the header.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#define PROBE_HAVE_GRAPH_API 1
#else
#define PROBE_HAVE_GRAPH_API 0
#endif

namespace {

// --- reporting ---------------------------------------------------------------

std::string g_id = "?";
int g_fail = 0;
int g_skip = 0;

void say(const char* verdict, const char* fmt, ...) {
  char msg[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  std::printf("%-8s %-12s %s\n", verdict, g_id.c_str(), msg);
  std::fflush(stdout);
  if (std::strcmp(verdict, "FAIL") == 0) g_fail++;
  if (std::strcmp(verdict, "SKIP") == 0) g_skip++;
}

#define PASS(...) say("PASS", __VA_ARGS__)
#define FAIL(...) say("FAIL", __VA_ARGS__)
#define OBSERVED(...) say("OBSERVED", __VA_ARGS__)
#define SKIP(...) say("SKIP", __VA_ARGS__)
#define INFO(...) say("INFO", __VA_ARGS__)

// A machine-readable result for the parent, which combines several children's
// answers into check 6's verdict.
void key(const std::string& k, const std::string& v) {
  std::printf("KEY %s %s\n", k.c_str(), v.c_str());
  std::fflush(stdout);
}

void heading(const char* title, const char* assumption) {
  std::printf("\n== %s: %s\n   assumption: %s\n", g_id.c_str(), title,
              assumption);
  std::fflush(stdout);
}

std::string rc(CUresult r) {
  const char* name = nullptr;
  if (cuGetErrorName(r, &name) != CUDA_SUCCESS || !name) name = "?";
  char buf[160];
  std::snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(r), name);
  return buf;
}

const char* yn(bool b) { return b ? "yes" : "no"; }

// A setup step that has to work for the check to mean anything.
void need(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  FAIL("setup failed, so this check observed nothing: %s -> %s", what,
       rc(r).c_str());
  std::_Exit(1);
}

void need_true(bool ok, const char* what) {
  if (ok) return;
  FAIL("setup failed, so this check observed nothing: %s", what);
  std::_Exit(1);
}

// --- driver helpers ----------------------------------------------------------

CUdevice g_dev = 0;
std::string g_image_path;
std::vector<char> g_image;
std::string g_image_origin;
long g_iterations = 5000000;

void init() {
  need(cuInit(0), "cuInit(0)");
  need(cuDeviceGet(&g_dev, 0), "cuDeviceGet(0)");
}

CUcontext current() {
  CUcontext c = nullptr;
  cuCtxGetCurrent(&c);
  return c;
}

// A created context, left on nobody's stack: cuCtxCreate pushes it, and the
// pop takes it off again, so every check chooses its currency explicitly.
CUcontext make_ctx() {
  CUcontext c = nullptr;
  need(cuCtxCreate(&c, 0, g_dev), "cuCtxCreate");
  CUcontext popped = nullptr;
  need(cuCtxPopCurrent(&popped), "cuCtxPopCurrent after cuCtxCreate");
  need_true(popped == c, "cuCtxCreate did not leave its context on top");
  return c;
}

void use(CUcontext c, const char* what) { need(cuCtxSetCurrent(c), what); }

// Kernel image: a fatbin or PTX from --image, or PTX compiled on the spot with
// NVRTC if a libnvrtc can be found (RGPU_NVRTC, then the usual sonames).
const char* kKernelSource = R"(
extern "C" __global__ void vecadd(const float* a, const float* b, float* c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
)";

bool image_from_nvrtc(std::string* why) {
  std::vector<const char*> names;
  if (const char* e = std::getenv("RGPU_NVRTC")) names.push_back(e);
  for (const char* n : {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12",
                        "libnvrtc.so.11.2"}) {
    names.push_back(n);
  }
  void* lib = nullptr;
  const char* found = nullptr;
  for (const char* n : names) {
    lib = ::dlopen(n, RTLD_NOW | RTLD_LOCAL);
    if (lib) {
      found = n;
      break;
    }
  }
  if (!lib) {
    *why = "no --image given and no libnvrtc found (set RGPU_NVRTC)";
    return false;
  }
  using Prog = void*;
  auto create = reinterpret_cast<int (*)(Prog*, const char*, const char*, int,
                                         const char* const*,
                                         const char* const*)>(
      ::dlsym(lib, "nvrtcCreateProgram"));
  auto compile = reinterpret_cast<int (*)(Prog, int, const char* const*)>(
      ::dlsym(lib, "nvrtcCompileProgram"));
  auto ptx_size =
      reinterpret_cast<int (*)(Prog, size_t*)>(::dlsym(lib, "nvrtcGetPTXSize"));
  auto get_ptx =
      reinterpret_cast<int (*)(Prog, char*)>(::dlsym(lib, "nvrtcGetPTX"));
  auto log_size = reinterpret_cast<int (*)(Prog, size_t*)>(
      ::dlsym(lib, "nvrtcGetProgramLogSize"));
  auto get_log = reinterpret_cast<int (*)(Prog, char*)>(
      ::dlsym(lib, "nvrtcGetProgramLog"));
  auto destroy =
      reinterpret_cast<int (*)(Prog*)>(::dlsym(lib, "nvrtcDestroyProgram"));
  if (!create || !compile || !ptx_size || !get_ptx || !destroy) {
    *why = std::string(found) + " lacks the nvrtc entry points needed";
    return false;
  }
  Prog prog = nullptr;
  if (create(&prog, kKernelSource, "vecadd.cu", 0, nullptr, nullptr) != 0) {
    *why = "nvrtcCreateProgram failed";
    return false;
  }
  if (compile(prog, 0, nullptr) != 0) {
    std::string log;
    size_t n = 0;
    if (log_size && get_log && log_size(prog, &n) == 0 && n > 1) {
      log.resize(n);
      get_log(prog, &log[0]);
    }
    *why = "nvrtcCompileProgram failed: " + log;
    destroy(&prog);
    return false;
  }
  size_t n = 0;
  ptx_size(prog, &n);
  g_image.assign(n + 1, 0);
  get_ptx(prog, g_image.data());
  destroy(&prog);
  g_image_origin = std::string("PTX from ") + found;
  return true;
}

bool load_image(std::string* why) {
  if (!g_image.empty()) return true;
  if (!g_image_path.empty()) {
    std::FILE* f = std::fopen(g_image_path.c_str(), "rb");
    if (!f) {
      *why = "cannot open " + g_image_path;
      return false;
    }
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    g_image.assign(static_cast<size_t>(len > 0 ? len : 0) + 1, 0);
    size_t got = std::fread(g_image.data(), 1, static_cast<size_t>(len), f);
    std::fclose(f);
    if (len <= 0 || got != static_cast<size_t>(len)) {
      g_image.clear();
      *why = "short read on " + g_image_path;
      return false;
    }
    g_image_origin = g_image_path;
    return true;
  }
  return image_from_nvrtc(why);
}

// A module and its vecadd function in the current context.
void load_kernel(CUmodule* m, CUfunction* f) {
  std::string why;
  need_true(load_image(&why), why.c_str());
  need(cuModuleLoadData(m, g_image.data()), "cuModuleLoadData");
  need(cuModuleGetFunction(f, *m, "vecadd"), "cuModuleGetFunction(vecadd)");
}

// Three float vectors in the current context: a = i, b = 2i, c = -1. A launch
// of vecadd that ran makes c = 3i everywhere; one that did not leaves -1.
struct Vec {
  CUdeviceptr a = 0, b = 0, c = 0;
  int n = 4096;
};

void vec_reset_c(Vec& v) {
  std::vector<float> c(v.n, -1.0f);
  need(cuMemcpyHtoD(v.c, c.data(), v.n * sizeof(float)), "HtoD c");
}

void vec_alloc(Vec& v) {
  const size_t bytes = v.n * sizeof(float);
  need(cuMemAlloc(&v.a, bytes), "cuMemAlloc a");
  need(cuMemAlloc(&v.b, bytes), "cuMemAlloc b");
  need(cuMemAlloc(&v.c, bytes), "cuMemAlloc c");
  std::vector<float> a(v.n), b(v.n);
  for (int i = 0; i < v.n; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
  need(cuMemcpyHtoD(v.a, a.data(), bytes), "HtoD a");
  need(cuMemcpyHtoD(v.b, b.data(), bytes), "HtoD b");
  vec_reset_c(v);
}

CUresult vec_launch(CUfunction f, Vec& v, CUstream s) {
  const int threads = 256;
  const int blocks = (v.n + threads - 1) / threads;
  void* args[] = {&v.a, &v.b, &v.c, &v.n};
  return cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, s, args, nullptr);
}

// 1 if the kernel ran over all of c, 0 if it ran over none of it, -1 if mixed
// or c could not be read.
int vec_ran(Vec& v) {
  std::vector<float> c(v.n, 0.0f);
  if (cuMemcpyDtoH(c.data(), v.c, v.n * sizeof(float)) != CUDA_SUCCESS) {
    return -1;
  }
  int ran = 0, not_ran = 0;
  for (int i = 0; i < v.n; i++) {
    if (c[i] == static_cast<float>(3 * i)) ran++;
    if (c[i] == -1.0f) not_ran++;
  }
  // i == 0 is 0 either way only if it ran; -1 otherwise, so no overlap.
  if (ran == v.n) return 1;
  if (not_ran == v.n) return 0;
  return -1;
}

const char* ran_text(int r) {
  return r == 1 ? "ran" : r == 0 ? "did not run" : "partly ran or unreadable";
}

// =============================================================================
// Check 0: what we are talking to.
// =============================================================================
void check_env() {
  heading("environment", "the probe talks to the real driver, on one GPU");
  Dl_info info{};
  const bool found =
      ::dladdr(reinterpret_cast<void*>(&cuInit), &info) && info.dli_fname;
  // rgpu::log is exported by the rgpu client shim and by nothing NVIDIA ships.
  const bool shim = ::dlsym(RTLD_DEFAULT, "_ZN4rgpu3logEPKcz") != nullptr;
  if (shim) {
    FAIL("[probe] cuInit comes from the rgpu client shim (%s), not the "
         "driver; nothing below would be about hardware. Unset "
         "LD_LIBRARY_PATH / LD_PRELOAD",
         found ? info.dli_fname : "?");
    std::_Exit(1);
  }
  INFO("cuInit resolved from %s", found ? info.dli_fname : "(unknown)");
  init();
  int version = 0, count = 0;
  cuDriverGetVersion(&version);
  cuDeviceGetCount(&count);
  char name[256] = {0};
  cuDeviceGetName(name, sizeof(name), g_dev);
  size_t total = 0;
  cuDeviceTotalMem(&total, g_dev);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       g_dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       g_dev);
  INFO("driver API version %d, %d device(s); device 0 is %s, sm_%d%d, %zu MiB",
       version, count, name, major, minor, total >> 20);
  INFO("probe built against cuda.h CUDA_VERSION %d", CUDA_VERSION);
  if (count != 1) {
    INFO("expected exactly one GPU; the checks all use device 0 and nothing "
         "below depends on there being no second one");
  }
  const char* optional[] = {"cuStreamBeginCapture_v2",
                            "cuStreamEndCapture",
                            "cuGraphClone",
                            "cuGraphInstantiateWithFlags",
                            "cuGraphGetNodes",
                            "cuGraphExecGetFlags",
                            "cuThreadExchangeStreamCaptureMode",
                            "cuFuncGetParamInfo"};
  std::string missing;
  for (const char* s : optional) {
    if (!::dlsym(RTLD_DEFAULT, s)) missing += std::string(" ") + s;
  }
  INFO("driver entry points the later checks need and this driver lacks:%s",
       missing.empty() ? " none" : missing.c_str());
  std::string why;
  if (load_image(&why)) {
    INFO("kernel image for checks 4, 5 and 6: %s (%zu bytes)",
         g_image_origin.c_str(), g_image.size() - 1);
  } else {
    INFO("no kernel image: %s. Checks 4 and 5 will SKIP", why.c_str());
  }
}

// =============================================================================
// Checks 1, 2, 3: another context's device pointer, used with B current.
// =============================================================================
void check_cross_context_memory() {
  heading("a device pointer from context A, used while context B is current",
          "copies and frees infer the owning context from the pointer "
          "(unified addressing); CU_POINTER_ATTRIBUTE_CONTEXT names the "
          "allocating context");
  init();
  CUcontext A = make_ctx();
  CUcontext B = make_ctx();
  const size_t n = 64u << 20;
  use(A, "cuCtxSetCurrent(A)");
  CUdeviceptr pA = 0;
  need(cuMemAlloc(&pA, n), "cuMemAlloc under A");
  use(B, "cuCtxSetCurrent(B)");
  CUdeviceptr pB = 0;
  need(cuMemAlloc(&pB, 4096), "cuMemAlloc under B");

  // --- 1 ---------------------------------------------------------------------
  // Verifies: the fake's rule that a copy of another context's pointer
  // succeeds, used there instead of refusing (tests/fake_cuda.cpp:176-186,
  // "What a real driver does with an object that belongs to another context:
  // it uses it there"), and the slice A contract "Do not rely on a
  // cross-context memcpy ... failing" (issue-2-sliceA-report.md, "Contract for
  // slices B and C (current)"; spec "What changed from the first draft",
  // "Test 1 asserts placement, not a refused memcpy").
  // PASS: both copies succeed and the bytes round-trip, with B still current.
  // FAIL: hardware refuses; the fake is more lenient than the driver, and a
  //       wrong-context bug the fake lets through would fail on a GPU. The
  //       placement-based tests stay valid either way.
  g_id = "1";
  std::vector<unsigned char> out(n), back(n, 0);
  for (size_t i = 0; i < n; i++) out[i] = static_cast<unsigned char>(i * 31 + 7);
  const CUresult h2d = cuMemcpyHtoD(pA, out.data(), n);
  const CUresult d2h = cuMemcpyDtoH(back.data(), pA, n);
  const bool same = h2d == CUDA_SUCCESS && d2h == CUDA_SUCCESS &&
                    std::memcmp(out.data(), back.data(), n) == 0;
  const bool still_b = current() == B;
  if (same && still_b) {
    PASS("[fake] HtoD and DtoH of A's pointer with B current both succeeded "
         "and the bytes round-tripped; B is still current");
  } else {
    FAIL("[fake] HtoD -> %s, DtoH -> %s, bytes %s, B still current: %s. The "
         "fake lets cross-context copies through; hardware does not",
         rc(h2d).c_str(), rc(d2h).c_str(), same ? "match" : "differ",
         yn(still_b));
  }
  {
    const CUresult sync_b = cuCtxSynchronize();
    use(A, "cuCtxSetCurrent(A)");
    const CUresult sync_a = cuCtxSynchronize();
    use(B, "cuCtxSetCurrent(B)");
    OBSERVED("synchronize afterwards (a sticky error would show here): "
             "B -> %s, A -> %s",
             rc(sync_b).c_str(), rc(sync_a).c_str());
  }

  // --- 2 ---------------------------------------------------------------------
  // Verifies: CU_POINTER_ATTRIBUTE_CONTEXT names the context that allocated
  // the pointer, not the current one, and DEVICE_ORDINAL names its device.
  // Slice A contract, "Prove the bug through placement" (read CONTEXT or
  // DEVICE_ORDINAL); the fake answers from the allocation record
  // (tests/fake_cuda.cpp cuPointerGetAttribute, ~line 830); tests/hw/e2e_client
  // `threads` proves per-thread placement on one GPU with CONTEXT, because
  // DEVICE_ORDINAL cannot tell two contexts on one device apart.
  // The control (B's pointer read with A current) is what makes this
  // distinguish "the allocating context" from "whatever is current".
  // PASS: A's pointer reads A under B, and B's pointer reads B under A.
  // FAIL [server]: the attribute cannot prove placement, and every
  //       placement-based test (threadctx_smoke, reconnect_smoke threads, the
  //       e2e threads check) needs another oracle on hardware.
  g_id = "2";
  CUcontext owner_a = nullptr, owner_b = nullptr;
  const CUresult ra =
      cuPointerGetAttribute(&owner_a, CU_POINTER_ATTRIBUTE_CONTEXT, pA);
  use(A, "cuCtxSetCurrent(A)");
  const CUresult rb =
      cuPointerGetAttribute(&owner_b, CU_POINTER_ATTRIBUTE_CONTEXT, pB);
  use(B, "cuCtxSetCurrent(B)");
  if (ra == CUDA_SUCCESS && rb == CUDA_SUCCESS && owner_a == A &&
      owner_b == B) {
    PASS("[server] CONTEXT names the allocating context, not the current one "
         "(A's pointer under B -> A, B's pointer under A -> B)");
  } else {
    FAIL("[server] CONTEXT of A's pointer under B -> %s %p (A=%p B=%p); of "
         "B's pointer under A -> %s %p. Placement cannot be proved with this "
         "attribute",
         rc(ra).c_str(), (void*)owner_a, (void*)A, (void*)B, rc(rb).c_str(),
         (void*)owner_b);
  }
  int ordinal = -1;
  const CUresult ro =
      cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pA);
  if (ro == CUDA_SUCCESS && ordinal == 0) {
    PASS("[server] DEVICE_ORDINAL of A's pointer read under B is 0");
  } else {
    FAIL("[server] DEVICE_ORDINAL of A's pointer -> %s, ordinal %d",
         rc(ro).c_str(), ordinal);
  }

  // --- 3 ---------------------------------------------------------------------
  // Verifies: a free of another context's pointer succeeds (the fake's same
  // "uses it there" rule, tests/fake_cuda.cpp:176-186) and really frees it.
  // The server does not depend on this: release_inventory
  // (server/inventory.cpp ~line 915) makes each recorded context current
  // before freeing what was made in it.
  // PASS: cuMemFree succeeds, the pointer stops answering, and the free
  //       memory grows by about 64 MiB.
  // FAIL [fake]: hardware refuses (the fake is more lenient), or claims
  //       success without freeing.
  g_id = "3";
  size_t free0 = 0, free1 = 0, total = 0;
  cuMemGetInfo(&free0, &total);
  const CUresult fr = cuMemFree(pA);
  CUcontext gone = nullptr;
  const CUresult after =
      cuPointerGetAttribute(&gone, CU_POINTER_ATTRIBUTE_CONTEXT, pA);
  cuMemGetInfo(&free1, &total);
  const long long delta_mib =
      (static_cast<long long>(free1) - static_cast<long long>(free0)) >> 20;
  if (fr == CUDA_SUCCESS && after != CUDA_SUCCESS) {
    PASS("[fake] cuMemFree of A's pointer with B current succeeded; the "
         "pointer now answers %s; free memory changed by %+lld MiB (64 "
         "allocated)",
         rc(after).c_str(), delta_mib);
  } else if (fr == CUDA_SUCCESS) {
    FAIL("[fake] cuMemFree returned success but the pointer still answers "
         "(context %p); free memory changed by %+lld MiB",
         (void*)gone, delta_mib);
  } else {
    FAIL("[fake] cuMemFree of A's pointer with B current -> %s; the fake "
         "accepts it",
         rc(fr).c_str());
    use(A, "cuCtxSetCurrent(A)");
    OBSERVED("the same free with A current -> %s", rc(cuMemFree(pA)).c_str());
    use(B, "cuCtxSetCurrent(B)");
  }
  cuMemFree(pB);
  cuCtxSetCurrent(nullptr);
  cuCtxDestroy(A);
  cuCtxDestroy(B);
}

// =============================================================================
// Check 4: cuModuleUnload of A's module with B current.
// =============================================================================
// Verifies: the fake refuses it with CUDA_ERROR_INVALID_VALUE
// (tests/fake_cuda.cpp:188-191 and cuModuleUnload ~line 982, current_only;
// asserted by tests/test_fake_cuda.cpp:340,359; slice A contract "Codes that
// apply"). Taken from the header's "Unloads a module hmod from the current
// context" and its return list, not observed.
// The server does not depend on it: release_inventory makes the module's
// context current before unloading.
// PASS: CUDA_ERROR_INVALID_VALUE, and the module is still usable and unloads
//       with A current (so the refusal was about the context).
// FAIL [fake] with SUCCESS: hardware unloads another context's module; the
//       fake is stricter than the driver and test_fake_cuda asserts a false
//       contract.
// FAIL [fake] with another code: change the fake's code to that one.
void check_module_unload() {
  heading("cuModuleUnload of a module loaded under A, with B current",
          "refused with CUDA_ERROR_INVALID_VALUE (header text, not observed)");
  std::string why;
  init();
  if (!load_image(&why)) {
    SKIP("no kernel image: %s", why.c_str());
    return;
  }
  CUcontext A = make_ctx();
  CUcontext B = make_ctx();
  use(B, "cuCtxSetCurrent(B)");
  // Control: unloading works at all under B, for B's own module.
  CUmodule mb = nullptr;
  CUfunction fb = nullptr;
  load_kernel(&mb, &fb);
  const CUresult own = cuModuleUnload(mb);
  use(A, "cuCtxSetCurrent(A)");
  CUmodule ma = nullptr;
  CUfunction fa = nullptr;
  load_kernel(&ma, &fa);
  use(B, "cuCtxSetCurrent(B)");
  const CUresult cross = cuModuleUnload(ma);
  OBSERVED("control: B's own module unloaded with B current -> %s",
           rc(own).c_str());
  if (cross == CUDA_SUCCESS) {
    // Not touched again: the handle may now point at freed memory.
    FAIL("[fake] unloading A's module with B current succeeded; the fake "
         "refuses it with CUDA_ERROR_INVALID_VALUE, so it and "
         "test_fake_cuda.cpp assert a contract hardware does not have");
  } else {
    use(A, "cuCtxSetCurrent(A)");
    CUfunction again = nullptr;
    const CUresult alive = cuModuleGetFunction(&again, ma, "vecadd");
    const CUresult home = cuModuleUnload(ma);
    if (cross == CUDA_ERROR_INVALID_VALUE && alive == CUDA_SUCCESS &&
        home == CUDA_SUCCESS) {
      PASS("[fake] refused with %s; the module was untouched and unloaded "
           "with A current",
           rc(cross).c_str());
    } else {
      FAIL("[fake] refused with %s (fake: 400 INVALID_VALUE); afterwards, "
           "with A current, cuModuleGetFunction -> %s and cuModuleUnload -> %s",
           rc(cross).c_str(), rc(alive).c_str(), rc(home).c_str());
    }
  }
  cuCtxSetCurrent(nullptr);
  cuCtxDestroy(A);
  cuCtxDestroy(B);
}

// =============================================================================
// Check 5: launching A's function with B current.
// =============================================================================
// 5a, null stream. Verifies the fake's INFERRED refusal
// (tests/fake_cuda.cpp:192-201 and cuLaunchKernel ~line 1034, "fctx != sctx"
// with a null stream meaning the current context; slice A contract "With a
// null stream it also applies unless the function's context is current. That
// case is inferred"). Nothing in server/ or client/ depends on it; the
// contract forbids building a proof on it.
// PASS: CUDA_ERROR_INVALID_HANDLE and the kernel did not run.
// FAIL [fake]: any other outcome. SUCCESS-and-ran means the null stream
//       means the function's context's default stream (or the launch runs
//       wherever the function lives), and the fake is stricter than hardware.
//
// 5b, a stream from B. Verifies the fake's DOCUMENTED refusal ("The CUDA
// context associated with this stream must match that associated with
// function f", cuLaunchKernelEx), same fake code.
// PASS: CUDA_ERROR_INVALID_HANDLE and the kernel did not run.
// FAIL [fake]: any other outcome.
void check_launch(bool with_stream) {
  heading(with_stream
              ? "cuLaunchKernel of A's function on a stream from B, B current"
              : "cuLaunchKernel of A's function on a null stream, B current",
          with_stream
              ? "refused with CUDA_ERROR_INVALID_HANDLE (documented)"
              : "refused with CUDA_ERROR_INVALID_HANDLE (inferred)");
  std::string why;
  init();
  if (!load_image(&why)) {
    SKIP("no kernel image: %s", why.c_str());
    return;
  }
  CUcontext A = make_ctx();
  CUcontext B = make_ctx();
  use(A, "cuCtxSetCurrent(A)");
  CUmodule m = nullptr;
  CUfunction f = nullptr;
  load_kernel(&m, &f);
  Vec v;
  vec_alloc(v);
  // Control: the kernel works in its own context, so "did not run" below is
  // about the launch and not about the kernel.
  need(vec_launch(f, v, nullptr), "control launch with A current");
  need(cuCtxSynchronize(), "control synchronize");
  need_true(vec_ran(v) == 1, "control launch with A current did not compute");
  vec_reset_c(v);

  use(B, "cuCtxSetCurrent(B)");
  CUstream sb = nullptr;
  if (with_stream) need(cuStreamCreate(&sb, 0), "cuStreamCreate under B");
  const CUresult launched = vec_launch(f, v, sb);
  const CUresult sync_b = cuCtxSynchronize();
  use(A, "cuCtxSetCurrent(A)");
  const CUresult sync_a = cuCtxSynchronize();
  const int ran = vec_ran(v);
  OBSERVED("launch -> %s; then synchronize B -> %s, A -> %s; the kernel %s",
           rc(launched).c_str(), rc(sync_b).c_str(), rc(sync_a).c_str(),
           ran_text(ran));
  if (launched == CUDA_ERROR_INVALID_HANDLE && ran == 0) {
    PASS("[fake] refused with CUDA_ERROR_INVALID_HANDLE and nothing ran, as "
         "the fake does");
  } else {
    FAIL("[fake] launch -> %s and the kernel %s; the fake refuses with "
         "CUDA_ERROR_INVALID_HANDLE (tests/fake_cuda.cpp cuLaunchKernel)",
         rc(launched).c_str(), ran_text(ran));
  }
  if (with_stream) {
    // The documented rule is about the stream and the function, whatever is
    // current; with A current it should refuse the same way.
    vec_reset_c(v);
    const CUresult home = vec_launch(f, v, sb);
    const CUresult s_a = cuCtxSynchronize();
    use(B, "cuCtxSetCurrent(B)");
    const CUresult s_b = cuCtxSynchronize();
    use(A, "cuCtxSetCurrent(A)");
    OBSERVED("the same launch on B's stream with A current -> %s "
             "(synchronize A -> %s, B -> %s); the kernel %s",
             rc(home).c_str(), rc(s_a).c_str(), rc(s_b).c_str(),
             ran_text(vec_ran(v)));
  }
  cuCtxSetCurrent(nullptr);
  cuCtxDestroy(A);
  cuCtxDestroy(B);
}

// =============================================================================
// Check 6: which context owns a graph ended, cloned or instantiated from A's
// work while B is current.
// =============================================================================
// Verifies: server/inventory.cpp stamp_of (~line 146-183, "live in their
// source's context ... That they live in the source's context is inferred,
// not documented"), used by w_cuStreamEndCapture, w_cuGraphClone and
// w_cuGraphInstantiateWithFlags; the fake mints them in the source's context
// (tests/fake_cuda.cpp:203-208, cuStreamEndCapture / cuGraphClone /
// cuGraphInstantiateWithFlags ~line 1155-1240). No header text says either
// way; cuGraphCreate takes no context at all, so graphs may belong to none.
//
// Ownership is observed by destroying one context and asking whether each
// object still answers a non-destructive query (cuGraphGetNodes for graphs,
// cuGraphExecGetFlags for executables). Each (context destroyed, object) pair
// is its own child process, a control run destroys nothing, and a second
// control asks the same query of a handle that was destroyed on purpose, to
// learn whether "answers" can be trusted at all. The parent combines them:
// PASS: survives B's destruction and dies with A's - it is A's, as the
//       server records it.
// FAIL [server]: dies with B's - it is the current context's. The server
//       records it under A, so B's destruction is not seen for it, and expiry
//       frees a dead handle (a stale free).
// FAIL [fake]: survives both - it belongs to no context. The server forgets
//       it when A is destroyed and leaks it (the safe direction); the fake,
//       which destroys it with A, is not faithful.
#if PROBE_HAVE_GRAPH_API
struct GraphApi {
  decltype(&cuStreamBeginCapture) begin = nullptr;
  decltype(&cuStreamEndCapture) end = nullptr;
  decltype(&cuGraphClone) clone = nullptr;
  decltype(&cuGraphInstantiate) instantiate = nullptr;
  decltype(&cuGraphGetNodes) get_nodes = nullptr;
  decltype(&cuGraphExecGetFlags) exec_flags = nullptr;
  decltype(&cuGraphLaunch) launch = nullptr;
  decltype(&cuGraphDestroy) destroy = nullptr;
  decltype(&cuGraphExecDestroy) exec_destroy = nullptr;
};

bool resolve(GraphApi* api, std::string* missing) {
#define RESOLVE(field, fn, sym)                                          \
  api->field = reinterpret_cast<decltype(&fn)>(::dlsym(RTLD_DEFAULT, sym)); \
  if (!api->field) *missing += " " sym;
  RESOLVE(begin, cuStreamBeginCapture, "cuStreamBeginCapture_v2");
  RESOLVE(end, cuStreamEndCapture, "cuStreamEndCapture");
  RESOLVE(clone, cuGraphClone, "cuGraphClone");
  RESOLVE(instantiate, cuGraphInstantiate, "cuGraphInstantiateWithFlags");
  RESOLVE(get_nodes, cuGraphGetNodes, "cuGraphGetNodes");
  RESOLVE(exec_flags, cuGraphExecGetFlags, "cuGraphExecGetFlags");
  RESOLVE(launch, cuGraphLaunch, "cuGraphLaunch");
  RESOLVE(destroy, cuGraphDestroy, "cuGraphDestroy");
  RESOLVE(exec_destroy, cuGraphExecDestroy, "cuGraphExecDestroy");
#undef RESOLVE
  return missing->empty();
}

struct GraphSetup {
  CUcontext A = nullptr, B = nullptr;
  CUstream sa = nullptr;
  CUdeviceptr pa = 0;
  size_t n = 1u << 20;
  CUgraph graph = nullptr, clone = nullptr;
  CUgraphExec exec = nullptr;
  CUresult end_rc = CUDA_ERROR_UNKNOWN, clone_rc = CUDA_ERROR_UNKNOWN,
           inst_rc = CUDA_ERROR_UNKNOWN;
  bool ended_under_b = false;
};

// A's stream captures a memset (and a kernel launch, when there is a kernel)
// into a graph; the capture is ended, cloned and instantiated with B current.
// RELAXED, so that making B current in the middle of a capture is not one of
// the "potentially unsafe" calls that would invalidate it.
void graph_setup(const GraphApi& api, GraphSetup* s, bool verbose) {
  init();
  s->A = make_ctx();
  s->B = make_ctx();
  use(s->A, "cuCtxSetCurrent(A)");
  need(cuStreamCreate(&s->sa, 0), "cuStreamCreate under A");
  need(cuMemAlloc(&s->pa, s->n), "cuMemAlloc under A");
  need(cuMemsetD8(s->pa, 0, s->n), "cuMemsetD8");
  std::string why;
  CUmodule m = nullptr;
  CUfunction f = nullptr;
  Vec v;
  const bool kernel = load_image(&why);
  if (kernel) {
    load_kernel(&m, &f);
    vec_alloc(v);
  }
  auto capture = [&](bool end_under_b) {
    use(s->A, "cuCtxSetCurrent(A)");
    need(api.begin(s->sa, CU_STREAM_CAPTURE_MODE_RELAXED),
         "cuStreamBeginCapture(RELAXED) on A's stream");
    need(cuMemsetD8Async(s->pa, 7, s->n, s->sa), "captured cuMemsetD8Async");
    if (kernel) need(vec_launch(f, v, s->sa), "captured cuLaunchKernel");
    if (end_under_b) use(s->B, "cuCtxSetCurrent(B) during the capture");
    CUgraph g = nullptr;
    s->end_rc = api.end(s->sa, &g);
    return g;
  };
  s->graph = capture(true);
  s->ended_under_b = s->graph != nullptr;
  if (!s->graph && s->end_rc != CUDA_SUCCESS) {
    // The capture may still be open; close it where it was begun, so that
    // a second one can start.
    use(s->A, "cuCtxSetCurrent(A)");
    CUgraph leftover = nullptr;
    if (api.end(s->sa, &leftover) == CUDA_SUCCESS && leftover) {
      api.destroy(leftover);
    }
  }
  if (!s->graph) {
    if (verbose) {
      OBSERVED("cuStreamEndCapture with B current -> %s and a null graph; "
               "ending the capture with A current instead, so the graph "
               "itself cannot answer the EndCapture question",
               rc(s->end_rc).c_str());
    }
    s->graph = capture(false);
  }
  need_true(s->graph != nullptr, "no capture produced a graph");
  use(s->B, "cuCtxSetCurrent(B)");
  s->clone_rc = api.clone(&s->clone, s->graph);
  s->inst_rc = api.instantiate(&s->exec, s->graph, 0);
  if (verbose) {
    OBSERVED("with B current: cuStreamEndCapture -> %s (graph made while B "
             "current: %s), cuGraphClone -> %s, cuGraphInstantiate -> %s; "
             "captured %s",
             rc(s->end_rc).c_str(), yn(s->ended_under_b),
             rc(s->clone_rc).c_str(), rc(s->inst_rc).c_str(),
             kernel ? "a memset and a kernel launch" : "a memset only");
  }
}

// Whether a handle still names a live object, asked without changing it. A
// driver that has the entry point but reports it unsupported is asked a
// fallback question instead - a clone of a graph, which is destroyed again,
// or the destroy of an executable, which is the last use of it anyway.
std::string graph_state(const GraphApi& api, CUgraph g) {
  size_t nodes = 0;
  const CUresult r = api.get_nodes(g, nullptr, &nodes);
  if (r == CUDA_SUCCESS) return "alive nodes=" + std::to_string(nodes);
  if (r != CUDA_ERROR_NOT_SUPPORTED) {
    return "dead " + std::to_string(static_cast<int>(r));
  }
  CUgraph tmp = nullptr;
  const CUresult c = api.clone(&tmp, g);
  if (c == CUDA_SUCCESS) {
    api.destroy(tmp);
    return "alive (cuGraphGetNodes unsupported; asked by cuGraphClone)";
  }
  return "dead " + std::to_string(static_cast<int>(c)) +
         " (cuGraphGetNodes unsupported; asked by cuGraphClone)";
}

std::string exec_state(const GraphApi& api, CUgraphExec e) {
  cuuint64_t flags = 0;
  const CUresult r = api.exec_flags(e, &flags);
  if (r == CUDA_SUCCESS) return "alive";
  if (r != CUDA_ERROR_NOT_SUPPORTED) {
    return "dead " + std::to_string(static_cast<int>(r));
  }
  const CUresult d = api.exec_destroy(e);
  if (d == CUDA_SUCCESS) {
    return "alive (cuGraphExecGetFlags unsupported; asked by cuGraphExecDestroy)";
  }
  return "dead " + std::to_string(static_cast<int>(d)) +
         " (cuGraphExecGetFlags unsupported; asked by cuGraphExecDestroy)";
}

// which: "none", "A", "B" or "dead"; object: "graph", "clone", "exec" or
// "all".
void check_graph(const std::string& which, const std::string& object) {
  if (which == "none") {
    heading("ownership of graphs derived from A's capture while B is current",
            "they live in their source's context (inferred, no header text)");
  }
  GraphApi api;
  std::string missing;
  init();
  if (!resolve(&api, &missing)) {
    SKIP("this driver lacks:%s", missing.c_str());
    key("6.skip", "missing");
    return;
  }
  const std::string k = "6." + which + "." + object;

  if (which == "dead") {
    // How a handle that is certainly dead answers. If it answers "alive", a
    // live answer below proves nothing.
    GraphSetup s;
    graph_setup(api, &s, false);
    if (object == "graph") {
      const bool use_clone = s.clone_rc == CUDA_SUCCESS;
      need(api.destroy(use_clone ? s.clone : s.graph), "cuGraphDestroy");
      const std::string st = graph_state(api, use_clone ? s.clone : s.graph);
      OBSERVED("a destroyed graph's handle answers: %s", st.c_str());
      key(k, st);
    } else {
      if (s.inst_rc != CUDA_SUCCESS) {
        key(k, "unavailable");
        return;
      }
      need(api.exec_destroy(s.exec), "cuGraphExecDestroy");
      const std::string st = exec_state(api, s.exec);
      OBSERVED("a destroyed executable's handle answers: %s", st.c_str());
      key(k, st);
    }
    return;
  }

  GraphSetup s;
  graph_setup(api, &s, which == "none");
  key("6.ended_under_b", s.ended_under_b ? "1" : "0");
  if (which == "none") {
    // The executable really does the captured work, and with nothing
    // destroyed every object answers as alive - otherwise the query cannot
    // tell anything apart.
    if (s.inst_rc == CUDA_SUCCESS) {
      use(s.A, "cuCtxSetCurrent(A)");
      const CUresult launched = api.launch(s.exec, s.sa);
      const CUresult synced = cuStreamSynchronize(s.sa);
      unsigned char byte = 0;
      cuMemcpyDtoH(&byte, s.pa, 1);
      OBSERVED("launching the executable on A's stream -> %s, synchronize -> "
               "%s; the captured memset %s",
               rc(launched).c_str(), rc(synced).c_str(),
               byte == 7 ? "ran" : "did not run");
    }
    key("6.none.graph", graph_state(api, s.graph));
    key("6.none.clone", s.clone_rc == CUDA_SUCCESS ? graph_state(api, s.clone)
                                                   : "unavailable");
    key("6.none.exec", s.inst_rc == CUDA_SUCCESS ? exec_state(api, s.exec)
                                                 : "unavailable");
    return;
  }

  if (object == "clone" && s.clone_rc != CUDA_SUCCESS) {
    key(k, "unavailable");
    return;
  }
  if (object == "exec" && s.inst_rc != CUDA_SUCCESS) {
    key(k, "unavailable");
    return;
  }
  CUcontext doomed = which == "A" ? s.A : s.B;
  CUcontext other = which == "A" ? s.B : s.A;
  cuCtxSetCurrent(nullptr);
  need(cuCtxDestroy(doomed), "cuCtxDestroy");
  use(other, "cuCtxSetCurrent(the surviving context)");
  std::string st;
  if (object == "graph") st = graph_state(api, s.graph);
  if (object == "clone") st = graph_state(api, s.clone);
  if (object == "exec") st = exec_state(api, s.exec);
  OBSERVED("after destroying %s, the %s answers: %s", which.c_str(),
           object.c_str(), st.c_str());
  key(k, st);
}
#else
void check_graph(const std::string& which, const std::string&) {
  if (which == "none") {
    SKIP("compiled out: cuda.h CUDA_VERSION %d predates the graph API (needs "
         "12080)",
         CUDA_VERSION);
  }
  key("6.skip", "compiled-out");
}
#endif

// =============================================================================
// Check 7: cuDevicePrimaryCtxReset with two retains outstanding.
// =============================================================================
// Verifies:
//  - the fake's reset leaves the retain count alone and the context active
//    (tests/fake_cuda.cpp cuDevicePrimaryCtxReset_v2 ~line 628-650, quoting
//    "Resetting the primary context does not release it"). The server erases
//    the session's owed retains on a reset (server/inventory.cpp forget_primary
//    ~line 251, "forgetting them can at worst leak one retain", per
//    issue-4-report.md review follow-up 1 & 2), so a surviving count means
//    that erase leaks retains - known and accepted - and a dropped count
//    means the erase is exactly right;
//  - the handle is the same afterwards: spec "What changed from the first
//    draft", "Primary contexts keep currency and carry no generation in
//    slots ... the handle survives, and a CUDA thread keeps it current"
//    (server/client_threads.h SavedContext, no generation);
//  - what was in the context is gone (the server forgets it; forget_primary).
void check_primary_reset() {
  heading("cuDevicePrimaryCtxReset after two retains",
          "retain count and handle survive; contents are destroyed; the "
          "context stays current and usable");
  init();
  CUcontext p1 = nullptr, p2 = nullptr, p3 = nullptr;
  need(cuDevicePrimaryCtxRetain(&p1, g_dev), "retain 1");
  need(cuDevicePrimaryCtxRetain(&p2, g_dev), "retain 2");
  unsigned int flags = 0;
  int active0 = -1;
  cuDevicePrimaryCtxGetState(g_dev, &flags, &active0);
  OBSERVED("two retains give the same handle: %s; active before reset: %d",
           yn(p1 == p2), active0);
  use(p1, "cuCtxSetCurrent(primary)");
  CUdeviceptr x = 0;
  need(cuMemAlloc(&x, 16u << 20), "cuMemAlloc in the primary context");

  const CUresult reset = cuDevicePrimaryCtxReset(g_dev);
  if (reset != CUDA_SUCCESS) {
    FAIL("[server] cuDevicePrimaryCtxReset with retains outstanding and the "
         "context current -> %s. The server allows a reset when its session "
         "is alone and the e2e check expects it to succeed; the fake always "
         "resets",
         rc(reset).c_str());
    return;
  }
  int active1 = -1;
  cuDevicePrimaryCtxGetState(g_dev, &flags, &active1);
  const CUcontext cur = current();
  CUcontext owner = nullptr;
  const CUresult old_ptr =
      cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, x);
  CUdeviceptr y = 0;
  const CUresult fresh = cuMemAlloc(&y, 4096);
  OBSERVED("after reset: active %d, still current: %s, the old allocation "
           "answers %s, a new allocation with it current -> %s",
           active1, yn(cur == p1), rc(old_ptr).c_str(), rc(fresh).c_str());
  if (old_ptr != CUDA_SUCCESS) {
    PASS("[server] the reset destroyed the allocation made before it, as "
         "forget_primary assumes");
  } else {
    FAIL("[server] an allocation made before the reset still answers "
         "(context %p); forget_primary forgets it anyway, so expiry would "
         "leak it",
         (void*)owner);
  }
  if (fresh == CUDA_SUCCESS) cuMemFree(y);

  need(cuDevicePrimaryCtxRetain(&p3, g_dev), "retain after reset");
  if (p3 == p1) {
    PASS("[server] the primary context's handle is the same after the reset");
  } else {
    FAIL("[server] the handle changed across the reset (%p -> %p); slots keep "
         "primary contexts with no generation and would name a dead handle",
         (void*)p1, (void*)p3);
  }

  // Three retains are owed if the count survived (2 + 1), one if it did not.
  // The active flag after each release tells which; a fourth release shows
  // how the driver treats one nobody paid for.
  int act[4] = {-1, -1, -1, -1};
  CUresult rel[4];
  cuCtxSetCurrent(nullptr);
  for (int i = 0; i < 4; i++) {
    rel[i] = cuDevicePrimaryCtxRelease(g_dev);
    cuDevicePrimaryCtxGetState(g_dev, &flags, &act[i]);
  }
  OBSERVED("releases after the reset: %s (active %d), %s (active %d), %s "
           "(active %d), %s (active %d)",
           rc(rel[0]).c_str(), act[0], rc(rel[1]).c_str(), act[1],
           rc(rel[2]).c_str(), act[2], rc(rel[3]).c_str(), act[3]);
  const bool survived = rel[0] == CUDA_SUCCESS && rel[1] == CUDA_SUCCESS &&
                        rel[2] == CUDA_SUCCESS && act[0] == 1 && act[1] == 1 &&
                        act[2] == 0;
  const bool dropped = rel[0] == CUDA_SUCCESS && act[0] == 0;
  if (survived) {
    PASS("[fake] the retain count survived the reset (two releases left it "
         "active, the third made it inactive), as the fake models. Note: the "
         "server erases a session's owed retains on reset, so each reset "
         "leaks that session's retains until the server exits");
  } else if (dropped) {
    FAIL("[fake] the reset dropped the retain count (one release after "
         "reset+retain made it inactive); the fake keeps it. The server's "
         "erase in forget_primary is exactly right for this driver");
  } else {
    FAIL("[fake] neither reading fits the release/active sequence above");
  }
  if (rel[3] == CUDA_SUCCESS) {
    OBSERVED("an over-release returned SUCCESS: the driver does not refuse "
             "it, so an over-release by the server would be silent");
  } else {
    OBSERVED("an over-release was refused with %s (the fake refuses with 201 "
             "INVALID_CONTEXT)",
             rc(rel[3]).c_str());
  }
}

// =============================================================================
// Check 8: releasing the last retain.
// =============================================================================
// Verifies:
//  - cuDevicePrimaryCtxGetState reports inactive right after the last
//    release: server/inventory.cpp primary_inactive / w_cuDevicePrimaryCtxRelease_v2
//    (~line 340-370) read it to decide that the release destroyed the context
//    and forget everything in it. If it still says active, the server never
//    forgets, and a later expiry frees dead handles;
//  - the contents are destroyed with it (fake cuDevicePrimaryCtxRelease_v2
//    take_contents, ~line 605);
//  - a thread that still has it current gets CUDA_ERROR_CONTEXT_IS_DESTROYED
//    (slice A contract "Codes that apply": "including a primary context with
//    no retains"; fake enter_locked ~line 147);
//  - the handle is the same when retained again (spec: "the handle survives";
//    issue-2-sliceA-report.md concern 5, g_primary_dev never pruned).
void check_primary_last_release() {
  heading("releasing the last retain of the primary context",
          "inactive afterwards; contents gone; calls under it fail with "
          "CONTEXT_IS_DESTROYED; same handle when retained again");
  init();
  CUcontext p = nullptr, q = nullptr;
  need(cuDevicePrimaryCtxRetain(&p, g_dev), "retain");
  use(p, "cuCtxSetCurrent(primary)");
  CUdeviceptr x = 0;
  need(cuMemAlloc(&x, 16u << 20), "cuMemAlloc in the primary context");
  unsigned int flags = 0;
  int before = -1, after = -1;
  cuDevicePrimaryCtxGetState(g_dev, &flags, &before);
  const CUresult rel = cuDevicePrimaryCtxRelease(g_dev);
  cuDevicePrimaryCtxGetState(g_dev, &flags, &after);
  if (rel == CUDA_SUCCESS && before == 1 && after == 0) {
    PASS("[server] GetState active went 1 -> 0 with the last release");
  } else {
    FAIL("[server] release -> %s, active %d -> %d. primary_inactive reads "
         "this flag to learn that a release destroyed the context",
         rc(rel).c_str(), before, after);
  }
  const bool still_current = current() == p;
  CUdeviceptr y = 0;
  const CUresult use_it = cuMemAlloc(&y, 4096);
  if (use_it == CUDA_SUCCESS) cuMemFree(y);
  OBSERVED("the released context is still current on this thread: %s",
           yn(still_current));
  if (use_it == CUDA_ERROR_CONTEXT_IS_DESTROYED) {
    PASS("[server] cuMemAlloc with the released primary context current -> "
         "%s",
         rc(use_it).c_str());
  } else {
    FAIL("[server] cuMemAlloc with the released primary context current -> "
         "%s; the contract (and the fake) say CONTEXT_IS_DESTROYED",
         rc(use_it).c_str());
  }
  CUcontext owner = nullptr;
  const CUresult old_ptr =
      cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, x);
  if (old_ptr != CUDA_SUCCESS) {
    PASS("[server] the allocation died with the last release (answers %s)",
         rc(old_ptr).c_str());
  } else {
    FAIL("[server] an allocation outlived the last release (context %p)",
         (void*)owner);
  }
  need(cuDevicePrimaryCtxRetain(&q, g_dev), "retain again");
  if (q == p) {
    PASS("[server] retained again, the handle is the same");
  } else {
    FAIL("[server] retained again, the handle changed (%p -> %p): a slot or "
         "g_primary_dev entry naming the old handle names nothing",
         (void*)p, (void*)q);
  }
  // A thread that kept it current, after somebody else retains it again.
  const bool kept = current() == p;
  const CUresult again = cuMemAlloc(&y, 4096);
  OBSERVED("after the new retain, the thread that kept the old handle current "
           "(still current: %s) allocates -> %s",
           yn(kept), rc(again).c_str());
  if (again == CUDA_SUCCESS) cuMemFree(y);
  cuCtxSetCurrent(nullptr);
  cuDevicePrimaryCtxRelease(g_dev);
}

// =============================================================================
// Check 9: cuCtxSetCurrent on a destroyed context.
// =============================================================================
// Verifies: server/client_threads.cpp show_context (~line 232-250) treats a
// failed cuCtxSetCurrent as "destroyed where no sweep saw it" and marks the
// entry gone; the fake refuses with CUDA_ERROR_INVALID_CONTEXT
// (tests/fake_cuda.cpp cuCtxSetCurrent ~line 547, bindable_locked).
// PASS: refused (the code is reported; the fake uses 201).
// FAIL [server]: accepted. show_context would make a destroyed context
//       current and run a client thread's request under it instead of
//       failing it cleanly; the detection of unswept destroys does not work.
// Also observed: whether the driver hands a destroyed context's address to
// the next cuCtxCreate (server/client_threads.h:37 assumes it may; the fake
// only does so under RGPU_FAKE_REUSE_CONTEXTS=1).
void check_set_destroyed() {
  heading("cuCtxSetCurrent on a destroyed context handle",
          "refused, so the server can detect a destroy it did not see");
  init();
  CUcontext c = make_ctx();
  need(cuCtxDestroy(c), "cuCtxDestroy");
  const CUresult r = cuCtxSetCurrent(c);
  if (r != CUDA_SUCCESS) {
    PASS("[server] refused with %s (fake: 201 INVALID_CONTEXT)",
         rc(r).c_str());
  } else {
    CUdeviceptr y = 0;
    const CUresult use_it = cuMemAlloc(&y, 4096);
    FAIL("[server] accepted; afterwards current is %s and cuMemAlloc -> %s. "
         "show_context relies on a refusal to notice destroyed contexts",
         current() == c ? "the destroyed handle" : "something else",
         rc(use_it).c_str());
    cuCtxSetCurrent(nullptr);
  }
  CUcontext d = nullptr;
  const CUresult pushed = cuCtxPushCurrent(c);
  if (pushed == CUDA_SUCCESS) cuCtxPopCurrent(&d);
  OBSERVED("cuCtxPushCurrent of the destroyed handle -> %s", rc(pushed).c_str());

  std::vector<CUcontext> destroyed = {c};
  int reused = 0;
  for (int i = 0; i < 8; i++) {
    CUcontext e = make_ctx();
    for (CUcontext old : destroyed) reused += (old == e);
    destroyed.push_back(e);
    cuCtxDestroy(e);
  }
  OBSERVED("of 8 new contexts, %d got the address of a context destroyed "
           "before it",
           reused);
}

// =============================================================================
// Check 10: a thread whose current context another thread destroyed.
// =============================================================================
// Verifies: the header's "If ctx is current to other threads, then ctx will
// remain current to those threads" and CUDA_ERROR_CONTEXT_IS_DESTROYED, which
// the server emulates rather than uses: w_cuCtxGetCurrent answers the
// destroyed top (server/client_threads.cpp ~line 197-212), and
// client_thread_after corrects INVALID_CONTEXT to CONTEXT_IS_DESTROYED
// (~line 323-370); asserted by tests/threadctx_smoke.cpp:549,671,746.
// PASS: cuCtxGetCurrent -> SUCCESS naming the destroyed handle; cuMemAlloc ->
//       CUDA_ERROR_CONTEXT_IS_DESTROYED.
// FAIL [server]: a different answer; the server's emulation invents a
//       behaviour, and threadctx_smoke asserts it. A crash is OBSERVED: the
//       server never runs a request under a destroyed context, so it is
//       unaffected, but the emulated codes are unverifiable.
void check_destroyed_by_other_thread() {
  heading("a thread whose current context was destroyed by another thread",
          "the handle stays current; calls fail with CONTEXT_IS_DESTROYED");
  init();
  std::mutex mu;
  std::condition_variable cv;
  int stage = 0;
  CUcontext c = nullptr;
  auto advance = [&](int to) {
    std::lock_guard<std::mutex> lk(mu);
    stage = to;
    cv.notify_all();
  };
  auto await = [&](int at) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return stage >= at; });
  };
  std::thread t([&] {
    need(cuCtxCreate(&c, 0, g_dev), "cuCtxCreate on the thread");
    CUdeviceptr m = 0;
    need(cuMemAlloc(&m, 4096), "cuMemAlloc on the thread");
    advance(1);
    await(2);
    CUcontext seen = nullptr;
    const CUresult got = cuCtxGetCurrent(&seen);
    CUdeviceptr y = 0;
    const CUresult alloc = cuMemAlloc(&y, 4096);
    const CUresult sync = cuCtxSynchronize();
    CUdevice dev = -1;
    const CUresult getdev = cuCtxGetDevice(&dev);
    if (got == CUDA_SUCCESS && seen == c) {
      PASS("[server] cuCtxGetCurrent -> SUCCESS and still names the destroyed "
           "context");
    } else {
      FAIL("[server] cuCtxGetCurrent -> %s naming %p (destroyed context %p); "
           "w_cuCtxGetCurrent answers the destroyed handle",
           rc(got).c_str(), (void*)seen, (void*)c);
    }
    if (alloc == CUDA_ERROR_CONTEXT_IS_DESTROYED) {
      PASS("[server] cuMemAlloc -> %s", rc(alloc).c_str());
    } else {
      FAIL("[server] cuMemAlloc -> %s; client_thread_after reports "
           "CUDA_ERROR_CONTEXT_IS_DESTROYED for it",
           rc(alloc).c_str());
    }
    OBSERVED("cuCtxSynchronize -> %s, cuCtxGetDevice -> %s",
             rc(sync).c_str(), rc(getdev).c_str());
    CUcontext popped = nullptr;
    const CUresult pop = cuCtxPopCurrent(&popped);
    OBSERVED("cuCtxPopCurrent -> %s, returning the destroyed handle: %s; "
             "current afterwards: %p",
             rc(pop).c_str(), yn(popped == c), (void*)current());
    advance(3);
  });
  await(1);
  const CUresult destroyed = cuCtxDestroy(c);
  OBSERVED("cuCtxDestroy from the other thread -> %s", rc(destroyed).c_str());
  advance(2);
  await(3);
  t.join();
}

// =============================================================================
// Check 11: the context stack.
// =============================================================================
// Verifies:
//  - cuCtxSetCurrent(NULL) pops one entry, not the whole stack:
//    server/client_threads.cpp w_cuCtxSetCurrent (~line 128-150, "null pops
//    it") and the fake (~line 547). clear() tolerates either for the serving
//    thread, but the emulation offered to client threads pops one.
//  - cuCtxSetCurrent(ctx) replaces the top rather than pushing
//    (w_cuCtxSetCurrent, "Replaces the top of the thread's stack").
//  - cuCtxCreate pushes (client_threads_created, "The driver pushed the new
//    context").
//  - cuCtxDestroy of the current context pops it (client_threads_destroyed,
//    "The header says it was popped").
// PASS/FAIL [server] each.
void check_stack() {
  heading("the context stack", "SetCurrent(NULL) pops one; SetCurrent "
          "replaces the top; create pushes; destroying the top pops it");
  init();
  CUcontext A = nullptr, B = nullptr, C = nullptr;
  need(cuCtxCreate(&A, 0, g_dev), "cuCtxCreate A");
  need(cuCtxCreate(&B, 0, g_dev), "cuCtxCreate B");
  const bool pushed = current() == B;
  // The stack is [A, B] if create pushes.
  const CUresult nul = cuCtxSetCurrent(nullptr);
  const CUcontext after_null = current();
  if (pushed && nul == CUDA_SUCCESS && after_null == A) {
    PASS("[server] cuCtxCreate pushed, and cuCtxSetCurrent(NULL) with a stack "
         "two deep popped one: A is current");
  } else if (after_null == nullptr) {
    FAIL("[server] cuCtxSetCurrent(NULL) with a stack two deep -> %s and "
         "cleared it (nothing current); w_cuCtxSetCurrent pops only one",
         rc(nul).c_str());
  } else {
    FAIL("[server] create pushed: %s; cuCtxSetCurrent(NULL) -> %s leaving %p "
         "current (A=%p B=%p)",
         yn(pushed), rc(nul).c_str(), (void*)after_null, (void*)A, (void*)B);
  }
  // [A]: push B -> [A, B]; SetCurrent(A) replaces -> [A, A], or pushes ->
  // [A, B, A]; a pop then leaves A or B current.
  need(cuCtxPushCurrent(B), "cuCtxPushCurrent B");
  need(cuCtxSetCurrent(A), "cuCtxSetCurrent A");
  CUcontext popped = nullptr;
  need(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
  const CUcontext after_pop = current();
  if (after_pop == A) {
    PASS("[server] cuCtxSetCurrent replaced the top (a pop after it left A "
         "current)");
  } else {
    FAIL("[server] after push B, SetCurrent A, pop: current is %p (A=%p B=%p); "
         "SetCurrent did not replace the top",
         (void*)after_pop, (void*)A, (void*)B);
  }
  // Now [A]. Push C and destroy it while it is current.
  need(cuCtxCreate(&C, 0, g_dev), "cuCtxCreate C");
  need_true(current() == C, "cuCtxCreate C did not make it current");
  const CUresult destroy = cuCtxDestroy(C);
  const CUcontext after_destroy = current();
  if (destroy == CUDA_SUCCESS && after_destroy == A) {
    PASS("[server] destroying the current context popped it: A is current");
  } else {
    FAIL("[server] cuCtxDestroy of the current context -> %s leaving %p "
         "current (A=%p, destroyed %p)",
         rc(destroy).c_str(), (void*)after_destroy, (void*)A, (void*)C);
  }
  cuCtxSetCurrent(nullptr);
  const CUresult second = cuCtxSetCurrent(nullptr);
  CUcontext none = nullptr;
  const CUresult empty_pop = cuCtxPopCurrent(&none);
  OBSERVED("on an empty stack: cuCtxSetCurrent(NULL) -> %s, cuCtxPopCurrent "
           "-> %s",
           rc(second).c_str(), rc(empty_pop).c_str());
  cuCtxDestroy(A);
  cuCtxDestroy(B);
}

// =============================================================================
// Check 12: a new thread's stream capture mode.
// =============================================================================
// Verifies: server/client_threads.h:83-87 (a client thread starts at
// CU_STREAM_CAPTURE_MODE_GLOBAL, "what the header calls the default") and
// applied_mode's starting value (~line 108), which the server uses to skip
// the exchange; the fake's thread_local default (~line 1123).
// PASS: a new thread reads GLOBAL, and the mode is per thread (a mode set on
//       one thread is not seen by a thread started after it).
// FAIL [server]: the first request of a new client thread would run under the
//       wrong mode and the skip would never correct it.
void check_capture_mode() {
  heading("a newly created thread's stream capture mode",
          "GLOBAL, and kept per thread");
#if PROBE_HAVE_GRAPH_API
  init();
  auto exchange = reinterpret_cast<decltype(&cuThreadExchangeStreamCaptureMode)>(
      ::dlsym(RTLD_DEFAULT, "cuThreadExchangeStreamCaptureMode"));
  if (!exchange) {
    SKIP("this driver lacks cuThreadExchangeStreamCaptureMode");
    return;
  }
  auto read_mode = [&](CUresult* r) {
    CUstreamCaptureMode m = CU_STREAM_CAPTURE_MODE_GLOBAL;
    *r = exchange(&m);
    CUstreamCaptureMode back = m;
    if (*r == CUDA_SUCCESS) exchange(&back);  // put it back
    return m;
  };
  CUresult r1 = CUDA_ERROR_UNKNOWN, r2 = CUDA_ERROR_UNKNOWN,
           r3 = CUDA_ERROR_UNKNOWN;
  CUstreamCaptureMode fresh = CU_STREAM_CAPTURE_MODE_RELAXED;
  std::thread([&] { fresh = read_mode(&r1); }).join();
  // Set THREAD_LOCAL here, then start another thread.
  CUstreamCaptureMode mine = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
  r2 = exchange(&mine);
  CUstreamCaptureMode later = CU_STREAM_CAPTURE_MODE_RELAXED;
  CUresult r4 = CUDA_ERROR_UNKNOWN;
  std::thread([&] { later = read_mode(&r4); }).join();
  CUstreamCaptureMode kept = CU_STREAM_CAPTURE_MODE_GLOBAL;
  r3 = exchange(&kept);
  OBSERVED("new thread's mode -> %s, %d; after this thread set THREAD_LOCAL "
           "(-> %s), a newer thread's mode -> %s, %d, and this thread's own -> "
           "%s, %d",
           rc(r1).c_str(), static_cast<int>(fresh), rc(r2).c_str(),
           rc(r4).c_str(), static_cast<int>(later), rc(r3).c_str(),
           static_cast<int>(kept));
  if (r1 == CUDA_SUCCESS && fresh == CU_STREAM_CAPTURE_MODE_GLOBAL) {
    PASS("[server] a new thread starts in CU_STREAM_CAPTURE_MODE_GLOBAL");
  } else {
    FAIL("[server] a new thread's mode is %d (exchange -> %s), not GLOBAL (0)",
         static_cast<int>(fresh), rc(r1).c_str());
  }
  if (r4 == CUDA_SUCCESS && later == CU_STREAM_CAPTURE_MODE_GLOBAL &&
      r3 == CUDA_SUCCESS && kept == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) {
    PASS("[server] the mode is per thread: another thread's THREAD_LOCAL was "
         "not seen by a new thread, and was kept by its own");
  } else {
    FAIL("[server] the mode does not behave per thread (see the values above)");
  }
#else
  SKIP("compiled out: cuda.h CUDA_VERSION %d predates "
       "cuThreadExchangeStreamCaptureMode (needs 12080)",
       CUDA_VERSION);
#endif
}

// =============================================================================
// Check 13: what cuCtxGetCurrent costs.
// =============================================================================
// Measures: the one cuCtxGetCurrent the server now makes per request
// (client_thread_after, server/client_threads.cpp ~line 303-308; spec Risks,
// "its cost on hardware is unmeasured ... Take the measurement before taking
// the trade"). Also a same-context and an alternating cuCtxSetCurrent, which
// the server makes when client threads interleave. OBSERVED only; compare
// with a round trip of ~28 us (tests/bench.cpp on the fake).
void check_cost() {
  heading("what cuCtxGetCurrent and cuCtxSetCurrent cost",
          "a thread-local read, negligible next to a ~28 us round trip");
  init();
  CUcontext A = make_ctx();
  CUcontext B = make_ctx();
  const long n = g_iterations;
  auto time_ns = [&](const std::function<void()>& body, long iters) {
    for (long i = 0; i < 10000; i++) body();
    const auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; i++) body();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
  };
  CUcontext sink = nullptr;
  const double none = time_ns([&] { cuCtxGetCurrent(&sink); }, n);
  use(A, "cuCtxSetCurrent(A)");
  const double some = time_ns([&] { cuCtxGetCurrent(&sink); }, n);
  const double same = time_ns([&] { cuCtxSetCurrent(A); }, n / 10);
  bool flip = false;
  const double alternate = time_ns(
      [&] {
        flip = !flip;
        cuCtxSetCurrent(flip ? B : A);
      },
      n / 10);
  OBSERVED("cuCtxGetCurrent: %.1f ns/call with no context current, %.1f "
           "ns/call with one (%ld calls each)",
           none, some, n);
  OBSERVED("cuCtxSetCurrent: %.1f ns/call to the context already current, "
           "%.1f ns/call alternating between two (%ld calls each)",
           same, alternate, n / 10);
  cuCtxSetCurrent(nullptr);
  cuCtxDestroy(A);
  cuCtxDestroy(B);
}

// --- child dispatch ----------------------------------------------------------

int run_check(const std::string& id) {
  g_id = id;
  if (id == "0") check_env();
  else if (id == "1") check_cross_context_memory();
  else if (id == "4") check_module_unload();
  else if (id == "5a") check_launch(false);
  else if (id == "5b") check_launch(true);
  else if (id.rfind("6.", 0) == 0) {
    const size_t dot = id.find('.', 2);
    const std::string which = id.substr(2, dot == std::string::npos
                                               ? std::string::npos
                                               : dot - 2);
    const std::string object =
        dot == std::string::npos ? "all" : id.substr(dot + 1);
    check_graph(which, object);
  } else if (id == "7") check_primary_reset();
  else if (id == "8") check_primary_last_release();
  else if (id == "9") check_set_destroyed();
  else if (id == "10") check_destroyed_by_other_thread();
  else if (id == "11") check_stack();
  else if (id == "12") check_capture_mode();
  else if (id == "13") check_cost();
  else {
    std::fprintf(stderr, "unknown check %s\n", id.c_str());
    return 1;
  }
  return (g_fail ? 1 : 0) | (g_skip ? 2 : 0);
}

// --- parent ------------------------------------------------------------------

struct Check {
  const char* id;
  // A crash is an answer, not a failure of the probe: these use handles whose
  // context is gone on purpose.
  bool crash_is_data;
};

const Check kChecks[] = {
    {"0", false},          {"1", false},          {"4", false},
    {"5a", false},         {"5b", false},         {"6.none", false},
    {"6.dead.graph", true}, {"6.dead.exec", true}, {"6.B.graph", true},
    {"6.B.clone", true},   {"6.B.exec", true},    {"6.A.graph", true},
    {"6.A.clone", true},   {"6.A.exec", true},    {"7", false},
    {"8", false},          {"9", false},          {"10", true},
    {"11", false},         {"12", false},         {"13", false},
};

struct ChildResult {
  int status = 0;
  std::string output;
};

ChildResult run_child(const char* self, const std::string& id,
                      const std::vector<std::string>& pass_through) {
  ChildResult res;
  int fds[2];
  if (::pipe(fds) != 0) {
    std::perror("pipe");
    res.status = 1 << 8;
    return res;
  }
  pid_t pid = ::fork();
  if (pid == 0) {
    ::close(fds[0]);
    ::dup2(fds[1], 1);
    ::dup2(fds[1], 2);
    ::close(fds[1]);
    std::vector<std::string> args = {self, "--check", id};
    args.insert(args.end(), pass_through.begin(), pass_through.end());
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(&a[0]);
    argv.push_back(nullptr);
    ::execv("/proc/self/exe", argv.data());
    ::execv(self, argv.data());
    std::perror("execv");
    std::_Exit(127);
  }
  ::close(fds[1]);
  char buf[4096];
  for (;;) {
    ssize_t n = ::read(fds[0], buf, sizeof(buf));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    std::fwrite(buf, 1, static_cast<size_t>(n), stdout);
    std::fflush(stdout);
    res.output.append(buf, static_cast<size_t>(n));
  }
  ::close(fds[0]);
  while (::waitpid(pid, &res.status, 0) < 0 && errno == EINTR) {
  }
  return res;
}

// "alive", "dead", "crashed", "unavailable" or "" for a KEY value.
std::string kind(const std::string& v) {
  return v.substr(0, v.find(' '));
}

}  // namespace

int main(int argc, char** argv) {
  std::string check;
  std::string only;
  int timeout = 180;
  std::vector<std::string> pass_through;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(64);
      }
      return argv[++i];
    };
    if (a == "--check") {
      check = value();
    } else if (a == "--image") {
      g_image_path = value();
      pass_through.push_back("--image");
      pass_through.push_back(g_image_path);
    } else if (a == "--iterations") {
      g_iterations = std::atol(value().c_str());
      pass_through.push_back("--iterations");
      pass_through.push_back(std::to_string(g_iterations));
    } else if (a == "--only") {
      only = "," + value() + ",";
    } else if (a == "--timeout") {
      timeout = std::atoi(value().c_str());
    } else {
      std::fprintf(stderr,
                   "usage: %s [--image fatbin|ptx] [--only 1,6,7] "
                   "[--timeout seconds] [--iterations n]\n",
                   argv[0]);
      return 64;
    }
  }

  if (!check.empty()) {
    // A hung driver call must not hang the whole probe: the parent reports
    // SIGALRM as a timeout.
    ::alarm(static_cast<unsigned>(timeout));
    return run_check(check);
  }

  std::printf("rgpu driver probe: every check below runs in its own process "
              "against the real driver\n");
  int fails = 0, skips = 0, crashes = 0;
  std::map<std::string, std::string> keys;
  for (const Check& c : kChecks) {
    const std::string id = c.id;
    const std::string group = id.substr(0, id.find('.'));
    if (!only.empty() && only.find("," + group + ",") == std::string::npos &&
        only.find("," + id + ",") == std::string::npos) {
      continue;
    }
    pass_through.push_back("--timeout");
    pass_through.push_back(std::to_string(timeout));
    const ChildResult r = run_child(argv[0], id, pass_through);
    pass_through.resize(pass_through.size() - 2);
    size_t at = 0;
    while ((at = r.output.find("KEY ", at)) != std::string::npos) {
      if (at == 0 || r.output[at - 1] == '\n') {
        const size_t eol = r.output.find('\n', at);
        const std::string line = r.output.substr(at + 4, eol - at - 4);
        const size_t sp = line.find(' ');
        keys[line.substr(0, sp)] =
            sp == std::string::npos ? "" : line.substr(sp + 1);
      }
      at += 4;
    }
    if (WIFSIGNALED(r.status)) {
      const int sig = WTERMSIG(r.status);
      if (sig == SIGALRM) {
        std::printf("FAIL     %-12s timed out after %ds\n", id.c_str(),
                    timeout);
        fails++;
      } else if (c.crash_is_data) {
        std::printf("OBSERVED %-12s the process died with signal %d (%s) "
                    "after its last line above\n",
                    id.c_str(), sig, strsignal(sig));
        keys[id] = "crashed " + std::to_string(sig);
        crashes++;
      } else {
        std::printf("FAIL     %-12s the process died with signal %d (%s) "
                    "after its last line above\n",
                    id.c_str(), sig, strsignal(sig));
        fails++;
      }
    } else if (WIFEXITED(r.status)) {
      const int code = WEXITSTATUS(r.status);
      if (code & 1 || code > 3) fails++;
      if (code & 2) skips++;
    }
  }

  // Check 6, combined.
  const bool ran_graphs = keys.count("6.none.graph") || keys.count("6.skip");
  if (ran_graphs && !keys.count("6.skip")) {
    std::printf("\n== 6: verdict, combining the runs above\n");
    g_id = "6";
    const bool ended_under_b = keys["6.ended_under_b"] == "1";
    for (const char* object : {"graph", "clone", "exec"}) {
      const std::string none = keys["6.none." + std::string(object)];
      const std::string in_a = kind(keys["6.A." + std::string(object)]);
      const std::string in_b = kind(keys["6.B." + std::string(object)]);
      const std::string dead_ctl =
          kind(keys[std::string("6.dead.") +
                    (std::strcmp(object, "exec") == 0 ? "exec" : "graph")]);
      const char* what =
          std::strcmp(object, "graph") == 0
              ? (ended_under_b ? "the graph cuStreamEndCapture returned with "
                                 "B current"
                               : "the captured graph (EndCapture under B gave "
                                 "no graph, so this one was ended under A)")
          : std::strcmp(object, "clone") == 0
              ? "the clone made with B current"
              : "the executable instantiated with B current";
      if (kind(none) != "alive") {
        std::printf("SKIP     6.%-10s %s: not available (%s)\n", object, what,
                    none.empty() ? "no result" : none.c_str());
        skips++;
        continue;
      }
      const bool dead_b = in_b == "dead" || in_b == "crashed";
      const bool dead_a = in_a == "dead" || in_a == "crashed";
      const bool trust_alive = dead_ctl == "dead" || dead_ctl == "crashed";
      if (dead_b) {
        std::printf("FAIL     6.%-10s [server] %s died with B, the context "
                    "current when it was made (A: %s, B: %s). stamp_of records "
                    "it under its source's context, so B's destruction goes "
                    "unseen and expiry frees a dead handle\n",
                    object, what, in_a.c_str(), in_b.c_str());
        fails++;
      } else if (!trust_alive) {
        std::printf("OBSERVED 6.%-10s %s: A: %s, B: %s, but a destroyed "
                    "handle also answers \"%s\", so a live answer proves "
                    "nothing\n",
                    object, what, in_a.c_str(), in_b.c_str(),
                    dead_ctl.c_str());
      } else if (dead_a) {
        std::printf("PASS     6.%-10s [server] %s survived B's destruction "
                    "and died with A's: it is its source's, as stamp_of "
                    "records it\n",
                    object, what);
      } else {
        std::printf("FAIL     6.%-10s [fake] %s survived both contexts' "
                    "destruction: it belongs to no context. The server "
                    "forgets it when A is destroyed and leaks it (safe); the "
                    "fake destroys it with A\n",
                    object, what);
        fails++;
      }
    }
  }

  std::printf("\nsummary: %d FAIL, %d SKIP, %d crash(es) recorded as "
              "observations\n",
              fails, skips, crashes);
  return fails ? 1 : skips ? 2 : 0;
}
