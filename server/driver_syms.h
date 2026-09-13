// Lazily resolved driver entry points.
//
// The server is built against one set of CUDA headers but runs against
// whatever driver the host has, which is often older. Linking every entry
// point directly would mean the whole server fails to load because of one
// symbol it never calls. Resolving on first use instead turns that into a
// single call reporting CUDA_ERROR_NOT_SUPPORTED.
#pragma once

#include <dlfcn.h>

#include <cstdio>
#include <mutex>

#include "server/inventory.h"

namespace rgpu {

// Returns the address of `name` in the real driver, or nullptr. Logs the first
// time a name cannot be found, so an older driver explains itself.
inline void* driver_sym_raw(const char* name) {
  // Whatever is already in the process wins: the server links the driver for
  // the calls it makes directly, and the test build compiles a fake driver
  // straight into the binary.
  if (void* fn = ::dlsym(RTLD_DEFAULT, name)) return fn;

  static void* handle = [] {
    void* h = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) std::fprintf(stderr, "[rgpu-server] dlopen libcuda.so.1: %s\n",
                         ::dlerror());
    return h;
  }();
  void* fn = handle ? ::dlsym(handle, name) : nullptr;
  if (!fn) {
    std::fprintf(stderr,
                 "[rgpu-server] this driver has no %s; reporting it as "
                 "unsupported\n", name);
  }
  return fn;
}

// What the server should actually call for `name`.
//
// For most calls that is the driver's own entry point. For the handful that
// hand out something a client can still be holding when it dies - memory, a
// context, a module, a stream - it is a wrapper that makes the same call and
// writes down what came back, so the session can give it up later. The driver
// is resolved first either way, so a call this driver does not have is still
// reported as unsupported rather than quietly wrapped.
inline void* driver_sym(const char* name) {
  void* fn = driver_sym_raw(name);
  if (!fn) return nullptr;
  if (void* tracked = driver_wrapper(name)) return tracked;
  return fn;
}

}  // namespace rgpu
