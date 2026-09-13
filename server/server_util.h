// Small utilities shared by the server's translation units: the one log
// function, the one verbose flag, and the one named-function-table lookup.
#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace rgpu {

// Set once from RGPU_VERBOSE in main(), before any serving thread starts, and
// only read afterwards. See server/main.cpp.
extern bool g_verbose;

// "[rgpu-server] " + message + newline, to stderr.
inline void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[rgpu-server] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

// A CUDA call's name paired with the server's stand-in entry point for it, as
// the wrapper tables hold it.
struct NamedFn {
  const char* name;
  void* fn;
};

// The entry point for `name` in `table`, or nullptr if it is not there.
inline void* lookup(const NamedFn* table, size_t n, const char* name) {
  for (size_t i = 0; i < n; i++) {
    if (std::strcmp(table[i].name, name) == 0) return table[i].fn;
  }
  return nullptr;
}

}  // namespace rgpu
