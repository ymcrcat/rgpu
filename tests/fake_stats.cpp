#include "tests/fake_stats.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace rgpu_fake {
namespace {

std::atomic<long> g_counts[kKindCount];

// Must line up with enum Kind.
const char* const kNames[kKindCount] = {
    "allocs",  "retains", "contexts", "modules",  "streams", "events",
    "graphs",  "execs",   "cublas",   "cublaslt", "cudnn",
    "overreleases",
};

// Serialises the writers, so two sessions releasing at once do not both write
// the same temporary file.
std::mutex g_mu;

// Written whole and renamed into place, so a reader never sees half a line.
void publish() {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  if (!path) return;
  std::lock_guard<std::mutex> lk(g_mu);
  const std::string tmp = std::string(path) + ".tmp";
  std::FILE* f = std::fopen(tmp.c_str(), "w");
  if (!f) return;
  for (int i = 0; i < kKindCount; i++) {
    std::fprintf(f, "%s%s=%ld", i ? " " : "", kNames[i],
                 g_counts[i].load(std::memory_order_relaxed));
  }
  std::fprintf(f, "\n");
  std::fclose(f);
  std::rename(tmp.c_str(), path);
}

}  // namespace

void count(Kind k, int delta) {
  g_counts[k].fetch_add(delta, std::memory_order_relaxed);
  publish();
}

}  // namespace rgpu_fake
