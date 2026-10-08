#include "client/pending.h"

#include <mutex>
#include <unordered_set>

namespace rgpu {
namespace {

// Never destroyed, like every other global a CUDA call can reach: threads keep
// calling in while the process exits.
auto& g_mu = *new std::mutex();
auto& g_busy = *new std::unordered_set<uint64_t>();
bool g_unplaced = false;

}  // namespace

void work_on(uint64_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_busy.insert(stream);
}

void work_somewhere() {
  std::lock_guard<std::mutex> lk(g_mu);
  g_unplaced = true;
}

void settled(uint64_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_busy.erase(stream);
}

void settled_all() {
  std::lock_guard<std::mutex> lk(g_mu);
  g_busy.clear();
  g_unplaced = false;
}

bool nothing_to_wait_for(uint64_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_unplaced) return false;
  // The default stream waits for work on the others, so it is idle only when
  // they all are.
  if (stream == 0) return g_busy.empty();
  return g_busy.count(stream) == 0;
}

}  // namespace rgpu
