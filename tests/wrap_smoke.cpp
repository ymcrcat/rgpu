// Checks that request ids wrapping past 0xFFFFFFFF break nothing, including a
// reconnect whose replay straddles the wrap, without a GPU.
//
// Request ids are 32 bits and a long-lived client process can send more than
// four billion requests. Both sides order ids - the server to tell a request
// that already ran from one that has not, the client to tell which frames the
// server has acknowledged and which to send again - and an ordering that
// forgets the wrap either skips every request after it or replays ones that
// already ran.
//
// The client's ids are started just short of the wrap (RGPU_TEST_FIRST_REQ_ID)
// and the server breaks the connection at a set frame (RGPU_DROP_AFTER), so the
// break lands in a batch that spans the wrap. This test is the only client on
// its server and makes every call itself, one thread, through the client
// library's own call and call_async, so the frame count and each frame's id
// are known exactly.
//
// The call run without a reply is the fake driver's cuDeviceTotalMem, which it
// counts ("totalmem" in RGPU_FAKE_STATS), so the test can see that every one
// ran exactly once.
//
//   RGPU_DROP_AFTER=16 RGPU_FAKE_STATS=/tmp/s rgpu-server-fake 9727 &
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9727 RGPU_BATCH=1
//     RGPU_TEST_FIRST_REQ_ID=4294967286 RGPU_FAKE_STATS=/tmp/s ./wrap_smoke

#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda.h>

#include "client/rpc.h"
#include "common/generated/api_ids.h"

namespace {

int g_failures = 0;

#define EXPECT(cond, what)                                                \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what); \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

// A client waiting for a reply that will never come waits forever; a hang is
// what an ordering that forgets the wrap looks like, so it has to fail.
void on_timeout(int) {
  static const char msg[] =
      "FAIL: wrap_smoke timed out - a call across the request-id wrap was "
      "never answered\n";
  ssize_t ignored = ::write(2, msg, sizeof(msg) - 1);
  (void)ignored;
  ::_exit(1);
}

long totalmem_runs() {
  const char* path = std::getenv("RGPU_FAKE_STATS");
  if (!path) return -1;
  std::FILE* f = std::fopen(path, "r");
  if (!f) return 0;
  char buf[512] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  const std::string line = std::string(" ") + buf;
  const std::string key = " totalmem=";
  const size_t at = line.find(key);
  if (at == std::string::npos) return 0;
  return std::strtol(line.c_str() + at + key.size(), nullptr, 10);
}

// A call that replies.
CUresult device_count(int* count) {
  rgpu::Buffer req, rsp;
  req.put<uint8_t>(1);
  const CUresult r = rgpu::call(rgpu::API_cuDeviceGetCount, req, &rsp);
  if (r == CUDA_SUCCESS && !rsp.get(count)) return CUDA_ERROR_UNKNOWN;
  return r;
}

// A call sent without a reply, counted by the fake driver when it runs.
CUresult total_mem_async(int dev) {
  rgpu::Buffer req;
  req.put<uint8_t>(1);
  req.put<int>(dev);
  return rgpu::call_async(rgpu::API_cuDeviceTotalMem_v2, req);
}

}  // namespace

int main() {
  ::signal(SIGALRM, on_timeout);
  ::alarm(60);

  const char* first = std::getenv("RGPU_TEST_FIRST_REQ_ID");
  EXPECT(first && std::strtoul(first, nullptr, 0) == 4294967286ul,
         "RGPU_TEST_FIRST_REQ_ID must be 4294967286 for the frame numbers "
         "below to hold");
  EXPECT(totalmem_runs() == 0, "RGPU_FAKE_STATS is not set, or not fresh");

  constexpr int kBatch = 20;
  long sent_async = 0;

  // Frame 1, id 0xFFFFFFF6.
  int count = 0;
  EXPECT(device_count(&count) == CUDA_SUCCESS && count > 0,
         "the first call failed");

  // Frames 2-21: ids 0xFFFFFFF7 to 0xFFFFFFFF, then 1 to 11 - 0 names no
  // request, and is never used. Frame 13 (id 3) fails, with nothing to report
  // it in. The server breaks the connection on reading frame 16 (id 6), having
  // run frames 2-15, so the client resumes told request 5 completed: every
  // frame before the wrap is acknowledged, and frames 16-22 are sent again.
  for (int i = 0; i < kBatch; i++) {
    const int frame = 2 + i;
    EXPECT(total_mem_async(frame == 13 ? 99 : 0) == CUDA_SUCCESS,
           "a call without a reply was refused");
    sent_async++;
  }
  // Frame 22, id 12: flushes the batch and waits. It carries frame 13's
  // failure, once.
  count = 0;
  EXPECT(device_count(&count) == CUDA_ERROR_INVALID_DEVICE,
         "the call after the wrap and the break did not carry the failure of "
         "the call without a reply before it");

  // Calls go on working after the wrap, with nothing left over to report.
  for (int round = 0; round < 3; round++) {
    for (int i = 0; i < kBatch; i++) {
      EXPECT(total_mem_async(0) == CUDA_SUCCESS,
             "a call without a reply was refused after the wrap");
      sent_async++;
    }
    count = 0;
    EXPECT(device_count(&count) == CUDA_SUCCESS && count > 0,
           "a call after the wrap failed");
  }

  const long runs = totalmem_runs();
  std::printf("   %ld calls without a reply sent, %ld run\n", sent_async, runs);
  EXPECT(runs == sent_async,
         "calls without a reply across the wrap did not each run exactly once");

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: request ids wrap, and a replay across the wrap runs "
              "everything once\n");
  return 0;
}
