// Checks the CUDA graph calls survive the wire, without a GPU.
//
// This is the path torch.compile's reduce-overhead mode uses. The part worth
// testing is cudaStreamGetCaptureInfo: one of its outputs is a pointer to an
// array the driver owns, which cannot cross a wire as a pointer, so the server
// copies the contents and the client hands back a pointer to its own copy. A
// mistake there is a wild pointer rather than a wrong answer.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./graph_smoke

#include <cstdio>

#include <cuda_runtime_api.h>

static int g_failures = 0;

#define CHECK(call)                                                       \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,  \
                   #call, e_);                                            \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

int main() {
  cudaStream_t stream = nullptr;
  CHECK(cudaStreamCreate(&stream));

  cudaStreamCaptureStatus status = cudaStreamCaptureStatusActive;
  CHECK(cudaStreamIsCapturing(stream, &status));
  if (status != cudaStreamCaptureStatusNone) {
    std::fprintf(stderr, "FAIL: a fresh stream reports itself as capturing\n");
    g_failures++;
  }

  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  // The array of dependencies is the reason this call is hand-written.
  unsigned long long id = 0;
  cudaGraph_t capturing = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  size_t ndeps = 0;
  CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &id, &capturing, &deps,
                                    &ndeps));
  if (status != cudaStreamCaptureStatusActive) {
    std::fprintf(stderr, "FAIL: capture did not start\n");
    g_failures++;
  }
  if (ndeps != 2 || !deps) {
    std::fprintf(stderr, "FAIL: expected 2 dependencies, got %zu\n", ndeps);
    g_failures++;
  } else if (deps[0] == deps[1]) {
    // The values come from the server's array; two distinct nodes arriving as
    // one would mean the copy collapsed them.
    std::fprintf(stderr, "FAIL: dependency nodes are not distinct\n");
    g_failures++;
  }

  cudaGraph_t graph = nullptr;
  CHECK(cudaStreamEndCapture(stream, &graph));
  if (!graph) {
    std::fprintf(stderr, "FAIL: capture produced no graph\n");
    g_failures++;
  }

  // Instantiate and launch. The fake driver rejects a handle it did not hand
  // out, so these passing means the handles came back unchanged.
  cudaGraphExec_t exec = nullptr;
  CHECK(cudaGraphInstantiate(&exec, graph, 0));
  CHECK(cudaGraphLaunch(exec, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(cudaGraphExecDestroy(exec));
  CHECK(cudaGraphDestroy(graph));

  // Ending a capture that never began must fail, or the checks above would
  // pass against a server that agreed with everything.
  if (cudaStreamEndCapture(stream, &graph) == cudaSuccess) {
    std::fprintf(stderr, "FAIL: ending a capture twice was accepted\n");
    g_failures++;
  }

  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: CUDA graph capture survives the wire\n");
  return 0;
}
