// Sends the server requests a well-behaved client never would, without a GPU.
//
// The server trusts nothing it reads off the socket, or should not. Each case
// here is a request whose fields disagree with each other in a way that used
// to reach the driver: a buffer sized by one field while the driver is told
// another, a capacity large enough to exhaust memory. The server must answer
// each with an error and carry on serving - checked after every case by
// making a normal call that has to still work.
//
//   LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./hostile_smoke

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda.h>

#include "client/rpc.h"
#include "common/generated/api_ids.h"
#include "common/internal_ids.h"
#include "common/net.h"

static int g_failures = 0;

#define EXPECT(cond, what)                                             \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what); \
      g_failures++;                                                    \
    }                                                                  \
  } while (0)

namespace {

CUdeviceptr g_dev = 0;
constexpr size_t kAlloc = 4096;

// A normal round trip through the same allocation. If the server fell over,
// or its heap was trampled, this is where it shows.
void still_serving(const char* after) {
  std::vector<unsigned char> out(kAlloc, 0xAA), back(kAlloc, 0);
  bool ok = cuMemcpyHtoD(g_dev, out.data(), kAlloc) == CUDA_SUCCESS &&
            cuMemcpyDtoH(back.data(), g_dev, kAlloc) == CUDA_SUCCESS &&
            std::memcmp(out.data(), back.data(), kAlloc) == 0;
  if (!ok) {
    std::fprintf(stderr, "FAIL: the server stopped serving after %s\n", after);
    g_failures++;
  }
}

// A connection of our own to the server the shim is talking to, bypassing the
// shim, so the handshake can say whatever a test needs it to.
int dial() {
  const char* env = std::getenv("RGPU_SERVER");
  std::string spec = env ? env : "127.0.0.1:9713";
  const size_t c = spec.rfind(':');
  const std::string host = spec.substr(0, c), port = spec.substr(c + 1);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (::getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return -1;
  int fd = -1;
  for (addrinfo* ai = res; ai; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd >= 0) {
    // A server that neither answers nor closes would hang the test otherwise.
    timeval tv{5, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  return fd;
}

CUresult raw(uint32_t id, rgpu::Buffer& req) {
  rgpu::Buffer rsp;
  return rgpu::call(id, req, &rsp);
}

}  // namespace

int main() {
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS ||
      cuMemAlloc(&g_dev, kAlloc) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: could not set up\n");
    return 1;
  }
  still_serving("setup");

  // A client from before protocol 3, whose request headers are four bytes
  // shorter. The handshake has not changed size, so the server can read its
  // version - and must refuse it rather than go on to half-parse its frames.
  // It must also say which protocol it does speak before closing: an old
  // client reads the reply's version and names the mismatch, where a server
  // that just closes leaves it reporting only that the handshake failed.
  {
    int fd = dial();
    EXPECT(fd >= 0, "could not open a raw connection to the server");
    if (fd >= 0) {
      rgpu::Handshake hello{};
      hello.magic = rgpu::kMagicHello;
      hello.version = 2;
      hello.session_hi = 0x01d0c11e47ull;
      hello.session_lo = 0x2ull;
      rgpu::HandshakeReply reply{};
      const bool answered = rgpu::write_exact(fd, &hello, sizeof(hello)) &&
                            rgpu::read_exact(fd, &reply, sizeof(reply));
      EXPECT(answered, "a protocol 2 handshake was closed without a reply "
                       "naming the server's protocol");
      if (answered) {
        EXPECT(reply.magic == rgpu::kMagicHello,
               "the refusal did not carry the handshake magic");
        EXPECT(reply.version == rgpu::kProtocolVersion,
               "the refusal did not carry the server's protocol version");
        EXPECT(reply.resumed == 0,
               "the refusal claimed to resume a session");
        char byte = 0;
        EXPECT(::recv(fd, &byte, 1, 0) == 0,
               "the server kept a protocol 2 connection open after refusing it");
      }
      ::close(fd);
    }
    still_serving("a protocol 2 handshake");
  }

  // Device to host: the reply buffer is sized by the client's length field,
  // the driver by ByteCount. A client claiming 16 bytes and asking for 4096
  // had the driver write 4096 bytes into a 16-byte buffer on the server.
  {
    rgpu::Buffer req;
    req.put<uint8_t>(1);
    req.put<uint64_t>(16);
    req.put<CUdeviceptr>(g_dev);
    req.put<size_t>(kAlloc);
    EXPECT(raw(rgpu::API_cuMemcpyDtoH_v2, req) == CUDA_ERROR_INVALID_VALUE,
           "a device-to-host copy larger than its buffer was accepted");
    still_serving("an oversized device-to-host copy");
  }

  // Host to device: 16 bytes sent, the driver told to read 4096 of them,
  // reading past the end of the request on the server.
  {
    unsigned char sixteen[16] = {0};
    rgpu::Buffer req;
    req.put<CUdeviceptr>(g_dev);
    req.put<uint8_t>(1);
    req.put_sized(sixteen, sizeof(sixteen));
    req.put<size_t>(kAlloc);
    EXPECT(raw(rgpu::API_cuMemcpyHtoD_v2, req) == CUDA_ERROR_INVALID_VALUE,
           "a host-to-device copy longer than the bytes sent was accepted");
    still_serving("an overlong host-to-device copy");
  }

  // A graph-node capacity large enough that allocating it throws, which
  // used to take down every session on the server, not just this one.
  {
    rgpu::Buffer req;
    req.put<uint64_t>(0x1234);         // any graph
    req.put<uint8_t>(1);               // wants the nodes
    req.put<uint64_t>(1ull << 60);     // capacity
    EXPECT(raw(rgpu::API_rgpu_graph_nodes, req) == CUDA_ERROR_INVALID_VALUE,
           "an absurd graph-node capacity was accepted");
    still_serving("an absurd graph-node capacity");
  }

  // Consistent but enormous: the sizes agree, so only the allocation can
  // fail. That has to be an error for this call, not the end of the server.
  {
    rgpu::Buffer req;
    req.put<uint8_t>(1);
    req.put<uint64_t>(1ull << 50);
    req.put<CUdeviceptr>(g_dev);
    req.put<size_t>(1ull << 50);
    EXPECT(raw(rgpu::API_cuMemcpyDtoH_v2, req) != CUDA_SUCCESS,
           "a petabyte copy reported success");
    still_serving("a petabyte copy");
  }

  cuMemFree(g_dev);
  cuCtxDestroy(ctx);
  if (g_failures) {
    std::printf("\nFAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("\nPASS: the server refuses hostile requests and keeps serving\n");
  return 0;
}
