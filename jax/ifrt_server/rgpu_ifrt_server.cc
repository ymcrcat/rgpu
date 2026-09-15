/* A standalone IFRT proxy server, so a Mac running stock JAX can use a remote
   GPU through ordinary jax.Array placement.

   OpenXLA ships the server as a cc_library only - there is no cc_binary and no
   grpc_server_main anywhere in the repo - so this is the missing main(). It is
   modelled on xla/python/ifrt_proxy/integration_tests/scoped_pjrt_cpu_via_proxy.cc,
   which is the only in-tree example of standing the server up over a PJRT
   client.

   Build inside an OpenXLA checkout; see README.md next to this file.

     rgpu_ifrt_server --port=12345 --backend=gpu

   Then, on the client, through an ssh tunnel:

     IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true python your_script.py

   That variable is not optional and the value must be exactly lowercase
   "true": the client otherwise uses ALTS credentials and every connection is
   refused with "Invalid credentials". "1" and "TRUE" are silently ignored.

   There is no authentication of any kind here, exactly as with the rest of
   rgpu: bind to localhost and reach it over an ssh tunnel. Anyone who can
   connect can run arbitrary computations on the GPU and read its memory.
*/

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"

ABSL_FLAG(int, port, 12345, "port to listen on");
ABSL_FLAG(std::string, host, "127.0.0.1",
          "address to bind; localhost by default because there is no auth");
ABSL_FLAG(std::string, backend, "gpu", "gpu or cpu");

namespace {

// Called by the proxy server once per client session. Whatever IFRT client
// this returns is what the remote JAX process ends up driving.
absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> MakeBackend(
    xla::ifrt::AttributeMap /*initialization_data*/) {
  std::unique_ptr<xla::PjRtClient> pjrt;
  if (absl::GetFlag(FLAGS_backend) == "cpu") {
    xla::CpuClientOptions options;
    options.asynchronous = true;
    TF_ASSIGN_OR_RETURN(pjrt, xla::GetXlaPjrtCpuClient(options));
  } else {
    xla::GpuClientOptions options;
    TF_ASSIGN_OR_RETURN(pjrt, xla::GetStreamExecutorGpuClient(options));
  }
  LOG(INFO) << "backend has " << pjrt->device_count() << " device(s), platform "
            << pjrt->platform_name();
  return xla::ifrt::PjRtClient::Create(std::move(pjrt));
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string address =
      absl::StrCat(absl::GetFlag(FLAGS_host), ":", absl::GetFlag(FLAGS_port));

  auto server =
      xla::ifrt::proxy::GrpcServer::CreateFromIfrtClientFactory(address,
                                                                MakeBackend);
  if (!server.ok()) {
    LOG(ERROR) << "could not start the proxy server: " << server.status();
    return 1;
  }
  LOG(INFO) << "rgpu ifrt proxy server listening on " << (*server)->address()
            << " (no authentication; keep it behind an ssh tunnel)";
  (*server)->Wait();
  return 0;
}
