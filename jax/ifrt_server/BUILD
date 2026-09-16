# Copy this directory into an OpenXLA checkout at xla/python/ifrt_proxy/rgpu/
# and build //xla/python/ifrt_proxy/rgpu:rgpu_ifrt_server. See README.md.
#
# Unverified: these deps are read off the upstream headers and the
# scoped_pjrt_cpu_via_proxy example, not from a completed build. Target names
# move between XLA releases.

load("//xla:xla.default.bzl", "xla_cc_binary")

xla_cc_binary(
    name = "rgpu_ifrt_server",
    srcs = ["rgpu_ifrt_server.cc"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/pjrt/gpu:gpu_helpers",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        # Platform registration is an alwayslink static initialiser: without a
        # direct dependency it is not linked in, and the server starts fine and
        # then fails every session with
        #   Could not find registered platform with name: "cuda"
        "//xla/stream_executor/cuda:all_runtime",
        # Same alwayslink story: without this the GPU client allocates device
        # memory and then dies on
        #   Check failed: No collectives registered for platform: cuda
        "//xla/backends/gpu/collectives:gpu_collectives_plugin",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt_proxy/server:grpc_server",
        "//xla/python/pjrt_ifrt",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
