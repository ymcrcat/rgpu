cmake_minimum_required(VERSION 3.16)
project(rgpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

include_directories(${CMAKE_SOURCE_DIR})

# CUDA headers. We need cuda.h on both sides; only the server needs to link
# the driver library itself.
find_path(CUDA_INCLUDE_DIR cuda.h
  HINTS ${CMAKE_SOURCE_DIR}/third_party/cuda_include
        /usr/local/cuda/include /usr/include
  REQUIRED)
include_directories(${CUDA_INCLUDE_DIR})

set(GEN_CLIENT ${CMAKE_SOURCE_DIR}/client/generated/client_stubs.cpp)
set(GEN_CUDART ${CMAKE_SOURCE_DIR}/client/generated/cudart_stubs.cpp)
set(GEN_SERVER ${CMAKE_SOURCE_DIR}/server/generated/dispatch.cpp)
set(GEN_NAMES  ${CMAKE_SOURCE_DIR}/common/generated/api_names.cpp)

if(NOT EXISTS ${GEN_CLIENT})
  message(FATAL_ERROR "generated sources missing; run ./codegen/run.sh first")
endif()

# ---------------------------------------------------------------------------
# Client: a drop-in replacement for libcuda.so.1.
#
# Deliberately does not link the real driver library. On the client there is
# none, and this library is what stands in for it.
# ---------------------------------------------------------------------------
add_library(rgpu_client SHARED
  client/shim.cpp
  client/rpc.cpp
  ${GEN_CLIENT}
  ${GEN_NAMES}
)
set_target_properties(rgpu_client PROPERTIES
  OUTPUT_NAME cuda
  SOVERSION 1
)
set_source_files_properties(${GEN_CLIENT} PROPERTIES
  COMPILE_OPTIONS "-Wno-deprecated-declarations")
find_package(Threads REQUIRED)
target_link_libraries(rgpu_client PRIVATE Threads::Threads)

# ---------------------------------------------------------------------------
# Client: a drop-in replacement for libcudart.so.12.
#
# Stock libcudart cannot run on a remoted driver, because it demands
# undocumented export tables that cannot cross a process boundary. So the
# runtime API stops here and is translated into driver calls, which the shim
# above remotes.
# ---------------------------------------------------------------------------
add_library(rgpu_cudart SHARED
  client/cudart_impl.cpp
  ${GEN_CUDART}
)
set_target_properties(rgpu_cudart PROPERTIES
  OUTPUT_NAME cudart
  SOVERSION 12
)
target_link_libraries(rgpu_cudart PRIVATE rgpu_client Threads::Threads)
# Anything built against the real libcudart references its symbols through a
# version node named after the library, so ours has to define the same node.
target_link_options(rgpu_cudart PRIVATE
  "-Wl,--version-script=${CMAKE_SOURCE_DIR}/client/cudart.map")

# ---------------------------------------------------------------------------
# Client: a stand-in for libcublas.so.12.
#
# cuBLAS cannot run on the client either: like the stock runtime, it calls
# cuGetExportTable while initialising and fails without it. For now this is a
# discovery build, every entry point announcing itself by name, so a real run
# reports exactly which functions need forwarding.
# ---------------------------------------------------------------------------
set(CUBLAS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cublas_include)
if(EXISTS ${CUBLAS_INCLUDE_DIR}/cublas_v2.h AND
   EXISTS ${CMAKE_SOURCE_DIR}/client/generated/cublas_stubs.cpp)
  add_library(rgpu_cublas SHARED
    client/cublas_impl.cpp
    client/generated/cublas_stubs.cpp)
  set_target_properties(rgpu_cublas PROPERTIES OUTPUT_NAME cublas SOVERSION 12)
  target_include_directories(rgpu_cublas PRIVATE ${CUBLAS_INCLUDE_DIR})
  target_link_libraries(rgpu_cublas PRIVATE rgpu_client)
  if(EXISTS ${CMAKE_SOURCE_DIR}/client/cublas.map)
    target_link_options(rgpu_cublas PRIVATE
      "-Wl,--version-script=${CMAKE_SOURCE_DIR}/client/cublas.map")
  endif()
endif()

# ---------------------------------------------------------------------------
# Client: a stand-in for libcublasLt.so.12.
#
# Nothing is forwarded yet. The point is that the real cuBLASLt segfaults when
# it runs on the client over a remoted driver, which is a confusing way to
# fail; every entry point here reports itself and returns an error instead.
# PyTorch's addmm uses cuBLASLt by default, so set DISABLE_ADDMM_CUDA_LT=1 to
# route it through plain cuBLAS, which is forwarded.
# ---------------------------------------------------------------------------
if(EXISTS ${CUBLAS_INCLUDE_DIR}/cublasLt.h AND
   EXISTS ${CMAKE_SOURCE_DIR}/client/generated/cublaslt_stubs.cpp)
  add_library(rgpu_cublaslt SHARED
    client/cublaslt_impl.cpp
    client/generated/cublaslt_stubs.cpp)
  set_target_properties(rgpu_cublaslt PROPERTIES
    OUTPUT_NAME cublasLt SOVERSION 12)
  target_include_directories(rgpu_cublaslt PRIVATE ${CUBLAS_INCLUDE_DIR})
  target_link_libraries(rgpu_cublaslt PRIVATE rgpu_client)
  if(EXISTS ${CMAKE_SOURCE_DIR}/client/cublaslt.map)
    target_link_options(rgpu_cublaslt PRIVATE
      "-Wl,--version-script=${CMAKE_SOURCE_DIR}/client/cublaslt.map")
  endif()
endif()

# ---------------------------------------------------------------------------
# Client: a stand-in for libcudnn.so.9.
#
# PyTorch reaches cuDNN through its backend graph API, so the nine functions
# hand-written in client/cudnn_impl.cpp carry the whole surface. The generated
# stubs exist so the symbol set is complete and anything else names itself.
# ---------------------------------------------------------------------------
set(CUDNN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cudnn_include)
if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn.h AND
   EXISTS ${CMAKE_SOURCE_DIR}/client/generated/cudnn_stubs.cpp)
  add_library(rgpu_cudnn SHARED
    client/cudnn_impl.cpp
    client/generated/cudnn_stubs.cpp)
  set_target_properties(rgpu_cudnn PROPERTIES OUTPUT_NAME cudnn SOVERSION 9)
  target_include_directories(rgpu_cudnn PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(rgpu_cudnn PRIVATE rgpu_client)
  if(EXISTS ${CMAKE_SOURCE_DIR}/client/cudnn.map)
    target_link_options(rgpu_cudnn PRIVATE
      "-Wl,--version-script=${CMAKE_SOURCE_DIR}/client/cudnn.map")
  endif()
endif()

# ---------------------------------------------------------------------------
# Server: runs on the GPU host against the real driver.
#
# Links the toolkit's libcuda stub; the real driver is resolved at run time.
# ---------------------------------------------------------------------------
# The driver library itself where the host has one, otherwise the toolkit's
# stub. Generated dispatch resolves entry points at run time, so the link only
# has to satisfy the handful the server calls directly.
find_library(CUDA_DRIVER_LIB NAMES cuda libcuda.so.1
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64/stubs /usr/local/cuda/lib64)

if(CUDA_DRIVER_LIB)
  set(SERVER_SOURCES server/main.cpp server/inventory.cpp ${GEN_SERVER}
      ${GEN_NAMES})
  if(EXISTS ${CUBLAS_INCLUDE_DIR}/cublas_v2.h)
    list(APPEND SERVER_SOURCES server/cublas_server.cpp)
  endif()
  if(EXISTS ${CUBLAS_INCLUDE_DIR}/cublasLt.h)
    list(APPEND SERVER_SOURCES server/cublaslt_server.cpp)
  endif()
  if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn.h)
    list(APPEND SERVER_SOURCES server/cudnn_server.cpp)
  endif()
  add_executable(rgpu_server ${SERVER_SOURCES})
  if(EXISTS ${CUBLAS_INCLUDE_DIR}/cublas_v2.h)
    target_include_directories(rgpu_server PRIVATE ${CUBLAS_INCLUDE_DIR})
  endif()
  if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn.h)
    target_include_directories(rgpu_server PRIVATE ${CUDNN_INCLUDE_DIR})
  endif()
  # Generated code forwards deprecated entry points on purpose: an application
  # that calls them should still reach the real driver.
  set_source_files_properties(${GEN_SERVER} PROPERTIES
    COMPILE_OPTIONS "-Wno-deprecated-declarations")
  set_target_properties(rgpu_server PROPERTIES OUTPUT_NAME rgpu-server
                        ENABLE_EXPORTS ON)
  target_link_libraries(rgpu_server PRIVATE ${CUDA_DRIVER_LIB} ${CMAKE_DL_LIBS}
                        Threads::Threads)
else()
  message(STATUS "libcuda not found; skipping server target (client-only build)")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(test_wire tests/test_wire.cpp)
add_test(NAME wire COMMAND test_wire)

# The fake driver on its own, driven from several OS threads on two devices.
# Everything else reaches the fake through the server, where one thread serves
# a session, so this is the only test that can see whether the fake keeps
# contexts apart the way a real driver does.
add_executable(test_fake_cuda tests/test_fake_cuda.cpp tests/fake_cuda.cpp
  tests/fake_stats.cpp tests/generated/fake_driver.cpp)
target_link_libraries(test_fake_cuda PRIVATE Threads::Threads)
target_compile_options(test_fake_cuda PRIVATE -Wno-deprecated-declarations)
add_test(NAME fake_cuda COMMAND test_fake_cuda)
set_tests_properties(fake_cuda PROPERTIES ENVIRONMENT RGPU_FAKE_DEVICES=2)

# The GPU-free end-to-end test: client stubs and server dispatch over a real
# socket, with a fake driver underneath.
add_test(NAME remoting
  COMMAND ${CMAKE_SOURCE_DIR}/tests/run_smoke.sh ${CMAKE_BINARY_DIR})

# The remoting smoke test links the shim, so its CUDA calls go over the wire.
add_executable(rpc_smoke tests/rpc_smoke.cpp)
target_link_libraries(rpc_smoke PRIVATE rgpu_client)

# Checks the kernel launch path: parameter layout lookup, argument packing and
# the extra[] handoff.
add_executable(launch_smoke tests/launch_smoke.cpp)
target_link_libraries(launch_smoke PRIVATE rgpu_client)
# It calls the deprecated multi-device cooperative launch on purpose, to check
# that the shim refuses it.
target_compile_options(launch_smoke PRIVATE -Wno-deprecated-declarations)

# Checks that cuBLAS and cuBLASLt arguments survive the wire. Needs both shims
# and the fake libraries in the test server.
if(TARGET rgpu_cublas AND TARGET rgpu_cublaslt)
  add_executable(cublas_smoke tests/cublas_smoke.cpp)
  target_include_directories(cublas_smoke PRIVATE ${CUBLAS_INCLUDE_DIR})
  target_link_libraries(cublas_smoke PRIVATE rgpu_cublas rgpu_cublaslt
                        rgpu_client)
endif()

# The fake maths libraries, when their headers are available.
set(FAKE_CUBLAS_SOURCES "")
if(EXISTS ${CUBLAS_INCLUDE_DIR}/cublas_v2.h)
  list(APPEND FAKE_CUBLAS_SOURCES tests/fake_cublas.cpp)
endif()
if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn.h)
  list(APPEND FAKE_CUBLAS_SOURCES tests/fake_cudnn.cpp)
endif()

# Measures the per-call cost of the remoting layer.
add_executable(bench tests/bench.cpp)
target_link_libraries(bench PRIVATE rgpu_client)

# Checks that cuDNN backend attributes survive the wire.
if(TARGET rgpu_cudnn)
  add_executable(cudnn_smoke tests/cudnn_smoke.cpp)
  target_include_directories(cudnn_smoke PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(cudnn_smoke PRIVATE rgpu_cudnn rgpu_client)
endif()

# Sends the server requests a well-behaved client never would.
add_executable(hostile_smoke tests/hostile_smoke.cpp)
target_link_libraries(hostile_smoke PRIVATE rgpu_client)

# Checks that every request names the client thread that issued it, against a
# scripted server of its own.
add_executable(thread_id_smoke tests/thread_id_smoke.cpp)
target_link_libraries(thread_id_smoke PRIVATE rgpu_client Threads::Threads)
add_test(NAME thread_id COMMAND thread_id_smoke wire)

# Checks that a dropped connection does not lose the session.
add_executable(reconnect_smoke tests/reconnect_smoke.cpp)
target_link_libraries(reconnect_smoke PRIVATE rgpu_client)

# Checks that a session which never comes back gives its GPU resources up, and
# that a session which is still there keeps its own. Re-executes itself to get
# a second client process, so it needs nothing from the shell but a server.
add_executable(expiry_smoke tests/expiry_smoke.cpp)
target_link_libraries(expiry_smoke PRIVATE rgpu_client)
if(TARGET rgpu_cublas)
  target_link_libraries(expiry_smoke PRIVATE rgpu_cublas)
  target_include_directories(expiry_smoke PRIVATE ${CUBLAS_INCLUDE_DIR})
  target_compile_definitions(expiry_smoke PRIVATE RGPU_EXPIRY_CUBLAS)
endif()
if(TARGET rgpu_cublaslt)
  target_link_libraries(expiry_smoke PRIVATE rgpu_cublaslt)
  target_include_directories(expiry_smoke PRIVATE ${CUBLAS_INCLUDE_DIR})
  target_compile_definitions(expiry_smoke PRIVATE RGPU_EXPIRY_CUBLASLT)
endif()
if(TARGET rgpu_cudnn)
  target_link_libraries(expiry_smoke PRIVATE rgpu_cudnn)
  target_include_directories(expiry_smoke PRIVATE ${CUDNN_INCLUDE_DIR})
  target_compile_definitions(expiry_smoke PRIVATE RGPU_EXPIRY_CUDNN)
endif()

# Checks the CUDA graph path, which is what torch.compile's reduce-overhead
# mode uses. Linked against our runtime shim, since that is where the
# translation lives.
add_executable(graph_smoke tests/graph_smoke.cpp)
target_link_libraries(graph_smoke PRIVATE rgpu_cudart rgpu_client)

# A server backed by a fake driver instead of a GPU. Lets the whole stack run
# on a machine with no NVIDIA hardware: real client stubs, real wire format,
# real server dispatch, fake bottom layer.
add_executable(rgpu_server_fake
  server/main.cpp
  server/inventory.cpp
  ${GEN_SERVER}
  ${GEN_NAMES}
  tests/generated/fake_driver.cpp
  tests/fake_cuda.cpp
  tests/fake_stats.cpp
  ${FAKE_CUBLAS_SOURCES}
)
if(FAKE_CUBLAS_SOURCES)
  target_sources(rgpu_server_fake PRIVATE
    server/cublas_server.cpp server/cublaslt_server.cpp)
  target_include_directories(rgpu_server_fake PRIVATE ${CUBLAS_INCLUDE_DIR})
endif()
if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn.h)
  target_sources(rgpu_server_fake PRIVATE server/cudnn_server.cpp)
  target_include_directories(rgpu_server_fake PRIVATE ${CUDNN_INCLUDE_DIR})
endif()
set_target_properties(rgpu_server_fake PROPERTIES
  OUTPUT_NAME rgpu-server-fake
  # The fake driver is compiled into this binary, so its symbols must be in the
  # dynamic table for dlsym(RTLD_DEFAULT) to find them.
  ENABLE_EXPORTS ON)
target_link_libraries(rgpu_server_fake PRIVATE Threads::Threads)
target_compile_options(rgpu_server_fake PRIVATE -Wno-deprecated-declarations)
target_link_libraries(rgpu_server_fake PRIVATE ${CMAKE_DL_LIBS})

# The phase-2 test: stock libcudart on top of the shim. Built only when the
# runtime has been fetched, since it is not part of a plain client build.
find_library(CUDART_LIB cudart
  HINTS ${CMAKE_SOURCE_DIR}/third_party/cudart /usr/local/cuda/lib64
  NO_DEFAULT_PATH)
if(CUDART_LIB)
  add_executable(cudart_smoke tests/cudart_smoke.cpp)
  target_link_libraries(cudart_smoke PRIVATE ${CUDART_LIB})
  # cudart dlopens nothing here; it resolves libcuda.so.1 through the loader,
  # which LD_LIBRARY_PATH points at our shim.
else()
  message(STATUS "libcudart not found; run scripts/fetch_cudart.sh to enable "
                 "the runtime API test")
endif()

# Driver-API vector add. On the GPU host it links the real driver; on a client
# with no driver it links the shim, which carries the same SONAME. Either way
# the loader resolves libcuda.so.1 at run time.
add_executable(vecadd tests/cuda/vecadd.cpp)
if(CUDA_DRIVER_LIB)
  target_link_libraries(vecadd PRIVATE ${CUDA_DRIVER_LIB})
else()
  target_link_libraries(vecadd PRIVATE rgpu_client)
endif()
