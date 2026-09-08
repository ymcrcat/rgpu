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
  HINTS /usr/local/cuda/include /usr/include
  REQUIRED)
include_directories(${CUDA_INCLUDE_DIR})

set(GEN_CLIENT ${CMAKE_SOURCE_DIR}/client/generated/client_stubs.cpp)
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
# Server: runs on the GPU host against the real driver.
#
# Links the toolkit's libcuda stub; the real driver is resolved at run time.
# ---------------------------------------------------------------------------
find_library(CUDA_DRIVER_LIB cuda
  HINTS /usr/local/cuda/lib64/stubs /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

if(CUDA_DRIVER_LIB)
  add_executable(rgpu_server
    server/main.cpp
    ${GEN_SERVER}
    ${GEN_NAMES}
  )
  # Generated code forwards deprecated entry points on purpose: an application
  # that calls them should still reach the real driver.
  set_source_files_properties(${GEN_SERVER} PROPERTIES
    COMPILE_OPTIONS "-Wno-deprecated-declarations")
  set_target_properties(rgpu_server PROPERTIES OUTPUT_NAME rgpu-server)
  target_link_libraries(rgpu_server PRIVATE ${CUDA_DRIVER_LIB} Threads::Threads)
else()
  message(STATUS "libcuda not found; skipping server target (client-only build)")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(test_wire tests/test_wire.cpp)
add_test(NAME wire COMMAND test_wire)

# The GPU-free end-to-end test: client stubs and server dispatch over a real
# socket, with a fake driver underneath.
add_test(NAME remoting
  COMMAND ${CMAKE_SOURCE_DIR}/tests/run_smoke.sh ${CMAKE_BINARY_DIR})

# The remoting smoke test links the shim, so its CUDA calls go over the wire.
add_executable(rpc_smoke tests/rpc_smoke.cpp)
target_link_libraries(rpc_smoke PRIVATE rgpu_client)

# A server backed by a fake driver instead of a GPU. Lets the whole stack run
# on a machine with no NVIDIA hardware: real client stubs, real wire format,
# real server dispatch, fake bottom layer.
add_executable(rgpu_server_fake
  server/main.cpp
  ${GEN_SERVER}
  ${GEN_NAMES}
  tests/generated/fake_driver.cpp
  tests/fake_cuda.cpp
)
set_target_properties(rgpu_server_fake PROPERTIES OUTPUT_NAME rgpu-server-fake)
target_link_libraries(rgpu_server_fake PRIVATE Threads::Threads)
target_compile_options(rgpu_server_fake PRIVATE -Wno-deprecated-declarations)

# Driver-API vector add. On the GPU host it links the real driver; on a client
# with no driver it links the shim, which carries the same SONAME. Either way
# the loader resolves libcuda.so.1 at run time.
add_executable(vecadd tests/cuda/vecadd.cpp)
if(CUDA_DRIVER_LIB)
  target_link_libraries(vecadd PRIVATE ${CUDA_DRIVER_LIB})
else()
  target_link_libraries(vecadd PRIVATE rgpu_client)
endif()
