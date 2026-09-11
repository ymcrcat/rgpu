// Client-side RPC entry points used by the generated stubs.
#pragma once

#include "common/sizes.h"

#include <cstddef>
#include <cstdint>

#include <cuda.h>

#include "common/wire.h"

namespace rgpu {

// Sends one call and waits for its reply. Anything queued by call_async is
// sent first, so ordering is preserved. Returns the remote CUresult, or a
// local error if the connection failed.
CUresult call(uint32_t api_id, const Buffer& req, Buffer* rsp);

// Queues a call whose effect is only observable at a later synchronization
// point, and returns without waiting. This is what removes a round trip from
// every kernel launch and every asynchronous copy.
//
// The reply is not merely ignored, it is never sent: the frame carries
// kFlagNoReply. A failure therefore cannot be reported here, so the server
// holds it and returns it from the next call that does reply, which is how
// CUDA reports asynchronous failures anyway.
//
// Only for calls with no output parameters. The generator refuses to use it
// for anything that has to return data.
CUresult call_async(uint32_t api_id, const Buffer& req);

// Logs the first occurrence of an unimplemented entry point and returns
// CUDA_ERROR_NOT_SUPPORTED. The log is the worklist for filling the API out.
CUresult unimplemented(const char* name, const char* why);

// Length of a module image, which the CUDA API passes without one. Handles a
// fatbin (magic 0xBA55ED50), a raw cubin ELF, and NUL-terminated PTX.
// Returns 0 if the image is not recognized.
size_t image_size(const void* image);

// Size of the value cuPointerGetAttribute writes for a given attribute. The
// API takes a bare void* whose meaning depends on the attribute, so the
// generated stub asks this how many bytes to expect. Returns 0 for an
// attribute we do not know, which the caller reports as invalid.

// Same idea as unimplemented(), for CUDA runtime entry points we have not
// translated to driver calls yet.
void unimplemented_rt(const char* name);

void log(const char* fmt, ...);

}  // namespace rgpu
