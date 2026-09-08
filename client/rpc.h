// Client-side RPC entry points used by the generated stubs.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>

#include "common/wire.h"

namespace rgpu {

// Sends one call and waits for its reply. Returns the remote CUresult, or a
// local error if the connection failed.
CUresult call(uint32_t api_id, const Buffer& req, Buffer* rsp);

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
size_t pointer_attr_size(unsigned int attribute);

// Same idea as unimplemented(), for CUDA runtime entry points we have not
// translated to driver calls yet.
void unimplemented_rt(const char* name);

void log(const char* fmt, ...);

}  // namespace rgpu
