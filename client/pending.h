// Which streams have GPU work outstanding that a synchronize would wait for.
//
// A stream synchronize is a round trip, and a program that synchronizes after
// every small upload pays one per upload: llama.cpp does so about twenty
// times per generated token. Measured there, every one of those synchronizes
// followed nothing but uploads. An upload's bytes are taken from the caller
// when it is issued, so there is nothing for such a synchronize to wait for
// that the caller could observe: it can be queued behind the upload and return
// at once. A synchronize after anything else - a kernel, a memset, a matrix
// multiply - still waits.
//
// This is the record that decides which is which. It errs toward waiting:
// work whose stream this library cannot name counts against every stream, and
// only a wait for the whole context clears that.
#pragma once

#include <cstdint>

namespace rgpu {

// GPU work a synchronize of `stream` would wait for has been issued.
void work_on(uint64_t stream);
// The same, on a stream that cannot be named here. Counts against all of them.
void work_somewhere();
// A wait for `stream` has returned, or the stream is gone.
void settled(uint64_t stream);
// A wait for everything in the context has returned.
void settled_all();
// Whether a synchronize of `stream` would have nothing to wait for but
// uploads. The default stream orders itself against every other, so for it
// that means nothing outstanding anywhere.
bool nothing_to_wait_for(uint64_t stream);

}  // namespace rgpu
