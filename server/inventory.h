// What a session has taken from the driver and not given back.
//
// The resources a client asks for are created in this process and belong to
// it: a device allocation, a retained primary context, a loaded module, a
// cuBLAS handle. A client that exits or crashes without freeing them does not
// take them with it, and the server outlives client after client, so without a
// record of who asked for what the only thing that ever reclaims a dead
// client's GPU memory is restarting the server.
//
// So each session keeps a list. The recording happens underneath the generated
// dispatch, in wrappers that driver_sym() hands back in place of the driver's
// own entry points, which is why nothing in the generated code knows about
// this. The maths libraries are hand-written and call the note/forget
// functions below directly.
//
// One thread serves one session, and that same thread is the one that releases
// the inventory when the session finally expires, so in practice the inventory
// is touched by a single thread. It is locked anyway: the cost next to a
// driver call is nothing, and the failure it would prevent is a corrupted map
// on a path that only runs when something has already gone wrong.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cuda.h>

namespace rgpu {

struct Inventory {
  // Which context was current when this was created. Releasing it has to
  // happen under that context, and destroying the context makes it moot.
  struct Item {
    CUcontext ctx = nullptr;
  };
  // A maths-library handle. The library that minted it is the only thing that
  // knows how to destroy it, so it leaves a way to do that behind.
  struct LibHandle {
    CUcontext ctx = nullptr;
    const char* what = "";
    CUresult (*destroy)(uint64_t) = nullptr;
  };
  using Items = std::unordered_map<uint64_t, Item>;

  std::mutex mu;
  Items allocs, contexts, modules, streams, events, graphs, graph_execs;
  std::unordered_map<uint64_t, LibHandle> handles;
  // Per device, how many retains this session holds and has not released.
  // A primary context is shared, so this count, and not the handle, is what
  // the session owns: exactly this many releases are owed at the end and not
  // one more, or a session still using the device loses it.
  std::unordered_map<int, int> primary_retains;
  // The primary context handle per device, learned when it was retained. Only
  // used to find what a reset of that device, or the last release of its
  // primary context, threw away.
  std::unordered_map<int, CUcontext> primary_ctx;
};

// Binds this thread to the session it is serving, so the recording underneath
// knows whose resource it is looking at. Passing nullptr unbinds. Binding and
// unbinding is also how the server counts how many sessions are live, which is
// what makes it possible to refuse a call that would reach into all of them.
void inventory_bind(Inventory* inv);

// Records a maths-library handle against the session being served here, and
// forgets one the client destroyed itself. `what` is used in the log and must
// outlive the session, so a string literal.
void inventory_note_handle(uint64_t handle, const char* what,
                           CUresult (*destroy)(uint64_t));
void inventory_forget_handle(uint64_t handle);

// The entry point the server should call for `name`: a wrapper that records
// what the driver hands out, or nullptr if this call creates nothing worth
// remembering. See driver_sym() in server/driver_syms.h.
void* driver_wrapper(const char* name);

// Gives back everything the session still holds, in an order that is safe:
// what lives inside a context before the context itself, and the shared
// primary context last of all. Every failure is logged and none of them stops
// the rest. Returns a description of what happened, for the caller's log.
std::string release_inventory(Inventory& inv);

}  // namespace rgpu
