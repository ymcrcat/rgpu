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

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda.h>

namespace rgpu {

// Which context was current when a resource was made. Releasing it has to
// happen under that context, and destroying the context makes it moot.
//
// If that was a primary context, also its device and the generation of that
// device's primary context at the time. A primary context is destroyed by
// whichever session makes the last release in the process, or by a reset, and
// every session that still lists something made in it has to stop treating it
// as its own - including sessions that had nothing to do with the release. The
// generation is how they find out: see release_inventory.
//
// It is both what an inventory entry records (Inventory::Item) and what
// inventory_stamp() takes just before a resource is made.
struct InventoryStamp {
  CUcontext ctx = nullptr;
  int dev = -1;
  uint64_t gen = 0;
};

struct Inventory {
  using Item = InventoryStamp;
  // A maths-library handle. The library that minted it is the only thing that
  // knows how to destroy it, so it leaves a way to do that behind. Where it
  // was made is kept the same way as any other entry.
  struct LibHandle {
    Item made;
    const char* what = "";
    CUresult (*destroy)(uint64_t) = nullptr;
  };
  using Items = std::unordered_map<uint64_t, Item>;

  // The context-bound resource maps, in a fixed order. A context teardown
  // (forget_under / forget_primary) once walked all of these plus the handles
  // on every cuCtxDestroy / cuCtxDetach / primary reset, which is O(everything
  // the session owns). The reverse index below turns that into O(what the torn
  // down context owns): CtxMap names each swept map so the index can point at
  // an entry's map without a member pointer.
  enum CtxMap { kAllocs, kModules, kStreams, kEvents, kGraphs, kGraphExecs,
                kCtxMapCount };

  // A stream capture this session began and has not ended: the stream - 0
  // for the default stream of the context in `where` - and where the capture
  // is, which is where the stream is. Not a handle, but a capture left open is
  // held all the same: it keeps its stream capturing, and while one begun
  // other than RELAXED is open, the thread that began it - the serving thread
  // - is prohibited from potentially unsafe calls, freeing memory among them.
  // So expiry ends it, first of all, and releases the graph it produces.
  struct OpenCapture {
    uint64_t stream = 0;
    Item where;
  };

  std::mutex mu;
  Items allocs, contexts, modules, streams, events, graphs, graph_execs;
  // Loaded libraries (cuLibraryLoadData / cuLibraryLoadFromFile). Unlike a
  // module, a CUlibrary in CUDA 12 is context-independent - bound to no context
  // and outliving any context's destruction - so its entries carry no context,
  // device or generation (Item's defaults), a context teardown never forgets
  // them, and expiry unloads each unconditionally with cuLibraryUnload.
  Items libraries;
  std::vector<OpenCapture> captures;
  std::unordered_map<uint64_t, LibHandle> handles;

  // Reverse index from a context to the entries recorded against it, so tearing
  // a context down touches only its own entries instead of scanning every map.
  // For one context it holds, per swept map (indexed by CtxMap), the handles in
  // that map whose stamp names this context, plus the maths-library handles the
  // same. It is kept in step with the six maps and `handles` by note()/forget()
  // and the note/forget-handle entry points; an entry with a null context - a
  // context-independent one such as a loaded library, or one made with nothing
  // current - is never indexed and never swept, exactly as before. The captures
  // vector is left out: it holds only the handful a session leaves open, so a
  // teardown scans it directly.
  struct CtxEntries {
    std::array<std::unordered_set<uint64_t>, kCtxMapCount> in;
    std::unordered_set<uint64_t> handles;
  };
  std::unordered_map<CUcontext, CtxEntries> by_ctx;
  // Per device, how many retains this session holds and has not released.
  // A primary context is shared, so this count, and not the handle, is what
  // the session owns: exactly this many releases are owed at the end and not
  // one more, or a session still using the device loses it.
  std::unordered_map<int, int> primary_retains;
};

// Binds this thread to the session it is serving, so the recording underneath
// knows whose resource it is looking at. Passing nullptr unbinds. Binding and
// unbinding is also how the server counts how many sessions are live, which is
// what makes it possible to refuse a call that would reach into all of them.
void inventory_bind(Inventory* inv);

// Takes an InventoryStamp for the resource about to be made. Take one
// immediately before the call that makes the resource, not after it: creating
// a library handle can take hundreds of milliseconds on hardware, and a
// primary context destroyed in that time must be charged to the handle, so
// that expiry skips it rather than destroying it again.
InventoryStamp inventory_stamp();

// Records a maths-library handle against the session being served here, where
// `made` says it was made, and forgets one the client destroyed itself. `what`
// is used in the log and must outlive the session, so a string literal.
void inventory_note_handle(uint64_t handle, const InventoryStamp& made,
                           const char* what, CUresult (*destroy)(uint64_t));
void inventory_forget_handle(uint64_t handle);

// The entry point the server should call for `name`: a wrapper that records
// what the driver hands out, or nullptr if this call creates nothing worth
// remembering. See driver_sym() in server/driver_syms.h.
void* driver_wrapper(const char* name);

// Gives back everything the session still holds, in an order that is safe:
// the captures it left open ended first, so that nothing after them runs
// restricted by them, and their graphs released with the rest; what lives
// inside a context before the context itself; and the shared primary context
// last of all. The serving thread's own capture mode is the caller's to put
// back first (client_threads_relax_capture_mode). Every failure is logged and none of them stops
// the rest. Returns a description of what happened, for the caller's log.
std::string release_inventory(Inventory& inv);

}  // namespace rgpu
