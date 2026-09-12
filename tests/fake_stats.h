// What the fake driver and fake libraries have handed out and not got back.
//
// A leak is invisible from the client side: the server hands back real handles
// and real device pointers, and a client that never frees them simply stops
// mentioning them. The only place the truth lives is the bottom of the stack,
// so the fakes count it, and a test that wants to know whether an expired
// session's resources came back reads the counters from here.
//
// Counting is only in the fake build. Nothing in tests/ is linked into the
// real server.
#pragma once

namespace rgpu_fake {

// One counter per kind of thing a fake hands out. Order matches the names in
// tests/fake_stats.cpp, which is what a test reads.
enum Kind {
  kAlloc,
  kPrimaryRetain,
  kContext,
  kModule,
  kStream,
  kEvent,
  kGraph,
  kGraphExec,
  kCublas,
  kCublasLt,
  kCudnn,
  kKindCount,
};

// Adds `delta` to a counter and republishes the whole set to the file named by
// RGPU_FAKE_STATS, if one is named. Cheap enough for a fake; this is not a
// path any real deployment runs.
void count(Kind k, int delta);

}  // namespace rgpu_fake
