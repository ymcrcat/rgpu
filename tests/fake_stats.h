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
  // Stream captures begun and not yet ended. Not a handle anybody frees, but
  // one left open is a leak all the same: it holds its stream in capture and
  // restricts the thread that began it.
  kCapture,
  // Releases of a primary context that nobody had retained. Not a resource:
  // a mistake, and the one a test cannot otherwise see, because the fake
  // refuses the call and the count it would have corrupted stays right.
  kOverRelease,
  // Frees and destroys of something the fake never handed out, or already took
  // back. The same kind of mistake as an over-release, and just as invisible:
  // the fake refuses the call, so no resource count moves.
  kStale,
  // Not a resource and not a mistake: how many times cuCtxSetCurrent was
  // called, by anyone. The server restores each client thread's context before
  // its request only when it differs from the one already current, and this
  // is how a test sees that a client with one thread costs no switches at all.
  kCtxSetCurrent,
  // Device-memory reads and writes that ran while a context other than the
  // memory's own was current. They succeed - the driver infers placement from
  // the pointer - so the count is the only trace of work that ran under the
  // wrong context, which is what a batch flushed by another thread would do if
  // the server ran it under the flushing thread's context.
  kCrossContextUse,
  // Not a resource either: how many times cuDeviceTotalMem ran, counted as it
  // starts. With RGPU_FAKE_SLOW_TOTALMEM_MS it is also slow, which is how a
  // test gets a request still running when its client reconnects, and this is
  // how it sees whether the server ran the request again.
  kTotalMem,
  // Not a resource: how many times cuThreadExchangeStreamCaptureMode ran. The
  // server puts a client thread's capture mode back before its request only
  // when it differs from the serving thread's, and this is how a test sees
  // that a client whose threads keep one mode pays nothing for it.
  kCaptureModeExchange,
  // Not a resource: how many memsets ran, counted as each starts. A test that
  // queues memsets without replies reads it to see that they really were held
  // back until another call flushed them, rather than sent one by one.
  kMemset,
  kKindCount,
};

// Adds `delta` to a counter and republishes the whole set to the file named by
// RGPU_FAKE_STATS, if one is named. Cheap enough for a fake; this is not a
// path any real deployment runs.
void count(Kind k, int delta);

// The counter as it stands, for a test that links the fake directly rather
// than reading it back through RGPU_FAKE_STATS.
long value(Kind k);

}  // namespace rgpu_fake
