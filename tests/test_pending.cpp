// Unit test for the record of outstanding GPU work (client/pending.h), which
// decides whether a stream synchronize has to wait. No server, no shim.

// The build is RelWithDebInfo, which defines NDEBUG and so turns every
// assert into nothing. Undefined here, or this test checks nothing at all.
#undef NDEBUG
#include <cassert>
#include <cstdio>

#include "client/pending.h"

int main() {
  using namespace rgpu;
  const uint64_t a = 0x1000, b = 0x2000;

  // Nothing issued: nothing to wait for, on any stream.
  assert(nothing_to_wait_for(a) && nothing_to_wait_for(b) && nothing_to_wait_for(0));

  // Work on one stream is that stream's alone.
  work_on(a);
  assert(!nothing_to_wait_for(a));
  assert(nothing_to_wait_for(b));
  // The default stream orders itself against the others, so it has to wait.
  assert(!nothing_to_wait_for(0));

  // Waiting for another stream changes nothing for this one.
  settled(b);
  assert(!nothing_to_wait_for(a));
  // Waiting for this one does.
  settled(a);
  assert(nothing_to_wait_for(a) && nothing_to_wait_for(0));

  // Work on the default stream is its own, like any other's.
  work_on(0);
  assert(!nothing_to_wait_for(0) && nothing_to_wait_for(a));
  settled(0);
  assert(nothing_to_wait_for(0));

  // Work that cannot be placed counts against every stream, and waiting for
  // one stream does not clear it: only a wait for the whole context does.
  work_somewhere();
  assert(!nothing_to_wait_for(a) && !nothing_to_wait_for(b) && !nothing_to_wait_for(0));
  settled(a);
  assert(!nothing_to_wait_for(a));
  work_on(b);
  settled_all();
  assert(nothing_to_wait_for(a) && nothing_to_wait_for(b) && nothing_to_wait_for(0));

  std::printf("pending-work record OK\n");
  return 0;
}
