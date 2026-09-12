import torch

from rgpu import wire
from rgpu.server.session import Session


class Client:
    """Drives a Session directly, the way the network layer will."""

    def __init__(self):
        self.s = Session("cpu")
        self.seq = 0

    def send(self, kind, *fields):
        self.seq += 1
        return self.s.execute([self.seq, kind, *fields])

    def reply(self, kind, *fields):
        seq, status, value = wire.decode(self.send(kind, *fields))
        assert seq == self.seq
        return status, value

    def put(self, tid, t):
        self.send(wire.RUN, "aten::empty_strided", "default",
                  [list(t.shape), list(t.stride())],
                  {"dtype": t.dtype, "device": wire.Dev("rgpu")}, [tid])
        self.send(wire.UPLOAD, tid, wire.Host.of(t))

    def get(self, tid):
        status, value = self.reply(wire.DOWNLOAD, tid)
        assert status == wire.OK, value
        return value.tensor()


def test_ops_run_in_order_and_results_come_back():
    c = Client()
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    c.put(1, a)
    c.put(2, b)
    c.send(wire.RUN, "aten::add", "Tensor", [wire.Ref(1), wire.Ref(2)], {}, [3])
    assert torch.allclose(c.get(3), a + b)


def test_cpu_tensors_travel_inline():
    c = Client()
    a = torch.randn(5)
    c.put(1, a)
    c.send(wire.RUN, "aten::mul", "Tensor",
           [wire.Ref(1), wire.Host.of(torch.tensor(2.0))], {}, [2])
    assert torch.allclose(c.get(2), a * 2)


def test_views_alias_their_base_on_the_server():
    c = Client()
    c.put(1, torch.zeros(2, 3))
    c.send(wire.RUN, "aten::t", "default", [wire.Ref(1)], {}, [2])
    c.send(wire.RUN, "aten::add_", "Scalar", [wire.Ref(2), 1.0], {}, [None])
    assert torch.equal(c.get(1), torch.ones(2, 3))


def test_a_failure_poisons_what_depends_on_it_and_is_reported_once():
    c = Client()
    c.put(1, torch.randn(2, 3))
    c.put(2, torch.randn(4, 5))
    c.send(wire.RUN, "aten::mm", "default", [wire.Ref(1), wire.Ref(2)], {}, [3])
    c.send(wire.RUN, "aten::relu", "default", [wire.Ref(3)], {}, [4])
    status, (op, message) = c.reply(wire.SYNC)
    assert status == wire.ERROR and op == "aten::mm.default"
    assert c.reply(wire.SYNC)[0] == wire.OK          # reported once, not twice
    status, (op, _) = c.reply(wire.DOWNLOAD, 4)
    assert status == wire.ERROR and op == "aten::mm.default"  # but it sticks


def test_blocked_and_unknown_ops_are_refused():
    c = Client()
    c.send(wire.RUN, "aten::from_file", "default", ["/etc/passwd"], {}, [1])
    status, (op, message) = c.reply(wire.SYNC)
    assert status == wire.ERROR and "not allowed" in message
    c.send(wire.RUN, "aten::__dict__", "default", [], {}, [2])
    assert c.reply(wire.SYNC)[0] == wire.ERROR
    c.send(wire.RUN, "prims::add", "default", [], {}, [3])
    assert c.reply(wire.SYNC)[0] == wire.ERROR


def test_run_sync_reports_shapes_it_chose_and_keeps_the_tensor():
    c = Client()
    c.put(1, torch.tensor([0.0, 3.0, 0.0, 5.0]))
    status, desc = c.reply(wire.RUN_SYNC, "aten::nonzero", "default", [wire.Ref(1)], {})
    assert status == wire.OK
    tag, tid, dtype, shape, stride, offset = desc
    assert tag == "__tensor__" and tid >= wire.SERVER_ID_BASE
    assert shape == [2, 1] and dtype == torch.int64
    assert torch.equal(c.get(tid), torch.tensor([[1], [3]]))


def test_replayed_messages_are_ignored():
    c = Client()
    c.put(1, torch.zeros(3))
    c.send(wire.RUN, "aten::add_", "Scalar", [wire.Ref(1), 1.0], {}, [None])
    c.seq -= 1   # the same message again, as after a reconnect
    c.send(wire.RUN, "aten::add_", "Scalar", [wire.Ref(1), 1.0], {}, [None])
    assert torch.equal(c.get(1), torch.ones(3))


def test_an_op_returning_nothing_is_not_mistaken_for_a_failure():
    c = Client()
    c.put(1, torch.zeros(3))
    c.put(2, torch.zeros(3))
    c.send(wire.RUN, "aten::_foreach_add_", "Scalar", [[wire.Ref(1), wire.Ref(2)], 1.0], {}, [])
    assert c.reply(wire.SYNC)[0] == wire.OK
    assert torch.equal(c.get(2), torch.ones(3))


def test_free_forgets_tensors():
    c = Client()
    c.put(1, torch.zeros(3))
    c.send(wire.FREE, [1])
    assert c.reply(wire.DOWNLOAD, 1)[0] == wire.ERROR


def test_calling_a_graph_that_was_never_compiled_is_reported_at_the_next_wait():
    """Every other failure path records the first error, so the client hears
    about it at the next wait. This one only poisoned its outputs, so a CALL
    for a graph id that was never compiled went unreported unless an output
    happened to be downloaded."""
    c = Client()
    c.send(wire.CALL, 7, [], [1])
    status, (op, message) = c.reply(wire.SYNC)
    assert status == wire.ERROR and "graph 7" in op and "never compiled" in message


def test_seed_makes_random_ops_repeat():
    c = Client()
    for tid in (1, 2):
        c.send(wire.SEED, 7)
        c.send(wire.RUN, "aten::randn", "default", [[8]],
               {"dtype": torch.float32, "device": wire.Dev("rgpu")}, [tid])
    assert torch.equal(c.get(1), c.get(2))
