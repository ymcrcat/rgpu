import gc

import pytest
import torch

import rgpu
from rgpu import session, wire
from rgpu.tensor import id_of


def test_a_collected_tensor_is_freed_on_the_server():
    t = torch.ones(1000, device="rgpu")
    tid = id_of(t._rgpu_meta)
    assert torch.equal(session.get().request(wire.DOWNLOAD, tid).tensor(), torch.ones(1000))
    del t
    gc.collect()
    torch.rgpu.synchronize()   # a safe point: the free goes out first
    with pytest.raises(rgpu.RemoteError, match="no tensor"):
        session.get().request(wire.DOWNLOAD, tid)


def test_a_view_keeps_the_storage_after_its_base_is_freed():
    base = torch.arange(6.0).to("rgpu")
    view = base[2:]
    del base
    gc.collect()
    torch.rgpu.synchronize()
    assert torch.equal(view.cpu(), torch.arange(2.0, 6.0))


def test_a_long_chain_of_freed_temporaries_stays_correct():
    x = torch.randn(256, device="rgpu")
    for _ in range(200):
        x = x * 1.0 + 0.0
    gc.collect()
    torch.rgpu.synchronize()
    ids = [id_of(x._rgpu_meta)]
    assert torch.allclose(session.get().request(wire.DOWNLOAD, ids[0]).tensor(), x.cpu())
