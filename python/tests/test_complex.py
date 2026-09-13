"""Complex tensors on rgpu, checked against the same thing on the CPU.

Conjugation is not an op: torch flips a bit on the tensor and leaves the
values alone, then honours it lazily wherever they are read. The bit lives on
the tensor object, so on rgpu it has to live on the wrapper - the inner meta
tensor carrying it is not enough, because every use of the tensor goes through
the wrapper first. The negative bit works the same way, and .imag of a
conjugated tensor is where torch produces one.
"""

import pytest
import torch

import rgpu


X = torch.tensor([1 + 2j, 3 + 4j, -5 + 0.5j])


@pytest.fixture
def pair():
    """The same complex tensor on the CPU and on rgpu."""
    return X, X.to("rgpu")


def test_conj_sets_the_conjugate_bit(pair):
    x, r = pair
    assert x.conj().is_conj() is True
    assert r.conj().is_conj() == x.conj().is_conj()


def test_real_and_imag_of_a_conjugated_tensor_match_cpu(pair):
    x, r = pair
    assert torch.equal(r.conj().real.cpu(), x.conj().real)
    assert torch.equal(r.conj().imag.cpu(), x.conj().imag)
    # .imag of a conjugated tensor is a negated view, not fresh values.
    assert x.conj().imag.is_neg() is True
    assert r.conj().imag.is_neg() == x.conj().imag.is_neg()


def test_real_and_imag_without_conjugation_still_match_cpu(pair):
    x, r = pair
    assert torch.equal(r.real.cpu(), x.real)
    assert torch.equal(r.imag.cpu(), x.imag)
    assert r.imag.is_neg() == x.imag.is_neg() is False


def test_resolve_conj_matches_cpu(pair):
    x, r = pair
    resolved = r.conj().resolve_conj()
    assert resolved.is_conj() == x.conj().resolve_conj().is_conj() is False
    assert torch.equal(resolved.cpu(), x.conj().resolve_conj())


def test_resolve_neg_matches_cpu(pair):
    x, r = pair
    negated = torch._neg_view(r.real)
    assert negated.is_neg() == torch._neg_view(x.real).is_neg() is True
    assert torch.equal(negated.cpu(), torch._neg_view(x.real))
    assert negated.resolve_neg().is_neg() is False
    assert torch.equal(negated.resolve_neg().cpu(), torch._neg_view(x.real).resolve_neg())


@pytest.mark.parametrize("f", [
    lambda t: t.conj() + t,
    lambda t: t.conj() * 2,
    lambda t: t.conj() * t.conj(),
    lambda t: t.conj().abs(),
    lambda t: t.conj().sum(),
    lambda t: torch.matmul(t.conj(), t),
])
def test_arithmetic_on_a_conjugated_tensor_matches_cpu(pair, f):
    x, r = pair
    assert torch.allclose(f(r).cpu(), f(x))


def test_downloading_a_conjugated_tensor_gives_the_conjugated_values(pair):
    x, r = pair
    assert torch.equal(r.conj().cpu(), x.conj())
    assert torch.equal(r.conj().cpu(), x.conj().resolve_conj())


def test_conjugating_twice_clears_the_bit_as_on_cpu(pair):
    x, r = pair
    assert r.conj().conj().is_conj() == x.conj().conj().is_conj() is False
    assert torch.equal(r.conj().conj().cpu(), x)


def test_conj_of_a_real_tensor_sets_nothing(pair):
    x = torch.randn(4)
    assert x.to("rgpu").conj().is_conj() == x.conj().is_conj() is False


def test_backward_through_a_conjugated_tensor_matches_cpu():
    def grad_of(t):
        t = t.detach().requires_grad_(True)
        (t.conj() * t).abs().sum().backward()
        return t.grad
    x = torch.tensor([1 + 2j, 3 + 4j])
    assert torch.allclose(grad_of(x.to("rgpu")).cpu(), grad_of(x))
