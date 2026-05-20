"""Tests for the subtyping pass (cumulativity + variance)."""
import pytest

from lambdapy.check import (
    _is_subtype, _subtype_chk, _subtype_inf, type_chk, type_inf0,
)
from lambdapy.errors import TypeCheckError
from lambdapy.eval import eval_chk, eval_inf
from lambdapy.parser import parse
from lambdapy.repl import fresh_state, execute
from lambdapy.syntax import (
    Ann, Inf, Lam, Nat, Pi, Star, Zero,
    VNat, VPi, VStar, VZero,
)


# ---------------------------------------------------------------------------
# Universe cumulativity
# ---------------------------------------------------------------------------

def test_star_cumulative_up() -> None:
    assert _is_subtype(VStar(0), VStar(1)) is True
    assert _is_subtype(VStar(0), VStar(7)) is True


def test_star_cumulative_reflexive() -> None:
    for i in range(5):
        assert _is_subtype(VStar(i), VStar(i)) is True


def test_star_cumulative_not_down() -> None:
    assert _is_subtype(VStar(1), VStar(0)) is False
    assert _is_subtype(VStar(7), VStar(3)) is False


# ---------------------------------------------------------------------------
# Pi variance
# ---------------------------------------------------------------------------

def test_pi_domain_contravariant() -> None:
    # (Star 1 -> Nat) <: (Star 0 -> Nat)
    sup = VPi(VStar(0), lambda _: VNat())
    sub = VPi(VStar(1), lambda _: VNat())
    assert _is_subtype(sub, sup) is True
    assert _is_subtype(sup, sub) is False


def test_pi_codomain_covariant() -> None:
    # (Nat -> Star 0) <: (Nat -> Star 1)
    sub = VPi(VNat(), lambda _: VStar(0))
    sup = VPi(VNat(), lambda _: VStar(1))
    assert _is_subtype(sub, sup) is True
    assert _is_subtype(sup, sub) is False


def test_pi_both_axes() -> None:
    # (Star 1 -> Star 0) <: (Star 0 -> Star 1)
    sub = VPi(VStar(1), lambda _: VStar(0))
    sup = VPi(VStar(0), lambda _: VStar(1))
    assert _is_subtype(sub, sup) is True
    assert _is_subtype(sup, sub) is False


def test_pi_invariant_when_levels_match() -> None:
    a = VPi(VStar(0), lambda _: VStar(0))
    b = VPi(VStar(0), lambda _: VStar(0))
    assert _is_subtype(a, b) is True
    assert _is_subtype(b, a) is True


# ---------------------------------------------------------------------------
# Reflexivity / negatives
# ---------------------------------------------------------------------------

def test_nat_reflexive() -> None:
    assert _is_subtype(VNat(), VNat()) is True


def test_distinct_constructors_not_subtypes() -> None:
    assert _is_subtype(VNat(), VStar(0)) is False
    assert _is_subtype(VStar(0), VNat()) is False


# ---------------------------------------------------------------------------
# Type-in-type still rejected
# ---------------------------------------------------------------------------

def test_type_in_type_is_rejected() -> None:
    # Star(0) annotated as Star(0) should fail: Star(0) : Star(1), and
    # Star(1) </: Star(0).
    with pytest.raises(TypeCheckError):
        type_inf0([], Ann(Inf(Star(0)), Inf(Star(0))))


# ---------------------------------------------------------------------------
# Surface-level integration
# ---------------------------------------------------------------------------

def _run(src: str):
    state = fresh_state()
    last = ""
    for stmt in parse(src):
        state, last = execute(state, stmt)
    return state, last


def test_eqelim_apply_with_universe_argument() -> None:
    """The case from the original bug report."""
    src = (
        "let apply = eqElim * (\\ a b _ -> a -> b) (\\ _ x -> x) "
        ":: forall (a :: *) (b :: *) (p :: Eq * a b) . a -> b\n"
        "eval apply Nat Nat (Refl * Nat) Zero\n"
    )
    _, last = _run(src)
    assert "Zero : Nat" in last


def test_pi_contravariant_domain_accepted() -> None:
    src = (
        "let bigToNat = (\\a -> Zero) :: forall (a :: Type 1) . Nat\n"
        "let smallToNat : forall (a :: Type 0) . Nat = bigToNat\n"
    )
    _, last = _run(src)
    # Just running without exception is the assertion.
    assert "smallToNat" in last


def test_pi_contravariant_domain_wrong_direction_rejected() -> None:
    src = (
        "let smallToNat = (\\a -> Zero) :: forall (a :: Type 0) . Nat\n"
        "let bigToNat : forall (a :: Type 1) . Nat = smallToNat\n"
    )
    with pytest.raises(TypeCheckError):
        _run(src)


def test_pi_covariant_codomain_accepted() -> None:
    src = (
        "let mkType0 = (\\n -> Nat) :: Nat -> Type 0\n"
        "let mkType1 : Nat -> Type 1 = mkType0\n"
    )
    _, last = _run(src)
    assert "mkType1" in last


def test_pi_covariant_codomain_wrong_direction_rejected() -> None:
    src = (
        "let mkType1 = (\\n -> Type) :: Nat -> Type 2\n"
        "let mkType0 : Nat -> Type 0 = mkType1\n"
    )
    with pytest.raises(TypeCheckError):
        _run(src)