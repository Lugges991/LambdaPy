"""Tests for quotation and substitution."""
from __future__ import annotations

import pytest
from lambdapy.syntax import (
    App, Bound, Free, Global, Inf, Lam, Nat, Pi, Star, Succ, Zero,
    VLam, VNat, VNeutral, VPi, VStar, VSucc, VZero,
    NFree, Quote,
)
from lambdapy.eval import eval_chk, eval_inf, vapp
from lambdapy.quote import boundfree, quote, quote0, neutral_quote
from lambdapy.subst import subst_chk, subst_inf


# ---------------------------------------------------------------------------
# boundfree
# ---------------------------------------------------------------------------

def test_boundfree_global() -> None:
    assert boundfree(3, Global("x")) == Free(Global("x"))


def test_boundfree_quote_innermost() -> None:
    # At depth i+1, Quote(i) → Bound(0)
    assert boundfree(1, Quote(0)) == Bound(0)
    assert boundfree(2, Quote(1)) == Bound(0)


def test_boundfree_quote_outer() -> None:
    # At depth 2, Quote(0) → Bound(1)
    assert boundfree(2, Quote(0)) == Bound(1)


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------

def test_quote_neutral_free() -> None:
    v = VNeutral(NFree(Global("x")))
    assert quote0(v) == Inf(Free(Global("x")))


def test_quote_vstar() -> None:
    assert quote0(VStar(0)) == Inf(Star(0))
    assert quote0(VStar(3)) == Inf(Star(3))


def test_quote_vnat() -> None:
    assert quote0(VNat()) == Inf(Nat())


def test_quote_vzero() -> None:
    assert quote0(VZero()) == Inf(Zero())


def test_quote_vsucc() -> None:
    assert quote0(VSucc(VZero())) == Inf(Succ(Inf(Zero())))


def test_quote_vlam_identity() -> None:
    # VLam(lambda v: v) should quote to  Lam(Inf(Bound(0)))
    v = VLam(fn=lambda x: x)
    assert quote0(v) == Lam(Inf(Bound(0)))


def test_quote_vlam_const() -> None:
    # VLam(lambda _: VZero()) should quote to  Lam(Inf(Zero()))
    v = VLam(fn=lambda _: VZero())
    assert quote0(v) == Lam(Inf(Zero()))


def test_quote_vpi_nondep() -> None:
    # VPi(VNat(), lambda _: VNat()) → Inf(Pi(Inf(Nat()), Inf(Nat())))
    v = VPi(VNat(), lambda _: VNat())
    assert quote0(v) == Inf(Pi(Inf(Nat()), Inf(Nat())))


def test_quote_vpi_dep() -> None:
    # VPi(VNat(), lambda v: v)  is the "identity Pi" — range is the domain value
    # The body Bound(0) refers to the Pi's variable
    v = VPi(VNat(), lambda x: x)
    result = quote0(v)
    assert result == Inf(Pi(Inf(Nat()), Inf(Bound(0))))


# ---------------------------------------------------------------------------
# Definitional equality via quotation
# ---------------------------------------------------------------------------

def test_def_eq_identity_applied() -> None:
    # (\x. x) a  should be definitionally equal to  a
    id_val = eval_chk(Lam(Inf(Bound(0))), [])
    a = VNeutral(NFree(Global("a")))
    applied = vapp(id_val, a)
    assert quote0(applied) == quote0(a)


def test_def_eq_beta_reduction() -> None:
    # (\x. Succ x) Zero = Succ Zero
    lam = eval_chk(Lam(Inf(Succ(Inf(Bound(0))))), [])
    result = vapp(lam, VZero())
    assert quote0(result) == quote0(VSucc(VZero()))


def test_def_neq_different_values() -> None:
    assert quote0(VZero()) != quote0(VSucc(VZero()))


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def test_subst_bound_match() -> None:
    # subst_inf(0, Free(x), Bound(0)) = Free(x)
    assert subst_inf(0, Free(Global("x")), Bound(0)) == Free(Global("x"))


def test_subst_bound_no_match() -> None:
    assert subst_inf(0, Free(Global("x")), Bound(1)) == Bound(1)


def test_subst_free_unchanged() -> None:
    assert subst_inf(0, Free(Global("x")), Free(Global("y"))) == Free(Global("y"))


def test_subst_under_lam() -> None:
    # subst_chk(0, r, Lam(Inf(Bound(1)))) replaces Bound(1) inside the Lam body
    # (index becomes 1 inside because we're under a binder)
    r = Free(Global("r"))
    result = subst_chk(0, r, Lam(Inf(Bound(1))))
    assert result == Lam(Inf(r))


def test_subst_preserves_inner_bound() -> None:
    # subst_chk(0, r, Lam(Inf(Bound(0)))) — Bound(0) is the lambda's own var,
    # so it should NOT be replaced (we substitute for Bound(1) inside the lam)
    r = Free(Global("r"))
    result = subst_chk(0, r, Lam(Inf(Bound(0))))
    assert result == Lam(Inf(Bound(0)))


def test_subst_pi_range_increments() -> None:
    # Pi(domain, range) — substituting for Bound(0) increments to Bound(1) in range
    r = Free(Global("r"))
    pi = Pi(Inf(Bound(0)), Inf(Bound(1)))
    result = subst_inf(0, r, pi)
    # domain: Bound(0) → r; range: Bound(1) → r (i+1 = 1 matches Bound(1))
    assert result == Pi(Inf(r), Inf(r))
