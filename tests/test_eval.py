"""Tests for the NbE evaluator."""
from __future__ import annotations

import pytest
from lambdapy.syntax import (
    Ann, App, Bound, Cons, Free, Global, Inf, Lam, Nat, NatElim,
    Nil, Pi, Star, Succ, Vec, VecElim, Zero,
    VCons, VLam, VNat, VNeutral, VNil, VPi, VStar, VSucc, VVec, VZero,
    NFree, NApp, NNatElim,
)
from lambdapy.eval import eval_inf, eval_chk, vapp, nat_elim, vec_elim


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------

def test_free_var_evaluates_to_neutral() -> None:
    v = eval_inf(Free(Global("x")), [])
    assert isinstance(v, VNeutral)
    assert v.neutral == NFree(Global("x"))


def test_bound_var_looks_up_env() -> None:
    sentinel: object = VZero()
    result = eval_inf(Bound(0), [VZero()])  # type: ignore[list-item]
    assert result == VZero()


def test_ann_evaluates_expr() -> None:
    # (Zero : Nat) evaluates to VZero
    term = Ann(Inf(Zero()), Inf(Nat()))
    assert eval_inf(term, []) == VZero()


def test_star_evaluates_to_vstar() -> None:
    assert eval_inf(Star(0), []) == VStar(0)
    assert eval_inf(Star(2), []) == VStar(2)


def test_nat_evaluates_to_vnat() -> None:
    assert eval_inf(Nat(), []) == VNat()


def test_zero_evaluates_to_vzero() -> None:
    assert eval_inf(Zero(), []) == VZero()


def test_succ_evaluates_to_vsucc() -> None:
    result = eval_inf(Succ(Inf(Zero())), [])
    assert result == VSucc(VZero())


# ---------------------------------------------------------------------------
# Lambda and application
# ---------------------------------------------------------------------------

def test_lam_creates_vlam() -> None:
    lam = Lam(Inf(Bound(0)))
    val = eval_chk(lam, [])
    assert isinstance(val, VLam)


def test_identity_lam_applied() -> None:
    # (\x. x) applied to VZero() should give VZero()
    id_val = eval_chk(Lam(Inf(Bound(0))), [])
    assert vapp(id_val, VZero()) == VZero()


def test_const_lam() -> None:
    # (\x. \y. x) applied to a then b gives a
    const_val = eval_chk(Lam(Lam(Inf(Bound(1)))), [])
    a = VNeutral(NFree(Global("a")))
    b = VNeutral(NFree(Global("b")))
    inner = vapp(const_val, a)
    assert vapp(inner, b) == a


def test_app_on_neutral() -> None:
    # Applying a neutral function gives a neutral result
    f = VNeutral(NFree(Global("f")))
    result = vapp(f, VZero())
    assert isinstance(result, VNeutral)
    assert result.neutral == NApp(NFree(Global("f")), VZero())


def test_app_eval() -> None:
    # (\x. x) Zero evaluates to VZero()
    term = App(Ann(Lam(Inf(Bound(0))), Inf(Pi(Inf(Nat()), Inf(Nat())))), Inf(Zero()))
    assert eval_inf(term, []) == VZero()


# ---------------------------------------------------------------------------
# Pi type evaluation
# ---------------------------------------------------------------------------

def test_pi_evaluates_to_vpi() -> None:
    pi_term = Pi(Inf(Nat()), Inf(Nat()))
    val = eval_inf(pi_term, [])
    assert isinstance(val, VPi)
    assert val.domain == VNat()


def test_pi_range_is_closure() -> None:
    # Pi(Nat, Bound(0)) — range depends on argument
    pi_term = Pi(Inf(Nat()), Inf(Bound(0)))
    val = eval_inf(pi_term, [])
    assert isinstance(val, VPi)
    # Applying the range closure to VZero() should yield VZero()
    assert val.range_fn(VZero()) == VZero()


# ---------------------------------------------------------------------------
# Nat eliminator
# ---------------------------------------------------------------------------

def test_nat_elim_zero() -> None:
    # natElim P base step Zero = base
    motive = VLam(lambda _: VNat())
    base = VNeutral(NFree(Global("base")))
    step = VLam(lambda k: VLam(lambda r: VSucc(r)))
    result = nat_elim(motive, base, step, VZero())
    assert result == base


def test_nat_elim_succ_one() -> None:
    # natElim P base step (Succ Zero) = step Zero base
    motive = VLam(lambda _: VNat())
    base = VZero()
    step = VLam(lambda k: VLam(lambda r: VSucc(r)))
    result = nat_elim(motive, base, step, VSucc(VZero()))
    assert result == VSucc(VZero())


def test_nat_elim_succ_two() -> None:
    # natElim P Zero (\k r. Succ r) (Succ (Succ Zero)) = Succ (Succ Zero)
    motive = VLam(lambda _: VNat())
    base = VZero()
    step = VLam(lambda k: VLam(lambda r: VSucc(r)))
    n2 = VSucc(VSucc(VZero()))
    result = nat_elim(motive, base, step, n2)
    assert result == VSucc(VSucc(VZero()))


def test_nat_elim_neutral_goes_neutral() -> None:
    n = VNeutral(NFree(Global("n")))
    motive = VLam(lambda _: VNat())
    base = VZero()
    step = VLam(lambda k: VLam(lambda r: VSucc(r)))
    result = nat_elim(motive, base, step, n)
    assert isinstance(result, VNeutral)
    assert isinstance(result.neutral, NNatElim)


# ---------------------------------------------------------------------------
# Vec eliminator
# ---------------------------------------------------------------------------

def test_vec_elim_nil() -> None:
    nil_case = VNeutral(NFree(Global("nil_result")))
    cons_case = VLam(lambda n: VLam(lambda h: VLam(lambda t: VLam(lambda r: r))))
    result = vec_elim(VNat(), VLam(lambda _: VLam(lambda _: VNat())),
                      nil_case, cons_case, VZero(), VNil(VNat()))
    assert result == nil_case


def test_vec_elim_singleton() -> None:
    # vecElim _ _ nil_c cons_c 1 (Cons _ 0 Zero Nil)
    # should call cons_c 0 Zero Nil nil_c
    nil_case = VNeutral(NFree(Global("nil_result")))
    # cons_case returns the head
    cons_case = VLam(lambda n: VLam(lambda h: VLam(lambda t: VLam(lambda r: h))))
    vec = VCons(VNat(), VZero(), VZero(), VNil(VNat()))
    result = vec_elim(VNat(), VLam(lambda _: VLam(lambda _: VNat())),
                      nil_case, cons_case, VSucc(VZero()), vec)
    assert result == VZero()
