"""Tests for the bidirectional type checker."""
from __future__ import annotations

import pytest
from lambdapy.syntax import (
    Ann, App, Bound, Cons, Eq, EqElim, Fin, FSucc, FZero, FinElim,
    Free, Global, Inf, Lam, Local, Nat, NatElim, Nil, Pi, Refl,
    Star, Succ, Vec, VecElim, Zero,
    VEq, VFin, VFSucc, VFZero, VNat, VNeutral, VPi, VRefl,
    VStar, VSucc, VVec, VZero,
    NFree,
)
from lambdapy.context import HasType, HasKind, empty_context
from lambdapy.errors import TypeCheckError
from lambdapy.check import type_inf, type_inf0, type_chk
from lambdapy.eval import eval_chk


CTX = empty_context()


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

def test_star0_has_type_star1() -> None:
    assert type_inf0(CTX, Star(0)) == VStar(1)


def test_star1_has_type_star2() -> None:
    assert type_inf0(CTX, Star(1)) == VStar(2)


def test_pi_nat_nat_is_star0() -> None:
    # Nat → Nat : Star(0)
    pi = Pi(Inf(Nat()), Inf(Nat()))
    assert type_inf0(CTX, pi) == VStar(0)


def test_pi_universe_max() -> None:
    # (Star(0) → Star(1)) : Star(2)
    pi = Pi(Inf(Star(0)), Inf(Star(1)))
    assert type_inf0(CTX, pi) == VStar(2)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

def test_free_var_lookup() -> None:
    ctx = [(Global("n"), HasType(VNat()))]
    ty = type_inf0(ctx, Free(Global("n")))
    assert ty == VNat()


def test_unknown_var_raises() -> None:
    with pytest.raises(TypeCheckError):
        type_inf0(CTX, Free(Global("unknown")))


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def test_ann_zero_nat() -> None:
    ty = type_inf0(CTX, Ann(Inf(Zero()), Inf(Nat())))
    assert ty == VNat()


def test_ann_wrong_type_raises() -> None:
    # (Zero : Star(0)) should fail — Zero is not a type
    with pytest.raises(TypeCheckError):
        type_inf0(CTX, Ann(Inf(Zero()), Inf(Star(0))))


def test_ann_type_must_be_a_type() -> None:
    # (Zero : Zero) should fail — the annotation Zero is not a universe
    with pytest.raises(TypeCheckError):
        type_inf0(CTX, Ann(Inf(Zero()), Inf(Zero())))


# ---------------------------------------------------------------------------
# Nat
# ---------------------------------------------------------------------------

def test_nat_has_type_star0() -> None:
    assert type_inf0(CTX, Nat()) == VStar(0)


def test_zero_has_type_nat() -> None:
    assert type_inf0(CTX, Zero()) == VNat()


def test_succ_has_type_nat() -> None:
    assert type_inf0(CTX, Succ(Inf(Zero()))) == VNat()


def test_succ_wrong_arg_raises() -> None:
    # Succ Star should fail
    with pytest.raises(TypeCheckError):
        type_inf0(CTX, Succ(Inf(Star(0))))


def test_nat_elim_type_checks() -> None:
    # natElim (\_.Nat) Zero (\k.\r. Succ r) Zero : Nat
    motive = Lam(Inf(Nat()))
    base = Inf(Zero())
    step = Lam(Lam(Inf(Succ(Inf(Bound(0))))))
    k_term = Inf(Zero())
    result_ty = type_inf0(CTX, NatElim(motive, base, step, k_term))
    assert result_ty == VNat()


# ---------------------------------------------------------------------------
# Pi / lambda
# ---------------------------------------------------------------------------

def test_lam_identity_checks() -> None:
    # (\x. x) : Nat → Nat
    lam = Lam(Inf(Bound(0)))
    ty = VPi(VNat(), lambda _: VNat())
    type_chk(0, CTX, lam, ty)  # should not raise


def test_lam_wrong_type_raises() -> None:
    # (\x. x) cannot be checked against Nat
    lam = Lam(Inf(Bound(0)))
    with pytest.raises(TypeCheckError):
        type_chk(0, CTX, lam, VNat())


def test_app_identity() -> None:
    # ((\x.x) : Nat → Nat) Zero : Nat
    id_ann = Ann(Lam(Inf(Bound(0))), Inf(Pi(Inf(Nat()), Inf(Nat()))))
    term = App(id_ann, Inf(Zero()))
    assert type_inf0(CTX, term) == VNat()


def test_app_non_function_raises() -> None:
    with pytest.raises(TypeCheckError):
        type_inf0(CTX, App(Zero(), Inf(Zero())))


def test_dependent_id() -> None:
    # id : (A : Star(0)) → A → A
    # \A. \x. x
    id_ty = Pi(Inf(Star(0)), Inf(Pi(Inf(Bound(0)), Inf(Bound(1)))))
    id_term = Ann(Lam(Lam(Inf(Bound(0)))), Inf(id_ty))
    # Apply id to Nat to get Nat → Nat
    app1 = App(id_term, Inf(Nat()))
    result = type_inf0(CTX, app1)
    # Result should be Nat → Nat
    from lambdapy.quote import quote0
    assert quote0(result) == quote0(VPi(VNat(), lambda _: VNat()))


# ---------------------------------------------------------------------------
# Vec
# ---------------------------------------------------------------------------

def test_vec_type_checks() -> None:
    ty = type_inf0(CTX, Vec(Inf(Nat()), Inf(Zero())))
    assert ty == VStar(0)


def test_nil_has_vec_type() -> None:
    ty = type_inf0(CTX, Nil(Inf(Nat())))
    assert ty == VVec(VNat(), VZero())


def test_cons_has_vec_type() -> None:
    # Cons Nat 0 Zero (Nil Nat) : Vec Nat 1
    ty = type_inf0(CTX, Cons(Inf(Nat()), Inf(Zero()), Inf(Zero()), Inf(Nil(Inf(Nat())))))
    assert ty == VVec(VNat(), VSucc(VZero()))


def test_vec_elim_nil_case() -> None:
    # vecElim Nat (\n.\v. Nat) Zero (\n.\h.\t.\r. Succ r) Zero (Nil Nat) : Nat
    motive = Lam(Lam(Inf(Nat())))
    nil_case = Inf(Zero())
    cons_case = Lam(Lam(Lam(Lam(Inf(Succ(Inf(Bound(0))))))))
    term = VecElim(Inf(Nat()), motive, nil_case, cons_case, Inf(Zero()), Inf(Nil(Inf(Nat()))))
    result = type_inf0(CTX, term)
    assert result == VNat()


# ---------------------------------------------------------------------------
# Fin
# ---------------------------------------------------------------------------

def test_fin_type_checks() -> None:
    ty = type_inf0(CTX, Fin(Inf(Succ(Inf(Zero())))))
    assert ty == VStar(0)


def test_fzero_has_fin_type() -> None:
    ty = type_inf0(CTX, FZero(Inf(Zero())))
    assert ty == VFin(VSucc(VZero()))


def test_fsucc_has_fin_type() -> None:
    # FZero(n) : Fin(Succ n); FSucc(n, x) where x : Fin(n) : Fin(Succ n)
    # FZero(0) : Fin(1);  FSucc(1, FZero(0)) : Fin(2)
    ty = type_inf0(CTX, FSucc(Inf(Succ(Inf(Zero()))), Inf(FZero(Inf(Zero())))))
    assert ty == VFin(VSucc(VSucc(VZero())))


# ---------------------------------------------------------------------------
# Eq
# ---------------------------------------------------------------------------

def test_eq_type_checks() -> None:
    ty = type_inf0(CTX, Eq(Inf(Nat()), Inf(Zero()), Inf(Zero())))
    assert ty == VStar(0)


def test_refl_has_eq_type() -> None:
    ty = type_inf0(CTX, Refl(Inf(Nat()), Inf(Zero())))
    assert ty == VEq(VNat(), VZero(), VZero())


def test_eq_elim_type_checks() -> None:
    # eqElim Nat (\x.\y.\p. Nat) (\z. Zero) Zero Zero (Refl Nat Zero) : Nat
    motive = Lam(Lam(Lam(Inf(Nat()))))
    refl_case = Lam(Inf(Zero()))
    term = EqElim(
        Inf(Nat()), motive, refl_case,
        Inf(Zero()), Inf(Zero()), Inf(Refl(Inf(Nat()), Inf(Zero())))
    )
    result = type_inf0(CTX, term)
    assert result == VNat()
