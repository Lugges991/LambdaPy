"""Tests for the pretty printer."""
from __future__ import annotations

import pytest
from lambdapy.syntax import (
    Bound, Free, Global, Inf, Lam, Nat, Pi, Star, Succ, Zero,
    VLam, VNat, VNeutral, VPi, VStar, VSucc, VZero,
    NFree,
)
from lambdapy.eval import eval_chk, eval_inf
from lambdapy.pretty import pretty_check, pretty_infer, pretty_value


def test_pretty_zero() -> None:
    assert pretty_value(VZero()) == "Zero"


def test_pretty_succ_as_number() -> None:
    assert pretty_value(VSucc(VZero())) == "1"
    assert pretty_value(VSucc(VSucc(VZero()))) == "2"


def test_pretty_nat_type() -> None:
    assert pretty_value(VNat()) == "Nat"


def test_pretty_star() -> None:
    assert pretty_value(VStar(0)) == "Type"
    assert pretty_value(VStar(1)) == "Type 1"


def test_pretty_free_var() -> None:
    v = VNeutral(NFree(Global("x")))
    assert pretty_value(v) == "x"


def test_pretty_nondep_pi() -> None:
    # Nat -> Nat
    v = VPi(VNat(), lambda _: VNat())
    s = pretty_value(v)
    assert s == "Nat -> Nat"


def test_pretty_dep_pi() -> None:
    # (n : Nat) -> Nat  (non-dependent, so printed as Nat -> Nat)
    v = VPi(VNat(), lambda _: VNat())
    s = pretty_value(v)
    assert "->" in s


def test_pretty_lam() -> None:
    v = VLam(fn=lambda x: x)
    s = pretty_value(v)
    # Should be a lambda expression
    assert "\\" in s or "x" in s


def test_pretty_lam_const() -> None:
    # \x -> Zero
    v = VLam(fn=lambda _: VZero())
    s = pretty_value(v)
    assert "Zero" in s


def test_pretty_check_bound() -> None:
    t = Inf(Bound(0))
    assert pretty_check(t, ["x"]) == "x"


def test_pretty_check_lam() -> None:
    t = Lam(Inf(Bound(0)))
    s = pretty_check(t, [])
    assert "\\" in s
    assert "x" in s


def test_pretty_infer_star() -> None:
    assert pretty_infer(Star(0), []) == "Type"


def test_pretty_infer_nat() -> None:
    assert pretty_infer(Nat(), []) == "Nat"
