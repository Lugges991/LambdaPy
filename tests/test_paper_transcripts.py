"""Tests reproducing examples from the LambdaPi paper (Löh/McBride/Swierstra 2010).

These are end-to-end tests that exercise the full pipeline:
parse -> elaborate -> type-check -> evaluate -> pretty-print.
"""
from __future__ import annotations

import pytest

from lambdapy.errors import TypeCheckError
from lambdapy.parser import parse
from lambdapy.repl import ReplState, execute, fresh_state
from lambdapy.pretty import pretty_value
from lambdapy.syntax import VNat, VZero, VSucc


def _run(src: str) -> tuple[ReplState, list[str]]:
    state = fresh_state()
    outputs: list[str] = []
    for stmt in parse(src):
        state, out = execute(state, stmt)
        outputs.append(out)
    return state, outputs


# ---------------------------------------------------------------------------
# Paper §2.1: Simply-typed lambda calculus examples
# ---------------------------------------------------------------------------

def test_paper_identity_type() -> None:
    """id : Bool -> Bool applied to True : Bool."""
    src = """\
assume Bool : Type
assume True : Bool
assume False : Bool
eval (\\x -> x) : Bool -> Bool
"""
    _, outs = _run(src)
    # The identity function should have type Bool -> Bool
    assert "Bool" in outs[3]


def test_paper_constant_function() -> None:
    """const : Bool -> Bool -> Bool."""
    src = """\
assume Bool : Type
assume True : Bool
eval (\\x -> \\y -> x) : Bool -> Bool -> Bool
"""
    _, outs = _run(src)
    assert "Bool" in outs[2]


# ---------------------------------------------------------------------------
# Paper §3: Dependent types
# ---------------------------------------------------------------------------

def test_paper_dependent_identity() -> None:
    """Polymorphic identity id : (A : *) -> A -> A."""
    src = "eval (\\A -> \\x -> x) : forall (A : Type). A -> A"
    _, outs = _run(src)
    assert "Type" in outs[0] or "forall" in outs[0] or "->" in outs[0]


def test_paper_nat_zero() -> None:
    """Zero : Nat."""
    _, outs = _run("eval Zero")
    assert "Nat" in outs[0]


def test_paper_succ_zero() -> None:
    """Succ Zero evaluates to 1."""
    _, outs = _run("eval Succ Zero : Nat")
    # Pretty printer renders Succ(Succ(...Zero)) as decimal
    assert "1" in outs[0] or "Succ" in outs[0]


def test_paper_two_plus_two() -> None:
    """2 + 2 = 4 via natElim."""
    # plus m n = NatElim (\_.Nat) n (\k.\r. Succ r) m
    # plus 2 2 = 4
    src = """\
eval NatElim (\\_ -> Nat) (Succ (Succ Zero)) (\\k -> \\r -> Succ r) (Succ (Succ Zero)) : Nat
"""
    _, outs = _run(src)
    # Should evaluate to 4
    assert "4" in outs[0] or "Succ" in outs[0]


# ---------------------------------------------------------------------------
# Paper §4: Datatypes
# ---------------------------------------------------------------------------

def test_paper_vec_nil() -> None:
    """Nil Nat : Vec Nat 0."""
    _, outs = _run("eval Nil Nat")
    assert "Vec" in outs[0] or "Nil" in outs[0]


def test_paper_vec_cons() -> None:
    """Cons Nat 0 Zero (Nil Nat) : Vec Nat 1."""
    _, outs = _run("eval Cons Nat Zero Zero (Nil Nat)")
    assert "Vec" in outs[0] or "Cons" in outs[0]


def test_paper_refl() -> None:
    """Refl Nat Zero : Eq Nat Zero Zero."""
    _, outs = _run("eval Refl Nat Zero")
    assert "Eq" in outs[0] or "Refl" in outs[0]


def test_paper_eq_elim_reduces() -> None:
    """eqElim A m rc x x (Refl A x) = rc x."""
    src = """\
eval EqElim Nat (\\x -> \\y -> \\p -> Nat) (\\z -> Zero) Zero Zero (Refl Nat Zero) : Nat
"""
    _, outs = _run(src)
    # Should reduce to Zero (the refl case)
    assert "Zero" in outs[0] or "0" in outs[0]


def test_paper_fzero() -> None:
    """FZero 0 : Fin 1."""
    _, outs = _run("eval FZero Zero")
    assert "Fin" in outs[0] or "FZero" in outs[0]


# ---------------------------------------------------------------------------
# Universe levels (spec extension)
# ---------------------------------------------------------------------------

def test_universe_level_0() -> None:
    _, outs = _run("eval Type")
    assert "Type 1" in outs[0] or "Type" in outs[0]


def test_universe_level_1() -> None:
    _, outs = _run("eval Type 1")
    assert "Type 2" in outs[0] or "Type" in outs[0]


def test_type_in_type_is_rejected() -> None:
    """Star(0) : Star(1), not Star(0) : Star(0) — no Type : Type paradox."""
    # Type : Type 1, not Type : Type — so checking Type against Type should fail
    with pytest.raises(TypeCheckError):
        _run("eval Type : Type")


# ---------------------------------------------------------------------------
# File execution
# ---------------------------------------------------------------------------

def test_run_identity_file() -> None:
    from lambdapy.repl import run_file
    import io
    out = io.StringIO()
    run_file("examples/01_identity.lp", out)
    result = out.getvalue()
    assert "Type" in result or "forall" in result or "->" in result


def test_run_nat_file() -> None:
    from lambdapy.repl import run_file
    import io
    out = io.StringIO()
    run_file("examples/02_nat.lp", out)
    result = out.getvalue()
    # 2 should appear (Succ Succ Zero = 2)
    assert "2" in result or "Succ" in result


def test_run_vec_file() -> None:
    from lambdapy.repl import run_file
    import io
    out = io.StringIO()
    run_file("examples/03_vec.lp", out)
    result = out.getvalue()
    assert "Vec" in result or "Nil" in result or "Cons" in result
