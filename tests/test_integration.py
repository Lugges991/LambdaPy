"""Integration tests: parse -> type-check -> evaluate end-to-end."""
from __future__ import annotations

import io
import pytest

from lambdapy.errors import TypeCheckError
from lambdapy.parser import parse, AssumeStmt, EvalStmt, LetStmt
from lambdapy.repl import ReplState, execute, fresh_state, run_file
from lambdapy.pretty import pretty_value
from lambdapy.syntax import VNat, VZero, VSucc


def _run(src: str) -> tuple[ReplState, list[str]]:
    """Parse and execute all statements; return (final_state, outputs)."""
    state = fresh_state()
    outputs: list[str] = []
    for stmt in parse(src):
        state, out = execute(state, stmt)
        outputs.append(out)
    return state, outputs


# ---------------------------------------------------------------------------
# Basic expressions
# ---------------------------------------------------------------------------

def test_eval_zero() -> None:
    _, outs = _run("eval Zero")
    assert "Zero" in outs[0]
    assert "Nat" in outs[0]


def test_eval_succ_zero() -> None:
    _, outs = _run("eval Succ Zero")
    # "1 : Nat" or "Succ Zero : Nat"
    assert "Nat" in outs[0]


def test_eval_nat_type() -> None:
    _, outs = _run("eval Nat")
    assert "Type" in outs[0]


def test_eval_star() -> None:
    _, outs = _run("eval Type")
    assert "Type" in outs[0]


# ---------------------------------------------------------------------------
# Assume + use
# ---------------------------------------------------------------------------

def test_assume_and_use() -> None:
    src = "assume n : Nat\neval n"
    _, outs = _run(src)
    assert "n" in outs[1]
    assert "Nat" in outs[1]


def test_assume_multiple() -> None:
    src = "assume True False : Bool\nassume Bool : Type"
    # Note: execute order matters; assume Bool first
    state = fresh_state()
    for stmt in parse("assume Bool : Type\nassume True False : Bool"):
        state, _ = execute(state, stmt)
    assert Global_in_ctx(state, "Bool")
    assert Global_in_ctx(state, "True")
    assert Global_in_ctx(state, "False")


def Global_in_ctx(state: ReplState, name: str) -> bool:
    from lambdapy.syntax import Global
    return any(isinstance(n, Global) and n.name == name for n, _ in state.context)


# ---------------------------------------------------------------------------
# NatElim
# ---------------------------------------------------------------------------

def test_nat_elim_zero() -> None:
    # NatElim (\_.Nat) Zero (\k.\r. Succ r) Zero should evaluate to Zero
    src = "eval NatElim (\\_ -> Nat) Zero (\\k -> \\r -> Succ r) Zero : Nat"
    _, outs = _run(src)
    assert "Zero" in outs[0] or "0" in outs[0]


def test_nat_elim_succ() -> None:
    # NatElim (\_.Nat) Zero (\k.\r. Succ r) (Succ Zero) = Succ Zero = 1
    src = "eval NatElim (\\_ -> Nat) Zero (\\k -> \\r -> Succ r) (Succ Zero) : Nat"
    _, outs = _run(src)
    assert "1" in outs[0] or "Succ" in outs[0]


# ---------------------------------------------------------------------------
# Identity function
# ---------------------------------------------------------------------------

def test_identity_function() -> None:
    # let id : forall (A : Type). A -> A = \A x -> x
    src = "eval (\\A -> \\x -> x) : forall (A : Type). A -> A"
    # Just type-check it
    state, outs = _run(src)
    # Should not raise; result should mention forall or Type
    assert "Type" in outs[0] or "forall" in outs[0] or "->" in outs[0]


# ---------------------------------------------------------------------------
# Vec
# ---------------------------------------------------------------------------

def test_nil_type() -> None:
    _, outs = _run("eval Nil Nat")
    assert "Vec" in outs[0] or "Nil" in outs[0]


def test_cons_type() -> None:
    _, outs = _run("eval Cons Nat Zero Zero (Nil Nat)")
    assert "Vec" in outs[0]


# ---------------------------------------------------------------------------
# Eq / Refl
# ---------------------------------------------------------------------------

def test_refl_type() -> None:
    _, outs = _run("eval Refl Nat Zero")
    assert "Eq" in outs[0]


# ---------------------------------------------------------------------------
# Type errors are raised
# ---------------------------------------------------------------------------

def test_type_error_succ_of_type() -> None:
    with pytest.raises(TypeCheckError):
        _run("eval Succ Type : Nat")


def test_type_error_wrong_annotation() -> None:
    with pytest.raises(TypeCheckError):
        _run("eval Zero : Type")


# ---------------------------------------------------------------------------
# Multi-statement programs
# ---------------------------------------------------------------------------

def test_multi_statement_program() -> None:
    src = """\
assume A : Type
assume x : A
eval x
"""
    state, outs = _run(src)
    assert "x" in outs[2]
    assert "A" in outs[2]


# ---------------------------------------------------------------------------
# REPL output (via StringIO)
# ---------------------------------------------------------------------------

def test_repl_basic() -> None:
    import io
    from lambdapy.repl import run_repl
    inp = io.StringIO("eval Zero\n:quit\n")
    out = io.StringIO()
    run_repl(in_=inp, out=out, prompt="")
    result = out.getvalue()
    assert "Zero" in result or "0" in result


def test_repl_type_error_message() -> None:
    import io
    from lambdapy.repl import run_repl
    inp = io.StringIO("eval Zero : Type\n:quit\n")
    out = io.StringIO()
    run_repl(in_=inp, out=out, prompt="")
    result = out.getvalue()
    assert "error" in result.lower() or "Error" in result


def test_repl_context_command() -> None:
    import io
    from lambdapy.repl import run_repl
    inp = io.StringIO("assume n : Nat\n:context\n:quit\n")
    out = io.StringIO()
    run_repl(in_=inp, out=out, prompt="")
    result = out.getvalue()
    assert "n" in result
    assert "Nat" in result
