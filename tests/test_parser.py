"""Tests for the parser and elaborator."""
from __future__ import annotations

import pytest
from lambdapy.syntax import (
    Ann, App, Bound, Free, Global, Inf, Lam, Nat, NatElim, Pi,
    Star, Succ, Vec, Zero,
)
from lambdapy.parser import (
    AssumeStmt, CheckStmt, EvalStmt, LetStmt, parse,
)
from lambdapy.errors import ParseError


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

def test_assume_single() -> None:
    stmts = parse("assume Bool : Type")
    assert len(stmts) == 1
    s = stmts[0]
    assert isinstance(s, AssumeStmt)
    assert s.names == ["Bool"]
    assert s.type_term == Inf(Star(0))


def test_assume_multiple_names() -> None:
    stmts = parse("assume True False : Bool")
    assert isinstance(stmts[0], AssumeStmt)
    assert stmts[0].names == ["True", "False"]


def test_eval_zero() -> None:
    stmts = parse("eval Zero")
    assert isinstance(stmts[0], EvalStmt)
    assert stmts[0].term == Zero()


def test_eval_annotated() -> None:
    stmts = parse("eval Zero : Nat")
    assert isinstance(stmts[0], EvalStmt)
    assert stmts[0].term == Ann(Inf(Zero()), Inf(Nat()))


def test_let_stmt() -> None:
    stmts = parse("let x : Nat = Zero")
    assert isinstance(stmts[0], LetStmt)
    s = stmts[0]
    assert s.name == "x"
    assert s.type_term == Inf(Nat())
    assert s.body_term == Inf(Zero())


def test_check_stmt() -> None:
    stmts = parse("check Zero : Nat")
    # "check" is a CheckStmt; but the grammar says check parses a term
    # which could be Ann(Zero, Nat)
    assert isinstance(stmts[0], CheckStmt)


# ---------------------------------------------------------------------------
# Term elaboration
# ---------------------------------------------------------------------------

def test_variable_lookup_in_scope() -> None:
    # \x -> x elaborates to Lam(Inf(Bound(0)))
    stmts = parse("eval (\\x -> x) : Nat -> Nat")
    s = stmts[0]
    assert isinstance(s, EvalStmt)
    assert isinstance(s.term, Ann)
    assert s.term.expr == Lam(Inf(Bound(0)))


def test_variable_free() -> None:
    stmts = parse("eval x")
    assert isinstance(stmts[0], EvalStmt)
    assert stmts[0].term == Free(Global("x"))


def test_lambda_multi_arg() -> None:
    # \x y -> x  elaborates to  Lam(Lam(Inf(Bound(1))))
    stmts = parse("eval (\\x y -> x) : Nat -> Nat -> Nat")
    s = stmts[0]
    assert isinstance(s, EvalStmt)
    term = s.term
    assert isinstance(term, Ann)
    assert term.expr == Lam(Lam(Inf(Bound(1))))


def test_arrow_type() -> None:
    # Nat -> Nat parses as Pi(Inf(Nat()), Inf(Nat()))
    stmts = parse("assume f : Nat -> Nat")
    s = stmts[0]
    assert isinstance(s, AssumeStmt)
    assert s.type_term == Inf(Pi(Inf(Nat()), Inf(Nat())))


def test_arrow_right_associative() -> None:
    # Nat -> Nat -> Nat = Nat -> (Nat -> Nat)
    stmts = parse("assume f : Nat -> Nat -> Nat")
    s = stmts[0]
    assert isinstance(s, AssumeStmt)
    expected = Inf(Pi(Inf(Nat()), Inf(Pi(Inf(Nat()), Inf(Nat())))))
    assert s.type_term == expected


def test_universe_level() -> None:
    stmts = parse("assume A : Type 1")
    assert stmts[0].type_term == Inf(Star(1))  # type: ignore[union-attr]


def test_universe_level_zero() -> None:
    stmts = parse("assume A : Type")
    assert stmts[0].type_term == Inf(Star(0))  # type: ignore[union-attr]


def test_application_left_assoc() -> None:
    # f a b parses as App(App(f, a), b)
    stmts = parse("eval f a b")
    s = stmts[0]
    assert isinstance(s, EvalStmt)
    assert isinstance(s.term, App)
    assert isinstance(s.term.func, App)
    assert s.term.func.func == Free(Global("f"))


def test_forall_binder() -> None:
    # forall (x : Nat). Nat
    stmts = parse("assume f : forall (x : Nat). Nat")
    s = stmts[0]
    assert isinstance(s, AssumeStmt)
    assert s.type_term == Inf(Pi(Inf(Nat()), Inf(Nat())))


def test_dependent_pi() -> None:
    # forall (n : Nat). Vec Nat n
    stmts = parse("assume f : forall (n : Nat). Vec Nat n")
    s = stmts[0]
    assert isinstance(s, AssumeStmt)
    # The Vec application with Nat and Bound(0)
    ty = s.type_term
    assert isinstance(ty, Inf)
    assert isinstance(ty.term, Pi)


def test_multiple_stmts() -> None:
    src = "assume Nat : Type\nassume Zero : Nat"
    stmts = parse(src)
    assert len(stmts) == 2


def test_parse_error() -> None:
    with pytest.raises(ParseError):
        parse("this is not valid syntax @@@@")
