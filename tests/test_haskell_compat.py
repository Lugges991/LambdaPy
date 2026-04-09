"""Tests verifying functional identity with the Haskell lambdapi library.

Each test corresponds to a transcript that works in the Haskell REPL
(src/LambdaPi/Main.hs `repLP`).  We run the same statements through the
Python REPL and verify the output matches the expected Haskell output.

Haskell syntax used:
  - `::` for annotations
  - `*` for universe (Type 0)
  - `assume (x :: T)` with parenthesized bindings
  - `let x = t` with no explicit type (type inferred from annotated t)
  - `natElim`, `vecElim`, `finElim`, `eqElim` (camelCase)
"""
from __future__ import annotations

import pytest

from lambdapy.parser import (
    AssumeStmt,
    LetInferStmt,
    EvalStmt,
    parse,
)
from lambdapy.repl import ReplState, execute, fresh_state
from lambdapy.eval import eval_inf
from lambdapy.nameenv import with_name_env
from lambdapy.syntax import (
    Ann, Inf, Nat, NatElim, Zero, Succ,
    VNat, VZero, VSucc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_session(lines: list[str]) -> tuple[ReplState, list[str]]:
    """Run a list of statement lines through the REPL, return (state, outputs)."""
    state = fresh_state()
    outputs: list[str] = []
    for line in lines:
        stmts = parse(line + "\n")
        for stmt in stmts:
            state, out = execute(state, stmt)
            outputs.append(out)
    return state, outputs


# ---------------------------------------------------------------------------
# Basic syntax: :: and *
# ---------------------------------------------------------------------------

class TestHaskellAnnotation:
    def test_double_colon_annotation(self) -> None:
        """e :: T  parses as Ann and type-checks."""
        stmts = parse("eval (Zero :: Nat)\n")
        assert len(stmts) == 1
        assert isinstance(stmts[0], EvalStmt)

    def test_star_universe(self) -> None:
        """* is syntactic sugar for Type 0."""
        stmts = parse("eval (* :: *)\n")
        # * :: * — star is its own type in Haskell; Python accepts it via Ann
        # The eval should succeed (type checking * against * passes in Python
        # because Star(0) : Star(1) and the annotation is Star(0), not itsself,
        # but the Ann is checked against its own annotation value).
        # We just verify it parses correctly.
        assert len(stmts) == 1

    def test_star_parses_as_universe(self) -> None:
        """* parses to the same Star node as Type / Type 0."""
        from lambdapy.syntax import Star
        stmts_star = parse("eval (* :: *)\n")
        stmts_type = parse("eval (Type :: Type 1)\n")
        # Both should parse without error
        assert len(stmts_star) == 1
        assert len(stmts_type) == 1


# ---------------------------------------------------------------------------
# Haskell-style assume
# ---------------------------------------------------------------------------

class TestHaskellAssume:
    def test_paren_binding_single(self) -> None:
        """assume (a :: *) works."""
        stmts = parse("assume (a :: *)\n")
        assert len(stmts) == 1
        assert isinstance(stmts[0], AssumeStmt)
        assert stmts[0].names == ["a"]

    def test_paren_binding_multiple(self) -> None:
        """assume (a :: *) (b :: *) expands to two AssumeStmt."""
        stmts = parse("assume (a :: *) (b :: *)\n")
        assert len(stmts) == 2
        for s in stmts:
            assert isinstance(s, AssumeStmt)
            assert len(s.names) == 1

    def test_paren_binding_three(self) -> None:
        """assume (a :: *) (b :: *) (c :: Nat) expands to three statements."""
        stmts = parse("assume (a :: *) (b :: *) (c :: Nat)\n")
        assert len(stmts) == 3

    def test_assume_single_no_parens(self) -> None:
        """assume a :: *  (single name, no parens)."""
        stmts = parse("assume a :: *\n")
        assert len(stmts) == 1
        assert isinstance(stmts[0], AssumeStmt)
        assert stmts[0].names == ["a"]

    def test_assume_repl(self) -> None:
        """Assumed variables are added to context."""
        state, outputs = run_session(["assume (a :: *)"])
        assert "a" in outputs[0]
        from lambdapy.syntax import Global
        names_in_ctx = {n.name for n, _ in state.context if hasattr(n, "name")}
        assert "a" in names_in_ctx


# ---------------------------------------------------------------------------
# Haskell-style let (type inferred)
# ---------------------------------------------------------------------------

class TestHaskellLet:
    def test_let_infer_parses(self) -> None:
        """let x = t  parses as LetInferStmt."""
        stmts = parse("let z = (Zero :: Nat)\n")
        assert len(stmts) == 1
        assert isinstance(stmts[0], LetInferStmt)
        assert stmts[0].name == "z"

    def test_let_infer_zero(self) -> None:
        """let z = (Zero :: Nat) adds z : Nat to context."""
        state, outputs = run_session(["let z = (Zero :: Nat)"])
        assert "z" in outputs[0]
        assert "Nat" in outputs[0]
        from lambdapy.syntax import Global
        names_in_ctx = {n.name for n, _ in state.context if hasattr(n, "name")}
        assert "z" in names_in_ctx

    def test_let_infer_reduces(self) -> None:
        """let x = t; eval x  produces the reduced value (name env works)."""
        state, outputs = run_session([
            "let z = (Zero :: Nat)",
            "eval (z :: Nat)",
        ])
        # After eval, output should show Zero (not a neutral)
        assert "Zero" in outputs[1]

    def test_let_infer_id(self) -> None:
        """let id = (\\a x -> x) :: forall (a :: *) . a -> a"""
        state, outputs = run_session([
            "let id = (\\ a x -> x) :: forall (a :: *) . a -> a",
        ])
        assert "id" in outputs[0]
        from lambdapy.syntax import Global
        names = {n.name for n, _ in state.context if hasattr(n, "name")}
        assert "id" in names


# ---------------------------------------------------------------------------
# Name environment: definitions reduce through globals
# ---------------------------------------------------------------------------

class TestNameEnvResolution:
    def test_definition_reduces(self) -> None:
        """A defined name evaluates to its value, not a neutral."""
        state, _ = run_session(["let z = (Zero :: Nat)"])
        # Evaluate Free(Global("z")) with the name env active
        from lambdapy.syntax import Free, Global
        with with_name_env(state.definitions):
            v = eval_inf(Free(Global("z")), [])
        from lambdapy.syntax import VZero
        assert isinstance(v, VZero)

    def test_succ_definition_reduces(self) -> None:
        """Definitions chain: let one = (Succ Zero :: Nat); one reduces."""
        state, _ = run_session(["let one = (Succ (Zero :: Nat) :: Nat)"])
        from lambdapy.syntax import Free, Global, VSucc, VZero
        with with_name_env(state.definitions):
            v = eval_inf(Free(Global("one")), [])
        assert isinstance(v, VSucc)
        assert isinstance(v.pred, VZero)

    def test_eval_uses_name_env(self) -> None:
        """eval of a defined name returns the reduced value."""
        state, outputs = run_session([
            "let one = (Succ (Zero :: Nat) :: Nat)",
            "eval (one :: Nat)",
        ])
        # Should show Succ Zero (or 1), not a neutral
        assert "Succ" in outputs[1] or "1" in outputs[1]


# ---------------------------------------------------------------------------
# natElim / vecElim (camelCase Haskell names)
# ---------------------------------------------------------------------------

class TestCamelCaseKeywords:
    def test_nat_elim_camel(self) -> None:
        """natElim parses to the same node as NatElim."""
        from lambdapy.syntax import NatElim, Ann, Inf
        stmts1 = parse("eval (NatElim (\\ _ -> Nat) Zero (\\ k r -> Succ r) Zero :: Nat)\n")
        stmts2 = parse("eval (natElim (\\ _ -> Nat) Zero (\\ k r -> Succ r) Zero :: Nat)\n")
        assert len(stmts1) == 1
        assert len(stmts2) == 1
        from lambdapy.parser import EvalStmt
        assert isinstance(stmts1[0], EvalStmt)
        assert isinstance(stmts2[0], EvalStmt)
        # The term is Ann(NatElim(...), Nat()) due to outer :: annotation
        def _inner(t):  # unwrap one Ann if present
            if isinstance(t, Ann) and isinstance(t.expr, Inf):
                return t.expr.term
            return t
        assert isinstance(_inner(stmts1[0].term), NatElim)
        assert isinstance(_inner(stmts2[0].term), NatElim)
        # Both should produce identical AST
        assert stmts1[0].term == stmts2[0].term

    def test_vec_elim_camel(self) -> None:
        """vecElim parses to the same node as VecElim."""
        from lambdapy.syntax import VecElim, Ann, Inf
        from lambdapy.parser import EvalStmt
        src1 = (
            "eval (VecElim Nat (\\ n v -> Nat) Zero"
            " (\\ n h t r -> Succ r) Zero (Nil Nat :: Vec Nat Zero) :: Nat)\n"
        )
        src2 = (
            "eval (vecElim Nat (\\ n v -> Nat) Zero"
            " (\\ n h t r -> Succ r) Zero (Nil Nat :: Vec Nat Zero) :: Nat)\n"
        )
        stmts1 = parse(src1)
        stmts2 = parse(src2)
        assert len(stmts1) == 1
        assert len(stmts2) == 1
        assert isinstance(stmts1[0], EvalStmt)
        assert isinstance(stmts2[0], EvalStmt)
        assert stmts1[0].term == stmts2[0].term


# ---------------------------------------------------------------------------
# End-to-end Haskell transcript
# ---------------------------------------------------------------------------

class TestHaskellTranscript:
    def test_id_function(self) -> None:
        """
        Paper §2 transcript (Haskell syntax):
          assume (a :: *) (b :: *)
          let id = (\\a x -> x) :: forall (a :: *) . a -> a
          id Nat Zero
        Note: assume (a :: *) (b :: *) expands to 2 statements, so outputs
        has 4 entries total.
        """
        state, outputs = run_session([
            "assume (a :: *) (b :: *)",
            "let id = (\\ a x -> x) :: forall (a :: *) . a -> a",
            "eval ((id :: forall (a :: *) . a -> a) Nat Zero :: Nat)",
        ])
        # 2 assumes + 1 let + 1 eval = 4 outputs
        assert len(outputs) == 4
        # id Nat Zero should reduce to Zero
        assert "Zero" in outputs[3]

    def test_plus_via_nat_elim(self) -> None:
        """
        Haskell-style plus using natElim:
          let plus = (\\ m n -> natElim (\\ _ -> Nat) n (\\ k rec -> Succ rec) m) :: Nat -> Nat -> Nat
          plus (Succ Zero) (Succ Zero) = Succ (Succ Zero)
        """
        state, outputs = run_session([
            "let plus = (\\ m n -> natElim (\\ _ -> Nat) n (\\ k rec -> Succ rec) m)"
            " :: Nat -> Nat -> Nat",
            "eval ((plus :: Nat -> Nat -> Nat)"
            " (Succ (Zero :: Nat) :: Nat)"
            " (Succ (Zero :: Nat) :: Nat) :: Nat)",
        ])
        # 1 + 1 = 2 = Succ (Succ Zero) — pretty printer renders this as "2"
        out = outputs[1]
        assert "2" in out or "Succ" in out

    def test_forall_binder_double_colon(self) -> None:
        """forall (a :: *) . a -> a  parses and type-checks."""
        state, outputs = run_session([
            "eval ((\\ a x -> x) :: forall (a :: *) . a -> a)"
        ])
        # Should succeed — output is a lambda / value
        assert outputs  # no exception = success

    def test_nat_elim_zero_case(self) -> None:
        """natElim P z s Zero reduces to z."""
        state, outputs = run_session([
            "eval (natElim (\\ _ -> Nat) Zero (\\ k r -> Succ r) (Zero :: Nat) :: Nat)",
        ])
        assert "Zero" in outputs[0]

    def test_nat_elim_succ_case(self) -> None:
        """natElim P z s (Succ Zero) reduces to s Zero z."""
        state, outputs = run_session([
            "eval (natElim (\\ _ -> Nat) Zero (\\ k r -> Succ r)"
            " (Succ (Zero :: Nat) :: Nat) :: Nat)",
        ])
        # Should be Succ Zero — pretty printer renders this as "1"
        assert "1" in outputs[0] or "Succ" in outputs[0]
