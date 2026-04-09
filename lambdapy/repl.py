"""Interactive REPL (read-eval-print loop) and batch execution for LambdaPi."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from lambdapy.check import type_chk, type_inf0
from lambdapy.context import Context, HasType, empty_context
from lambdapy.errors import EvalError, ParseError, TypeCheckError
from lambdapy.eval import eval_chk, eval_inf
from lambdapy.nameenv import with_name_env
from lambdapy.parser import (
    AssumeStmt,
    CheckStmt,
    EvalStmt,
    LetInferStmt,
    LetStmt,
    Statement,
    parse,
)
from lambdapy.pretty import pretty_check, pretty_value
from lambdapy.quote import quote0
from lambdapy.syntax import Free, Global, Inf, Value

# ---------------------------------------------------------------------------
# REPL state
# ---------------------------------------------------------------------------


@dataclass
class ReplState:
    """Mutable interpreter state: typing context + evaluated definitions."""

    context: Context = field(default_factory=empty_context)
    # Evaluated values for `let` definitions (enables reduction through names)
    definitions: dict[str, Value] = field(default_factory=dict)


def fresh_state() -> ReplState:
    return ReplState()


# ---------------------------------------------------------------------------
# Statement execution
# ---------------------------------------------------------------------------


def execute(state: ReplState, stmt: Statement) -> tuple[ReplState, str]:
    """Execute a single statement.  Returns (new_state, output_string)."""
    match stmt:

        case AssumeStmt(names, type_term):
            # Evaluate the type and verify it's a valid type
            with with_name_env(state.definitions):
                ty_val = eval_chk(type_term, [])
                from lambdapy.check import _check_is_type  # type: ignore[attr-defined]
                try:
                    _check_is_type(0, state.context, type_term)
                except TypeCheckError as e:
                    raise TypeCheckError(f"assume: invalid type: {e.msg}") from e
            new_ctx = list(state.context)
            added = []
            for name in names:
                new_ctx = [(Global(name), HasType(ty_val))] + new_ctx
                added.append(name)
            new_state = ReplState(new_ctx, dict(state.definitions))
            ty_str = pretty_check(type_term, [])
            return new_state, f"Assumed: {', '.join(added)} : {ty_str}"

        case LetStmt(name, type_term, body_term):
            with with_name_env(state.definitions):
                # Check the type annotation
                ty_val = eval_chk(type_term, [])
                from lambdapy.check import _check_is_type  # type: ignore[attr-defined]
                _check_is_type(0, state.context, type_term)
                # Check the body against the type
                type_chk(0, state.context, body_term, ty_val)
                # Evaluate the body
                val = eval_chk(body_term, [])
            new_ctx: Context = [(Global(name), HasType(ty_val))] + list(state.context)
            new_defs = {**state.definitions, name: val}
            ty_str = pretty_check(type_term, [])
            return ReplState(new_ctx, new_defs), f"{name} : {ty_str}"

        case LetInferStmt(name, body_term):
            # Haskell-style: let x = t  (type inferred from annotated term)
            with with_name_env(state.definitions):
                ty = type_inf0(state.context, body_term)
                val = eval_inf(body_term, [])
            new_ctx = [(Global(name), HasType(ty))] + list(state.context)
            new_defs = {**state.definitions, name: val}
            ty_str = pretty_value(ty)
            return ReplState(new_ctx, new_defs), f"{name} : {ty_str}"

        case EvalStmt(term):
            with with_name_env(state.definitions):
                ty = type_inf0(state.context, term)
                val = eval_inf(term, [])
            val_str = pretty_value(val)
            ty_str = pretty_value(ty)
            return state, f"{val_str} : {ty_str}"

        case CheckStmt(term):
            match term:
                case Inf(t):
                    with with_name_env(state.definitions):
                        ty = type_inf0(state.context, t)
                    return state, f": {pretty_value(ty)}"
                case _:
                    return state, "check: provide an annotated term like  (expr : Type)"

        case _:  # pragma: no cover
            return state, f"Unknown statement: {stmt!r}"




# ---------------------------------------------------------------------------
# Batch file execution
# ---------------------------------------------------------------------------


def run_file(path: str | Path, out: TextIO = sys.stdout) -> ReplState:
    """Execute all statements in a .lp file.  Returns final state."""
    text = Path(path).read_text()
    state = fresh_state()
    stmts = parse(text)
    for stmt in stmts:
        try:
            state, output = execute(state, stmt)
            print(output, file=out)
        except (TypeCheckError, EvalError, ParseError) as exc:
            print(f"Error: {exc}", file=out)
            raise
    return state


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

_HELP = """\
LambdaPi REPL commands:
  :help          Show this help
  :context       List all assumptions and definitions
  :reset         Reset to empty state
  :load FILE     Load and execute a .lp file
  :quit          Exit the REPL
  (or any LambdaPi statement)
"""


def run_repl(
    in_: TextIO = sys.stdin,
    out: TextIO = sys.stdout,
    prompt: str = "λΠ> ",
) -> None:
    """Run the interactive REPL."""
    state = fresh_state()
    print("LambdaPi REPL  (:help for commands)", file=out)

    while True:
        try:
            try:
                line = input(prompt) if in_ is sys.stdin else in_.readline()
            except EOFError:
                print("\nBye.", file=out)
                break

            if not line:
                continue
            line = line.strip()
            if not line:
                continue

            # Meta-commands
            if line.startswith(":"):
                parts = line.split(None, 1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""
                match cmd:
                    case ":help":
                        print(_HELP, file=out)
                    case ":context":
                        _print_context(state, out)
                    case ":reset":
                        state = fresh_state()
                        print("Context reset.", file=out)
                    case ":load":
                        if not arg:
                            print("Usage: :load FILE", file=out)
                        else:
                            try:
                                state = run_file(arg, out)
                            except Exception as exc:
                                print(f"Load failed: {exc}", file=out)
                    case ":quit" | ":q":
                        print("Bye.", file=out)
                        break
                    case _:
                        print(f"Unknown command: {cmd}  (:help for commands)", file=out)
                continue

            # Parse and execute
            try:
                stmts = parse(line)
                for stmt in stmts:
                    state, output = execute(state, stmt)
                    print(output, file=out)
            except ParseError as exc:
                print(f"Parse error: {exc.msg}", file=out)
            except TypeCheckError as exc:
                print(f"Type error: {exc.msg}", file=out)
            except EvalError as exc:
                print(f"Eval error: {exc.msg}", file=out)
            except Exception as exc:
                print(f"Error: {exc}", file=out)

        except KeyboardInterrupt:
            print("\n(Interrupt — type :quit to exit)", file=out)


def _print_context(state: ReplState, out: TextIO) -> None:
    if not state.context:
        print("(empty context)", file=out)
        return
    for name, entry in reversed(state.context):
        match entry:
            case HasType(ty):
                ty_str = pretty_value(ty)
                match name:
                    case Global(n):
                        val_info = ""
                        if n in state.definitions:
                            val_str = pretty_value(state.definitions[n])
                            val_info = f" = {val_str}"
                        print(f"  {n} : {ty_str}{val_info}", file=out)
                    case _:
                        print(f"  {name!r} : {ty_str}", file=out)
