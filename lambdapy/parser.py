"""Surface syntax parser and elaborator for LambdaPi.

parse(text) -> list[Statement]

The elaborator converts the parse tree into de Bruijn core terms:
- Named binders (lambda / forall / arrow) are elaborated by maintaining a
  scope stack (list[str], innermost first).
- A name found at position k in the scope becomes Bound(k).
- A name not in scope becomes Free(Global(name)).

Statement types:
  AssumeStmt(names, type_term): assume multiple names with one type annotation
  LetStmt(name, type_term, body_term): define a name
  EvalStmt(term): evaluate and print
  CheckStmt(term): type-check and print type
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lark import Lark, Token, Tree

from lambdapy.errors import ParseError
from lambdapy.syntax import (
    Ann, App, Bound, CheckTerm, Cons, Eq, EqElim, Fin, FSucc, FZero,
    FinElim, Free, Global, Inf, InferTerm, Lam, Nat, NatElim, Nil, Pi,
    Refl, Star, Succ, Vec, VecElim, Zero,
)


# ---------------------------------------------------------------------------
# Statement types
# ---------------------------------------------------------------------------


@dataclass
class AssumeStmt:
    """assume name1 name2 ... : type_term"""
    names: list[str]
    type_term: CheckTerm


@dataclass
class LetStmt:
    """let name : type = body"""
    name: str
    type_term: CheckTerm
    body_term: CheckTerm


@dataclass
class LetInferStmt:
    """let name = body  (type inferred from body, which must be inferable)"""
    name: str
    body_term: InferTerm


@dataclass
class EvalStmt:
    """eval term — evaluates an InferTerm (must be annotated or inferable)"""
    term: InferTerm


@dataclass
class CheckStmt:
    """check term — type-checks a CheckTerm"""
    term: CheckTerm


Statement = AssumeStmt | LetStmt | LetInferStmt | EvalStmt | CheckStmt


# ---------------------------------------------------------------------------
# Grammar loading
# ---------------------------------------------------------------------------

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


def _make_parser() -> Lark:
    grammar = _GRAMMAR_PATH.read_text()
    return Lark(grammar, parser="earley", ambiguity="resolve")


_PARSER: Lark | None = None


def _get_parser() -> Lark:
    global _PARSER
    if _PARSER is None:
        _PARSER = _make_parser()
    return _PARSER


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(text: str) -> list[Statement]:
    """Parse LambdaPi source text and return a list of statements.

    Each statement must end with a newline.  The text is normalised so that
    every non-blank line ends with exactly one newline.
    """
    p = _get_parser()
    # Normalise: keep only non-blank, non-comment lines, each ending with \n
    lines = text.splitlines()
    kept = [
        line for line in lines
        if line.strip() and not line.strip().startswith("--")
    ]
    normalised = "\n".join(kept) + "\n"
    try:
        tree = p.parse(normalised)
    except Exception as exc:
        raise ParseError(str(exc)) from exc
    return _elab_start(tree)


# ---------------------------------------------------------------------------
# Tree elaboration
# ---------------------------------------------------------------------------


def _elab_start(tree: Tree) -> list[Statement]:  # type: ignore[type-arg]
    stmts: list[Statement] = []
    for child in tree.children:
        if isinstance(child, Tree):
            stmts.extend(_elab_statement(child))
    return stmts


def _elab_statement(tree: Tree) -> list[Statement]:  # type: ignore[type-arg]
    match tree.data:
        # Python-style: assume x y z : T  (all names share one type)
        case "assume_stmt":
            name_list_tree, type_tree = tree.children[0], tree.children[1]
            names = [str(c) for c in name_list_tree.children]
            ty = _elab_check(type_tree, [])
            return [AssumeStmt(names, ty)]

        # Haskell-style: assume (a :: *) (b :: *)  — one statement per binding
        case "assume_multi_stmt":
            stmts: list[Statement] = []
            for child in tree.children:
                if isinstance(child, Tree) and child.data == "paren_binding":
                    name = str(child.children[0])
                    # children[1] may be the "::" token; type is last child
                    type_tree = child.children[-1]
                    ty = _elab_check(type_tree, [])
                    stmts.append(AssumeStmt([name], ty))
            return stmts

        # Haskell-style: assume a :: *  (single name, no parens)
        case "assume_single_stmt":
            name = str(tree.children[0])
            # children[1] may be the "::" token; type is last non-token child
            type_tree = next(
                c for c in reversed(tree.children) if isinstance(c, Tree)
            )
            ty = _elab_check(type_tree, [])
            return [AssumeStmt([name], ty)]

        # Python-style: let x : T = body
        case "let_stmt":
            name = str(tree.children[0])
            type_tree = tree.children[1]
            body_tree = tree.children[2]
            ty = _elab_check(type_tree, [])
            body = _elab_check(body_tree, [])
            return [LetStmt(name, ty, body)]

        # Haskell-style: let x = t  (t must be inferable, e.g. annotated)
        case "let_infer_stmt":
            name = str(tree.children[0])
            body_tree = tree.children[1]
            body = _elab_infer(body_tree, [])
            return [LetInferStmt(name, body)]

        case "eval_stmt":
            term = _elab_infer(tree.children[0], [])
            return [EvalStmt(term)]

        case "check_stmt":
            term = _elab_check(tree.children[0], [])
            return [CheckStmt(term)]

        case _:
            raise ParseError(f"Unknown statement type: {tree.data!r}")


# ---------------------------------------------------------------------------
# Term elaboration
# ---------------------------------------------------------------------------

def _elab_check(tree: Tree, scope: list[str]) -> CheckTerm:  # type: ignore[type-arg]
    """Elaborate a parse tree node to a CheckTerm."""
    if isinstance(tree, Token):
        # Bare token — treat as variable
        return Inf(_elab_var(str(tree), scope))

    match tree.data:
        case "term":
            # `term` is a transparent wrapper — recurse into single child
            return _elab_check(tree.children[0], scope)

        case "lam_term":
            names = [str(c) for c in tree.children[:-1] if isinstance(c, Token)]
            body_tree = tree.children[-1]
            return _elab_lam(names, body_tree, scope)

        case "forall_term":
            binders = [c for c in tree.children if isinstance(c, Tree) and c.data == "binder"]
            body_tree = tree.children[-1]
            return _elab_forall(binders, body_tree, scope)

        case "arrow":
            domain_tree, range_tree = tree.children
            domain = _elab_check(domain_tree, scope)
            range_ = _elab_check(range_tree, ["_"] + scope)
            return Inf(Pi(domain, range_))

        case "ann":
            expr_tree, type_tree = tree.children
            expr = _elab_check(expr_tree, scope)
            type_ = _elab_check(type_tree, scope)
            return Inf(Ann(expr, type_))

        case "app_chain":
            # Single-element app_chain: don't force inference (lambda is OK)
            if len(tree.children) == 1:
                return _elab_check(tree.children[0], scope)
            return Inf(_elab_app_chain(tree.children, scope))

        case "paren":
            return _elab_check(tree.children[0], scope)

        case "universe":
            return Inf(_elab_universe(tree))

        case "nat":
            return Inf(Nat())

        case "zero":
            return Inf(Zero())

        case "succ_kw":
            return Inf(_make_succ_fn())

        case "nat_elim_kw":
            return Inf(_make_nat_elim_fn())

        case "vec_kw":
            return Inf(_make_vec_fn())

        case "nil_kw":
            return Inf(_make_nil_fn())

        case "cons_kw":
            return Inf(_make_cons_fn())

        case "vec_elim_kw":
            return Inf(_make_vec_elim_fn())

        case "fin_kw":
            return Inf(_make_fin_fn())

        case "fzero_kw":
            return Inf(_make_fzero_fn())

        case "fsucc_kw":
            return Inf(_make_fsucc_fn())

        case "fin_elim_kw":
            return Inf(_make_fin_elim_fn())

        case "eq_kw":
            return Inf(_make_eq_fn())

        case "refl_kw":
            return Inf(_make_refl_fn())

        case "eq_elim_kw":
            return Inf(_make_eq_elim_fn())

        case "var":
            name = str(tree.children[0])
            return Inf(_elab_var(name, scope))

        case _:
            raise ParseError(f"_elab_check: unexpected node {tree.data!r}")


def _elab_infer(tree: Tree, scope: list[str]) -> InferTerm:  # type: ignore[type-arg]
    result = _elab_check(tree, scope)
    match result:
        case Inf(t):
            return t
        case _:
            raise ParseError(
                "Expected an inferable term; wrap lambdas with a type annotation: (\\x -> ...) : T"
            )


# ---------------------------------------------------------------------------
# Lambda / forall helpers
# ---------------------------------------------------------------------------

def _elab_lam(names: list[str], body_tree: Tree, scope: list[str]) -> CheckTerm:  # type: ignore[type-arg]
    if not names:
        return _elab_check(body_tree, scope)
    inner = _elab_lam(names[1:], body_tree, [names[0]] + scope)
    return Lam(inner)


def _elab_forall(binders: list[Tree], body_tree: Tree, scope: list[str]) -> CheckTerm:  # type: ignore[type-arg]
    if not binders:
        return _elab_check(body_tree, scope)
    binder = binders[0]
    name = str(binder.children[0])
    domain_tree = binder.children[1]
    domain = _elab_check(domain_tree, scope)
    inner = _elab_forall(binders[1:], body_tree, [name] + scope)
    return Inf(Pi(domain, inner))


# ---------------------------------------------------------------------------
# Application chain
# ---------------------------------------------------------------------------

# Built-in arities: number of arguments that produce the direct constructor
_BUILTIN_ARITIES: dict[str, int] = {
    "succ_kw": 1,
    "nat_elim_kw": 4,
    "vec_kw": 2,
    "nil_kw": 1,
    "cons_kw": 4,
    "vec_elim_kw": 6,
    "fin_kw": 1,
    "fzero_kw": 1,
    "fsucc_kw": 2,
    "fin_elim_kw": 5,
    "eq_kw": 3,
    "refl_kw": 2,
    "eq_elim_kw": 6,
}


def _elab_app_chain(children: list, scope: list[str]) -> InferTerm:  # type: ignore[type-arg]
    """Left-fold application: f a b c → App(App(App(f,a),b),c).

    For built-in keywords that have a known arity, consume exactly that many
    arguments and construct the InferTerm directly (avoiding fake Ann stubs).
    Any remaining arguments are folded as normal Apps.
    """
    if not children:
        raise ParseError("Empty application chain")

    first = children[0]
    rest = children[1:]

    # Check if first child is a built-in keyword with enough arguments
    first_kind = _get_kind(first)
    if first_kind in _BUILTIN_ARITIES:
        arity = _BUILTIN_ARITIES[first_kind]
        if len(rest) >= arity:
            args = [_elab_check(rest[i], scope) for i in range(arity)]
            result: InferTerm = _builtin_apply(first_kind, args)
            # Apply remaining arguments as normal Apps
            for child in rest[arity:]:
                result = App(result, _elab_check(child, scope))
            return result

    # Normal case: left-fold
    result = _elab_atom_infer(first, scope)
    for child in rest:
        arg = _elab_check(child, scope)
        result = App(result, arg)
    return result


def _get_kind(tree: "Tree") -> str:  # type: ignore[type-arg]
    """Return the tree.data if it's a Tree, else empty string."""
    from lark import Tree as LarkTree
    if isinstance(tree, LarkTree):
        return tree.data  # type: ignore[return-value]
    return ""


def _builtin_apply(kind: str, args: list[CheckTerm]) -> InferTerm:
    """Construct the appropriate InferTerm from a built-in keyword + args."""
    match kind:
        case "succ_kw":
            return Succ(args[0])
        case "nat_elim_kw":
            return NatElim(args[0], args[1], args[2], args[3])
        case "vec_kw":
            return Vec(args[0], args[1])
        case "nil_kw":
            return Nil(args[0])
        case "cons_kw":
            return Cons(args[0], args[1], args[2], args[3])
        case "vec_elim_kw":
            return VecElim(args[0], args[1], args[2], args[3], args[4], args[5])
        case "fin_kw":
            return Fin(args[0])
        case "fzero_kw":
            return FZero(args[0])
        case "fsucc_kw":
            return FSucc(args[0], args[1])
        case "fin_elim_kw":
            return FinElim(args[0], args[1], args[2], args[3], args[4])
        case "eq_kw":
            return Eq(args[0], args[1], args[2])
        case "refl_kw":
            return Refl(args[0], args[1])
        case "eq_elim_kw":
            return EqElim(args[0], args[1], args[2], args[3], args[4], args[5])
        case _:  # pragma: no cover
            raise ParseError(f"Unknown built-in: {kind!r}")


def _elab_atom_infer(tree: Tree, scope: list[str]) -> InferTerm:  # type: ignore[type-arg]
    result = _elab_check(tree, scope)
    match result:
        case Inf(t):
            return t
        case _:
            raise ParseError(
                "Lambda in function position must be annotated: (\\x -> ...) : Type"
            )


# ---------------------------------------------------------------------------
# Variable lookup
# ---------------------------------------------------------------------------

def _elab_var(name: str, scope: list[str]) -> InferTerm:
    try:
        idx = scope.index(name)
        return Bound(idx)
    except ValueError:
        return Free(Global(name))


# ---------------------------------------------------------------------------
# Universe atoms
# ---------------------------------------------------------------------------

def _elab_universe(tree: Tree) -> InferTerm:  # type: ignore[type-arg]
    children = [c for c in tree.children if isinstance(c, Token)]
    for c in children:
        try:
            level = int(str(c))
            return Star(level)
        except ValueError:
            continue
    return Star(0)


# ---------------------------------------------------------------------------
# Built-in constructor/eliminator stubs
#
# Each built-in is treated as a curried function: elaborating the keyword
# produces an Ann(...) term whose body is a lambda wrapping the InferTerm
# constructor.  This allows  NatElim P z s n  to be parsed as four
# successive applications of the NatElim function.
# ---------------------------------------------------------------------------

def _make_succ_fn() -> InferTerm:
    return Ann(
        Lam(Inf(Succ(Inf(Bound(0))))),
        Inf(Pi(Inf(Nat()), Inf(Nat())))
    )


def _make_nat_elim_fn() -> InferTerm:
    # \P z s n -> NatElim P z s n
    return Ann(
        Lam(Lam(Lam(Lam(
            Inf(NatElim(Inf(Bound(3)), Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0))))
        )))),
        Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Nat()), Inf(Star(0))))))))))
    )


def _make_vec_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Inf(Vec(Inf(Bound(1)), Inf(Bound(0)))))),
        Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Nat()), Inf(Star(0))))))
    )


def _make_nil_fn() -> InferTerm:
    return Ann(
        Lam(Inf(Nil(Inf(Bound(0))))),
        Inf(Pi(Inf(Star(0)), Inf(Vec(Inf(Bound(0)), Inf(Zero())))))
    )


def _make_cons_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Lam(Lam(Inf(Cons(Inf(Bound(3)), Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0)))))))),
        Inf(Pi(
            Inf(Star(0)),
            Inf(Pi(
                Inf(Nat()),
                Inf(Pi(
                    Inf(Bound(1)),
                    Inf(Pi(
                        Inf(Vec(Inf(Bound(2)), Inf(Bound(1)))),
                        Inf(Vec(Inf(Bound(3)), Inf(Succ(Inf(Bound(2))))))
                    ))
                ))
            ))
        ))
    )


def _make_vec_elim_fn() -> InferTerm:
    # \A P nil cons n v -> VecElim A P nil cons n v
    return Ann(
        Lam(Lam(Lam(Lam(Lam(Lam(
            Inf(VecElim(Inf(Bound(5)), Inf(Bound(4)), Inf(Bound(3)),
                        Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0))))
        )))))),
        Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Star(0)))))))))))))),
    )


def _make_fin_fn() -> InferTerm:
    return Ann(
        Lam(Inf(Fin(Inf(Bound(0))))),
        Inf(Pi(Inf(Nat()), Inf(Star(0))))
    )


def _make_fzero_fn() -> InferTerm:
    return Ann(
        Lam(Inf(FZero(Inf(Bound(0))))),
        Inf(Pi(Inf(Nat()), Inf(Fin(Inf(Succ(Inf(Bound(0))))))))
    )


def _make_fsucc_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Inf(FSucc(Inf(Bound(1)), Inf(Bound(0)))))),
        Inf(Pi(Inf(Nat()), Inf(Pi(Inf(Fin(Inf(Bound(0)))), Inf(Fin(Inf(Succ(Inf(Bound(1))))))))))
    )


def _make_fin_elim_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Lam(Lam(Lam(
            Inf(FinElim(Inf(Bound(4)), Inf(Bound(3)), Inf(Bound(2)),
                        Inf(Bound(1)), Inf(Bound(0))))
        ))))),
        Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Star(0))))))))))))
    )


def _make_eq_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Lam(Inf(Eq(Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0))))))),
        Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Bound(0)), Inf(Pi(Inf(Bound(1)), Inf(Star(0))))))))
    )


def _make_refl_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Inf(Refl(Inf(Bound(1)), Inf(Bound(0)))))),
        Inf(Pi(
            Inf(Star(0)),
            Inf(Pi(
                Inf(Bound(0)),
                Inf(Eq(Inf(Bound(1)), Inf(Bound(0)), Inf(Bound(0))))
            ))
        ))
    )


def _make_eq_elim_fn() -> InferTerm:
    return Ann(
        Lam(Lam(Lam(Lam(Lam(Lam(
            Inf(EqElim(Inf(Bound(5)), Inf(Bound(4)), Inf(Bound(3)),
                       Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0))))
        )))))),
        Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Pi(Inf(Star(0)), Inf(Star(0)))))))))))))),
    )
