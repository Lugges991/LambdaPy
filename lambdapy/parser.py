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
    VPi, VStar, VNat, VVec, VNil, VZero, VCons, VSucc, VFin, VFSucc, VFZero,
    VRefl, VEq,
)
from lambdapy.eval import vapp


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


# Constructor for natural numbers
# \m mz ms n -> NatElim m mz ms n
# natElim type / LambdaPi-Paper:
#     (m : Nat -> *) -> m Zero -> ((k : Nat) -> m k -> m (Succ k)) -> (n : Nat) -> m n
# InferTerm:
#     forall (m : Nat -> Type). m Zero -> (forall (l : Nat). m l -> m (Succ l)) -> forall (k : Nat). m k
# Haskell code: https://github.com/ilya-klyuchnikov/lambdapi/blob/79ddf21581e03ea34a94cc00ffd5c8684d845ed9/src/LambdaPi/Main.hs#L17
def make_nat_elim_type():
    """Build the correct type of natElim as a Value."""
    return (  # nopep8
        # ∀m :: Nat -> ∗.
        VPi(VPi(VNat(), lambda _: VStar(0)), lambda m:
            #             m Zero ->
            VPi(vapp(m, VZero()), lambda _:
            #                       ( ∀l :: Nat.m l -> m (Succ l)) ->
            VPi(VPi(VNat(), lambda l: VPi(vapp(m, l), lambda _: vapp(m, VSucc(l)))), lambda _:
            #                                                          ∀k :: Nat.m k
            VPi(VNat(), lambda k: vapp(m, k)))))
    )


def _make_nat_elim_fn() -> InferTerm:
    """Build the vector folding function.

    Correct indices were derived by
        from lambdapy.quote import quote0
        quote0(make_nat_elim_type())
    """
    return Ann(
        Lam(Lam(Lam(Lam(
            Inf(NatElim(Inf(Bound(3)), Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0))))
        )))),
        Inf(Pi(Inf(Pi(Inf(Nat()), Inf(Star(0)))),  # m : Nat -> *
               Inf(Pi(Inf(App(Bound(0), Inf(Zero()))),  # mz : m Zero
               Inf(Pi(Inf(Pi(Inf(Nat()),   # ms : (l : Nat) ->
                      Inf(Pi(Inf(App(Bound(2), Inf(Bound(0)))),  # m l ->
                             Inf(App(Bound(3), Inf(Succ(Inf(Bound(1))))))))  # m (Succ l)  # nopep8
                    )),
                    Inf(Pi(Inf(Nat()),   # k : Nat
                           Inf(App(Bound(3), Inf(Bound(0))))))  # m k
               ))))
        ))
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


# Constructor for vectors
# \a m mn mc l xs -> VecElim a m mn mc l xs
# vecElim type / LambdaPi-Paper:
#    (a : *) -> (m : (k : Nat) -> Vec a k -> *) ->
#               m Zero (Nil a)
#            -> ((l : Nat) -> (x : a) -> (xs : Vec a l) ->
#               m l xs -> m (Succ l) (Cons a l x xs))
#            -> (k : Nat) -> (xs : Vec a k) -> m k xs
# InferTerm:
#     forall (x : Type). forall (y : forall (y : Nat). Vec x y -> Type). y Zero (Nil x) -> (forall (z : Nat). forall (w : x). forall (u : Vec x z). y z u -> y (Succ z) (Cons x z w u)) -> forall (z : Nat). forall (w : Vec x z). y z w
# Haskell code: https://github.com/ilya-klyuchnikov/lambdapi/blob/79ddf21581e03ea34a94cc00ffd5c8684d845ed9/src/LambdaPi/Main.hs#L26
def make_vec_elim_type():
    """Build the correct type of vecElim as a Value."""
    return (  # nopep8
        # ∀a :: *.
        VPi(VStar(0), lambda a:
            #      ∀m :: (∀k :: Nat. Vec a k -> *).
            VPi(VPi(VNat(), lambda k: VPi(VVec(a, k), lambda _: VStar(0))), lambda m:
            #         m Zero (Nil a)
            #     ->
            VPi(vapp(vapp(m, VZero()), VNil(a)), lambda _:
            #         ( ∀l :: Nat.
            VPi(VPi(VNat(), lambda l:
                #                  ∀x :: a.
                VPi(a, lambda x:
                #                           ∀xs :: Vec a l.
                VPi(VVec(a, l), lambda xs:
                #     m l xs ->
                VPi(vapp(vapp(m, l), xs), lambda _:
                #               m (Succ l) (Cons a l x xs))
                # ->
                vapp(vapp(m, VSucc(l)), VCons(a, l, x, xs)))))), lambda _:
            #        ∀k :: Nat.
            VPi(VNat(), lambda k:
            #                   ∀xs :: Vec α k.m k xs
            VPi(VVec(a, k), lambda xs: vapp(vapp(m, k), xs)))))))
    )


def _make_vec_elim_fn() -> InferTerm:  # noqa
    """Build the vector folding function.

    Correct indices were derived by
        from lambdapy.quote import quote0
        quote0(make_vec_elim_type())
    """
    return Ann(  # nopep8
        Lam(Lam(Lam(Lam(Lam(Lam(
            Inf(VecElim(Inf(Bound(5)), Inf(Bound(4)), Inf(Bound(3)),
                        Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0)))))
        ))))),
        # (a : *) ->
        Inf(Pi(Inf(Star(0)),
            # m : (n : Nat) -> Vec a n -> *
            Inf(Pi(Inf(Pi(Inf(Nat()), Inf(Pi(Inf(Vec(Inf(Bound(1)), Inf(Bound(0)))), Inf(Star(0)))))),
                # mn : m Zero (Nil a)
                Inf(Pi(Inf(App(App(Bound(0), Inf(Zero())), Inf(Nil(Inf(Bound(1)))))),
                    # mc : (n : Nat) ->
                    Inf(Pi(
                        Inf(Pi(Inf(Nat()),
                            # (x : a) ->
                            Inf(Pi(Inf(Bound(3)),
                                # (xs : Vec a n) ->
                                Inf(Pi(Inf(Vec(Inf(Bound(4)), Inf(Bound(1)))),
                                    # m n xs ->
                                    Inf(Pi(Inf(App(App(Bound(4), Inf(Bound(2))), Inf(Bound(0)))),
                                        # m (Succ n)
                                        Inf(App(App(Bound(5), Inf(Succ(Inf(Bound(3))))),
                                            # (Cons a n x xs)
                                            Inf(Cons(Inf(Bound(6)), Inf(Bound(3)), Inf(Bound(2)), Inf(Bound(1))))))
                                    ))
                                ))
                        )))),
                        # n : Nat
                        Inf(Pi(Inf(Nat()),
                            # (xs : Vec a n)
                            Inf(Pi(Inf(Vec(Inf(Bound(4)), Inf(Bound(0)))),
                                # m n xs
                                Inf(App(App(Bound(4), Inf(Bound(1))), Inf(Bound(0))))
                            ))
                        ))
                    ))
        ))))))
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


# Constructor for finites
# \m mz ms n f -> FinElim m mz ms n f
# finElim type:
#    (m : (n : Nat) -> Fin n -> *) -> ((n : Nat) -> m (Succ n) (FZero n)) -> ((n : Nat) -> (f : Fin n) -> m n f -> m (Succ n) (FSucc n f)) -> (n : Nat) -> (f : Fin n) -> m n f
# InferTerm:
#    forall (x : forall (x : Nat). Fin x -> Type). (forall (y : Nat). x (Succ y) (FZero y)) -> (forall (y : Nat). forall (z : Fin y). x y z -> x (Succ y) (FSucc y z)) -> forall (y : Nat). forall (z : Fin y). x y z
# Haskell code: https://github.com/ilya-klyuchnikov/lambdapi/blob/79ddf21581e03ea34a94cc00ffd5c8684d845ed9/src/LambdaPi/Main.hs#L49
def make_fin_elim_type():
    """Build the correct type of finElim as a Value."""
    return (  # nopep8
        # ∀m :: (∀n :: Nat. Fin n -> *).
        VPi(VPi(VNat(), lambda n: VPi(VFin(n), lambda _: VStar(0))), lambda m:
            #                            (∀n :: Nat. m (Succ n) (FZero n))
            #                           ->
            VPi(VPi(VNat(), lambda n: vapp(vapp(m, VSucc(n)), VFZero(n))), lambda _:
            #                              (∀n :: Nat.
            VPi(VPi(VNat(), lambda n:
                #                                      ∀f :: Fin n.
                VPi(VFin(n), lambda f:
                #                                                   m n f ->
                VPi(vapp(vapp(m, n), f), lambda _:
                #                                                            m (Succ n) (FSucc n f))
                #                        ->
                vapp(vapp(m, VSucc(n)), VFSucc(n, f))))), lambda _:
            #                               ∀n :: Nat. ∀f :: Fin n.
            VPi(VNat(), lambda n: VPi(VFin(n), lambda f:
            #                                                       m n f
            vapp(vapp(m, n), f))))))
    )


def _make_fin_elim_fn() -> InferTerm:
    return Ann(  # nopep8
        Lam(Lam(Lam(Lam(Lam(
            Inf(FinElim(Inf(Bound(4)), Inf(Bound(3)), Inf(Bound(2)),
                        Inf(Bound(1)), Inf(Bound(0))))
        ))))),
        # (m : (n : Nat) ->
        Inf(Pi(Inf(Pi(Inf(Nat()),
        # Fin n -> *) ->
        Inf(Pi(Inf(Fin(Inf(Bound(0)))), Inf(Star(0)))))),
            # ((n : Nat) ->
            Inf(Pi(Inf(Pi(Inf(Nat()),
            # m (Succ n) (FZero n)) ->
            Inf(App(App(Bound(1), Inf(Succ(Inf(Bound(0))))), Inf(FZero(Inf(Bound(0)))))))),
            # ((n : Nat) -> (f : Fin n) ->
            Inf(Pi(Inf(Pi(Inf(Nat()), Inf(Pi(Inf(Fin(Inf(Bound(0)))),
                # m n f -> m (Succ n) (FSucc n f)) ->
                Inf(Pi(Inf(App(App(Bound(3), Inf(Bound(1))), Inf(Bound(0)))),
                # m (Succ n) (FSucc n f)) ->
                Inf(App(App(Bound(4), Inf(Succ(Inf(Bound(2))))), Inf(FSucc(Inf(Bound(2)), Inf(Bound(1)))))))))))),
            # (n : Nat) -> (f : Fin n) -> m n f
            Inf(Pi(Inf(Nat()), Inf(Pi(Inf(Fin(Inf(Bound(0)))),
            # m n f
            Inf(App(App(Bound(4), Inf(Bound(1))), Inf(Bound(0))))))))))))
        ))
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


# Constructor for equality
# \a m mr x y eq -> EqElim a m mr x y eq
# eqElim type / LambdaPi-Paper:
#     ∀(α :: ∗).∀(m :: ∀(x :: α).∀(y :: α).Eq α x y -> ∗).
#                    (∀(z :: α).m z z (Refl α z))
#               -> ∀(x :: α).∀(y :: α).∀(p :: Eq α x y).m x y p
# InferTerm:
#     forall
# Haskell code: https://github.com/ilya-klyuchnikov/lambdapi/blob/79ddf21581e03ea34a94cc00ffd5c8684d845ed9/src/LambdaPi/Main.hs#L39
def make_eq_elim_type():
    """Build the correct type of eqElim as a Value."""
    return (  # nopep8
        # ∀(α :: ∗).
        VPi(VStar(0), lambda a:
            # ∀(m :: ∀(x :: α).∀(y :: α).Eq α x y -> ∗).
            VPi(VPi(a, lambda x: VPi(a, lambda y: VPi(VEq(a, x, y), lambda _: VStar(0)))), lambda m:
            # (∀(z :: α).m z z (Refl α z)) ->
            VPi(VPi(a, lambda z: vapp(vapp(vapp(m, z), z), VRefl(a, z))), lambda _:
            # ∀(x :: α).∀(y :: α).
            VPi(a, lambda x: VPi(a, lambda y:
            # ∀(p :: Eq α x y).
            VPi(VEq(a, x, y), lambda p:
            # m x y p
            vapp(vapp(vapp(m, x), y), p)))))))
    )


def _make_eq_elim_fn() -> InferTerm:
    """Build the equality folding function.

    Correct indices were derived by
        from lambdapy.quote import quote0
        quote0(make_eq_elim_type())
    """
    return Ann(
        Lam(Lam(Lam(Lam(Lam(Lam(
            Inf(EqElim(Inf(Bound(5)), Inf(Bound(4)), Inf(Bound(3)),
                       Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0))))
        )))))),
        Inf(Pi(Inf(Star(0)),
            Inf(Pi(Inf(Pi(Inf(Bound(0)), Inf(Pi(Inf(Bound(1)), Inf(Pi(Inf(Eq(Inf(Bound(2)), Inf(Bound(1)), Inf(Bound(0)))), Inf(Star(0)))))))),
                Inf(Pi(Inf(Pi(Inf(Bound(1)), Inf(App(App(App(Bound(1), Inf(Bound(0))), Inf(Bound(0))), Inf(Refl(Inf(Bound(2)), Inf(Bound(0)))))))),
                    Inf(Pi(Inf(Bound(2)), Inf(Pi(Inf(Bound(3)),
                        Inf(Pi(Inf(Eq(Inf(Bound(4)), Inf(Bound(1)), Inf(Bound(0)))),
                            Inf(App(App(App(Bound(4), Inf(Bound(2))), Inf(Bound(1))), Inf(Bound(0))))))))))))))
               ))
    )
