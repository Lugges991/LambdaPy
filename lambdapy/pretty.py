"""Pretty printer for LambdaPi values and terms.

pretty_value(v)       -> str   (via quote0)
pretty_check(t, names) -> str
pretty_infer(t, names) -> str

A `names` list tracks the names used by each de Bruijn binder (innermost
first).  When entering a new binder a fresh name is picked that doesn't clash
with any currently-in-scope name.
"""
from __future__ import annotations

from lambdapy.quote import quote0
from lambdapy.syntax import (
    Ann, App, Bound, CheckTerm, Cons, Eq, EqElim, Fin, FSucc, FZero,
    FinElim, Free, Global, Inf, InferTerm, Lam, Local, Nat, NatElim,
    Nil, Pi, Quote, Refl, Star, Succ, Vec, VecElim, Zero,
    Value,
)

# Name candidates used when freshening binder names
_CANDIDATES = list("xyzwuvabcdefghijklmnopqrst") + [f"x{i}" for i in range(100)]


def pretty_value(v: Value) -> str:
    """Convert a value to a human-readable string (via normal form)."""
    return pretty_check(quote0(v), [])


def pretty_check(term: CheckTerm, names: list[str]) -> str:
    """Pretty-print a CheckTerm given the current name stack (innermost first)."""
    match term:
        case Lam(body):
            fresh = _fresh(names)
            body_str = pretty_check(body, [fresh] + names)
            return f"\\{fresh} -> {body_str}"
        case Inf(t):
            return pretty_infer(t, names)
        case _:  # pragma: no cover
            return repr(term)


def pretty_infer(term: InferTerm, names: list[str]) -> str:
    """Pretty-print an InferTerm given the current name stack."""
    match term:
        case Bound(i):
            if i < len(names):
                return names[i]
            return f"?{i}"  # out-of-scope index (shouldn't happen in well-typed terms)

        case Free(Global(n)):
            return n

        case Free(Local(i)):
            return f"_local{i}"

        case Free(Quote(i)):
            return f"_quote{i}"

        case Star(0):
            return "Type"

        case Star(n):
            return f"Type {n}"

        case Ann(expr, type_):
            return f"({pretty_check(expr, names)} : {pretty_check(type_, names)})"

        case Pi(domain, range_):
            # Check whether the range actually uses Bound(0) — if not, print as ->
            if not _has_bound_zero(range_):
                domain_str = _paren_if_needed(pretty_check(domain, names), domain)
                range_str = pretty_check(range_, ["_"] + names)
                return f"{domain_str} -> {range_str}"
            else:
                fresh = _fresh(names)
                domain_str = pretty_check(domain, names)
                range_str = pretty_check(range_, [fresh] + names)
                return f"forall ({fresh} : {domain_str}). {range_str}"

        case App(func, arg):
            func_str = pretty_infer(func, names)
            arg_str = _paren_check(pretty_check(arg, names), arg)
            return f"{func_str} {arg_str}"

        case Nat():
            return "Nat"

        case Zero():
            return "Zero"

        case Succ(n):
            # Pretty-print decimal for numeric literals
            count = _count_succ(term)
            if count is not None:
                return str(count)
            return f"Succ {_paren_check(pretty_check(n, names), n)}"

        case NatElim(motive, base, step, k):
            return (f"NatElim {_pa(motive, names)} {_pa(base, names)}"
                    f" {_pa(step, names)} {_pa(k, names)}")

        case Vec(a, n):
            return f"Vec {_pa(a, names)} {_pa(n, names)}"

        case Nil(a):
            return f"Nil {_pa(a, names)}"

        case Cons(a, n, h, t):
            return (f"Cons {_pa(a, names)} {_pa(n, names)}"
                    f" {_pa(h, names)} {_pa(t, names)}")

        case VecElim(a, motive, nil_c, cons_c, n, v):
            return (f"VecElim {_pa(a, names)} {_pa(motive, names)}"
                    f" {_pa(nil_c, names)} {_pa(cons_c, names)}"
                    f" {_pa(n, names)} {_pa(v, names)}")

        case Fin(n):
            return f"Fin {_pa(n, names)}"

        case FZero(n):
            return f"FZero {_pa(n, names)}"

        case FSucc(n, x):
            return f"FSucc {_pa(n, names)} {_pa(x, names)}"

        case FinElim(motive, fz, fs, n, f):
            return (f"FinElim {_pa(motive, names)} {_pa(fz, names)}"
                    f" {_pa(fs, names)} {_pa(n, names)} {_pa(f, names)}")

        case Eq(a, x, y):
            return f"Eq {_pa(a, names)} {_pa(x, names)} {_pa(y, names)}"

        case Refl(a, x):
            return f"Refl {_pa(a, names)} {_pa(x, names)}"

        case EqElim(a, motive, rc, l, r, p):
            return (f"EqElim {_pa(a, names)} {_pa(motive, names)}"
                    f" {_pa(rc, names)} {_pa(l, names)} {_pa(r, names)} {_pa(p, names)}")

        case _:  # pragma: no cover
            return repr(term)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh(names: list[str]) -> str:
    """Pick a fresh name not currently in scope."""
    used = set(names)
    for c in _CANDIDATES:
        if c not in used:
            return c
    return f"v{len(names)}"


def _has_bound(term: CheckTerm, target: int) -> bool:
    """Return True if Bound(target) appears free in term."""
    match term:
        case Inf(t):
            return _has_bound_inf(t, target)
        case Lam(body):
            return _has_bound(body, target + 1)
        case _:
            return False


def _has_bound_inf(term: InferTerm, target: int) -> bool:
    match term:
        case Bound(n):
            return n == target
        case Free(_) | Star(_) | Nat() | Zero():
            return False
        case Ann(e, t):
            return _has_bound(e, target) or _has_bound(t, target)
        case Pi(d, r):
            return _has_bound(d, target) or _has_bound(r, target + 1)
        case App(f, a):
            return _has_bound_inf(f, target) or _has_bound(a, target)
        case Succ(n):
            return _has_bound(n, target)
        case NatElim(m, b, s, k):
            return any(_has_bound(x, target) for x in (m, b, s, k))
        case Vec(a, n):
            return _has_bound(a, target) or _has_bound(n, target)
        case Nil(a):
            return _has_bound(a, target)
        case Cons(a, n, h, t):
            return any(_has_bound(x, target) for x in (a, n, h, t))
        case VecElim(a, m, nc, cc, n, v):
            return any(_has_bound(x, target) for x in (a, m, nc, cc, n, v))
        case Fin(n):
            return _has_bound(n, target)
        case FZero(n):
            return _has_bound(n, target)
        case FSucc(n, f):
            return _has_bound(n, target) or _has_bound(f, target)
        case FinElim(m, fz, fs, n, f):
            return any(_has_bound(x, target) for x in (m, fz, fs, n, f))
        case Eq(a, x, y):
            return any(_has_bound(t, target) for t in (a, x, y))
        case Refl(a, x):
            return _has_bound(a, target) or _has_bound(x, target)
        case EqElim(a, m, rc, l, r, p):
            return any(_has_bound(t, target) for t in (a, m, rc, l, r, p))
        case _:
            return False


def _has_bound_zero(term: CheckTerm) -> bool:
    """Return True if Bound(0) appears free in term."""
    return _has_bound(term, 0)


def _has_bound_zero_inf(term: InferTerm) -> bool:
    return _has_bound_inf(term, 0)


def _has_bound_zero_shifted(term: CheckTerm, target: int) -> bool:
    """Return True if Bound(target) appears free in term."""
    return _has_bound(term, target)


def _paren_if_needed(s: str, term: CheckTerm) -> str:
    """Parenthesise a domain string if it contains '->' (to avoid ambiguity)."""
    if "->" in s or "forall" in s:
        return f"({s})"
    return s


def _paren_check(s: str, term: CheckTerm) -> str:
    """Parenthesise a CheckTerm argument if it's compound."""
    match term:
        case Inf(Bound(_)) | Inf(Free(_)) | Inf(Star(_)) | Inf(Nat()) | Inf(Zero()):
            return s
        case _:
            # Atoms don't need parens; compound terms do
            needs_parens = " " in s and not s.startswith("(")
            return f"({s})" if needs_parens else s


def _pa(term: CheckTerm, names: list[str]) -> str:
    """Pretty-print a CheckTerm argument (parenthesised if compound)."""
    s = pretty_check(term, names)
    return _paren_check(s, term)


def _count_succ(term: InferTerm) -> int | None:
    """If term is a chain of Succ applied to Zero, return the count; else None."""
    count = 0
    t: InferTerm = term
    while True:
        match t:
            case Zero():
                return count
            case Succ(Inf(inner)):
                count += 1
                t = inner
            case _:
                return None
