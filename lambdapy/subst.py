"""Substitution for LambdaPi terms.

Used by the elaborator (parser) to convert named binders to de Bruijn indices.
NOT used by the NbE evaluator at runtime.

subst_inf(i, r, t): replace Bound(i) with r in inferable term t.
subst_chk(i, r, t): replace Bound(i) with r in checkable term t.

When crossing a binder (Lam or Pi range), the index i is incremented because
a new variable has been pushed onto the de Bruijn stack.
"""
from __future__ import annotations

from lambdapy.syntax import (
    Ann, App, Bound, CheckTerm, Cons, EqElim, Eq, Fin, FSucc, FZero,
    FinElim, Free, Inf, InferTerm, Lam, Nat, NatElim, Nil, Pi, Refl,
    Star, Succ, Vec, VecElim, Zero,
)


def subst_inf(i: int, r: InferTerm, term: InferTerm) -> InferTerm:
    """Substitute r for Bound(i) in an inferable term."""
    match term:
        case Ann(expr, type_):
            return Ann(subst_chk(i, r, expr), subst_chk(i, r, type_))
        case Star(_):
            return term
        case Pi(domain, range_):
            return Pi(
                subst_chk(i, r, domain),
                subst_chk(i + 1, r, range_),  # range is under one binder
            )
        case Bound(j):
            return r if i == j else term
        case Free(_):
            return term
        case App(func, arg):
            return App(subst_inf(i, r, func), subst_chk(i, r, arg))
        # Nat
        case Nat() | Zero():
            return term
        case Succ(n):
            return Succ(subst_chk(i, r, n))
        case NatElim(motive, base, step, k):
            return NatElim(
                subst_chk(i, r, motive),
                subst_chk(i, r, base),
                subst_chk(i, r, step),
                subst_chk(i, r, k),
            )
        # Vec
        case Vec(elem_type, length):
            return Vec(subst_chk(i, r, elem_type), subst_chk(i, r, length))
        case Nil(elem_type):
            return Nil(subst_chk(i, r, elem_type))
        case Cons(elem_type, length, head, tail):
            return Cons(
                subst_chk(i, r, elem_type),
                subst_chk(i, r, length),
                subst_chk(i, r, head),
                subst_chk(i, r, tail),
            )
        case VecElim(elem_type, motive, nil_case, cons_case, length, vec):
            return VecElim(
                subst_chk(i, r, elem_type),
                subst_chk(i, r, motive),
                subst_chk(i, r, nil_case),
                subst_chk(i, r, cons_case),
                subst_chk(i, r, length),
                subst_chk(i, r, vec),
            )
        # Fin
        case Fin(n):
            return Fin(subst_chk(i, r, n))
        case FZero(n):
            return FZero(subst_chk(i, r, n))
        case FSucc(n, x):
            return FSucc(subst_chk(i, r, n), subst_chk(i, r, x))
        case FinElim(motive, fzero_case, fsucc_case, n, f):
            return FinElim(
                subst_chk(i, r, motive),
                subst_chk(i, r, fzero_case),
                subst_chk(i, r, fsucc_case),
                subst_chk(i, r, n),
                subst_chk(i, r, f),
            )
        # Eq
        case Eq(type_, left, right):
            return Eq(
                subst_chk(i, r, type_),
                subst_chk(i, r, left),
                subst_chk(i, r, right),
            )
        case Refl(type_, val):
            return Refl(subst_chk(i, r, type_), subst_chk(i, r, val))
        case EqElim(type_, motive, refl_case, left, right, proof):
            return EqElim(
                subst_chk(i, r, type_),
                subst_chk(i, r, motive),
                subst_chk(i, r, refl_case),
                subst_chk(i, r, left),
                subst_chk(i, r, right),
                subst_chk(i, r, proof),
            )
        case _:  # pragma: no cover
            raise ValueError(f"subst_inf: unhandled term {term!r}")


def subst_chk(i: int, r: InferTerm, term: CheckTerm) -> CheckTerm:
    """Substitute r for Bound(i) in a checkable term."""
    match term:
        case Inf(t):
            return Inf(subst_inf(i, r, t))
        case Lam(body):
            # Lambda introduces a new binder: Bound(0) is the lambda variable,
            # so the index we are substituting for becomes i+1 inside the body.
            return Lam(subst_chk(i + 1, r, body))
        case _:  # pragma: no cover
            raise ValueError(f"subst_chk: unhandled term {term!r}")
