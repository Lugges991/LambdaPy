"""Quotation: convert a Value back into a CheckTerm (normal form).

This is the basis for definitional equality:
    v1 ≡ v2   iff   quote0(v1) == quote0(v2)

The integer `i` tracks the current binder depth so that HOAS closures can be
opened by applying them to a fresh  VNeutral(NFree(Quote(i)))  probe and the
resulting de Bruijn index can be reconstructed via  boundfree.
"""
from __future__ import annotations

from lambdapy.syntax import (
    # Terms
    App, Bound, CheckTerm, Cons, Eq, EqElim, Fin, FSucc, FZero, FinElim,
    Free, Inf, InferTerm, Lam, Nat, NatElim, Nil, Pi, Refl, Star, Succ,
    Vec, VecElim, Zero,
    # Values
    Value, VCons, VEq, VFin, VFSucc, VFZero, VLam, VNat, VNeutral,
    VNil, VPi, VRefl, VStar, VSucc, VVec, VZero,
    # Neutrals
    Neutral, NApp, NEqElim, NFinElim, NFree, NNatElim, NVecElim,
    # Names
    Name, Quote,
)


def quote0(v: Value) -> CheckTerm:
    """Quote a value at depth 0 (top level)."""
    return quote(0, v)


def quote(i: int, v: Value) -> CheckTerm:
    """Quote a value at binder depth i."""
    match v:
        case VNeutral(n):
            return Inf(neutral_quote(i, n))
        case VLam(fn):
            # Open the closure with a fresh probe variable Quote(i)
            probe = VNeutral(NFree(Quote(i)))
            return Lam(quote(i + 1, fn(probe)))
        case VStar(n):
            return Inf(Star(n))
        case VPi(domain, range_fn):
            probe = VNeutral(NFree(Quote(i)))
            return Inf(Pi(quote(i, domain), quote(i + 1, range_fn(probe))))
        case VNat():
            return Inf(Nat())
        case VZero():
            return Inf(Zero())
        case VSucc(pred):
            return Inf(Succ(quote(i, pred)))
        case VNil(a):
            return Inf(Nil(quote(i, a)))
        case VCons(a, n, h, t):
            return Inf(Cons(quote(i, a), quote(i, n), quote(i, h), quote(i, t)))
        case VVec(a, n):
            return Inf(Vec(quote(i, a), quote(i, n)))
        case VFin(n):
            return Inf(Fin(quote(i, n)))
        case VFZero(n):
            return Inf(FZero(quote(i, n)))
        case VFSucc(n, x):
            return Inf(FSucc(quote(i, n), quote(i, x)))
        case VEq(a, x, y):
            return Inf(Eq(quote(i, a), quote(i, x), quote(i, y)))
        case VRefl(a, x):
            return Inf(Refl(quote(i, a), quote(i, x)))
        case _:  # pragma: no cover
            raise ValueError(f"quote: unhandled value {v!r}")


def neutral_quote(i: int, n: Neutral) -> InferTerm:
    """Quote a neutral term at binder depth i."""
    match n:
        case NFree(name):
            return boundfree(i, name)
        case NApp(func, arg):
            return App(neutral_quote(i, func), quote(i, arg))
        case NNatElim(motive, base, step, k):
            return NatElim(
                quote(i, motive),
                quote(i, base),
                quote(i, step),
                Inf(neutral_quote(i, k)),
            )
        case NVecElim(elem_type, motive, nil_case, cons_case, length, vec):
            return VecElim(
                quote(i, elem_type),
                quote(i, motive),
                quote(i, nil_case),
                quote(i, cons_case),
                quote(i, length),
                Inf(neutral_quote(i, vec)),
            )
        case NFinElim(motive, fzero_case, fsucc_case, n_val, f):
            return FinElim(
                quote(i, motive),
                quote(i, fzero_case),
                quote(i, fsucc_case),
                quote(i, n_val),
                Inf(neutral_quote(i, f)),
            )
        case NEqElim(type_, motive, refl_case, left, right, proof):
            return EqElim(
                quote(i, type_),
                quote(i, motive),
                quote(i, refl_case),
                quote(i, left),
                quote(i, right),
                Inf(neutral_quote(i, proof)),
            )
        case _:  # pragma: no cover
            raise ValueError(f"neutral_quote: unhandled neutral {n!r}")


def boundfree(i: int, name: Name) -> InferTerm:
    """Convert a Name back to an InferTerm.

    Quote(k) was introduced at depth k, so at the current depth i its
    de Bruijn index is  i - k - 1  (innermost = 0).
    """
    match name:
        case Quote(k):
            return Bound(max(0, i - k - 1))
        case _:
            return Free(name)
