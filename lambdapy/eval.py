"""Normalisation-by-evaluation (NbE) for LambdaPi.

The environment is a list[Value] where index 0 is the innermost binder
(de Bruijn convention).  Values use HOAS: lambda bodies are Python callables,
so beta-reduction is just Python function application.
"""
from __future__ import annotations

from lambdapy.errors import EvalError
from lambdapy.nameenv import get_name_env
from lambdapy.syntax import (
    # Terms
    Ann, App, Bound, CheckTerm, Cons, EqElim, Fin, FSucc, FZero,
    FinElim, Free, Global, InferTerm, Lam, Inf, Nat, NatElim, Nil, Pi,
    Refl, Star, Succ, Eq, Vec, VecElim, Zero,
    # Values
    Value, VCons, VEq, VFin, VFSucc, VFZero, VLam, VNat, VNeutral,
    VNil, VPi, VRefl, VStar, VSucc, VVec, VZero,
    # Neutrals
    Neutral, NApp, NEqElim, NFinElim, NFree, NNatElim, NVecElim,
)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_inf(term: InferTerm, env: list[Value]) -> Value:
    """Evaluate an inferable term to a value."""
    match term:
        case Ann(expr, _):
            return eval_chk(expr, env)
        case Star(n):
            return VStar(n)
        case Pi(domain, range_):
            d = eval_chk(domain, env)
            # Capture env by value in the closure to avoid aliasing bugs
            e = env
            return VPi(d, lambda v, r=range_, ev=e: eval_chk(r, [v] + ev))
        case Free(Global(name)):
            v = get_name_env().get(name)
            return v if v is not None else VNeutral(NFree(Global(name)))
        case Free(name):
            return VNeutral(NFree(name))
        case Bound(i):
            return env[i]
        case App(func, arg):
            return vapp(eval_inf(func, env), eval_chk(arg, env))
        # Nat
        case Nat():
            return VNat()
        case Zero():
            return VZero()
        case Succ(n):
            return VSucc(eval_chk(n, env))
        case NatElim(motive, base, step, k):
            return nat_elim(
                eval_chk(motive, env),
                eval_chk(base, env),
                eval_chk(step, env),
                eval_chk(k, env),
            )
        # Vec
        case Vec(elem_type, length):
            return VVec(eval_chk(elem_type, env), eval_chk(length, env))
        case Nil(elem_type):
            return VNil(eval_chk(elem_type, env))
        case Cons(elem_type, length, head, tail):
            return VCons(
                eval_chk(elem_type, env),
                eval_chk(length, env),
                eval_chk(head, env),
                eval_chk(tail, env),
            )
        case VecElim(elem_type, motive, nil_case, cons_case, length, vec):
            return vec_elim(
                eval_chk(elem_type, env),
                eval_chk(motive, env),
                eval_chk(nil_case, env),
                eval_chk(cons_case, env),
                eval_chk(length, env),
                eval_chk(vec, env),
            )
        # Fin
        case Fin(n):
            return VFin(eval_chk(n, env))
        case FZero(n):
            return VFZero(eval_chk(n, env))
        case FSucc(n, x):
            return VFSucc(eval_chk(n, env), eval_chk(x, env))
        case FinElim(motive, fzero_case, fsucc_case, n, f):
            return fin_elim(
                eval_chk(motive, env),
                eval_chk(fzero_case, env),
                eval_chk(fsucc_case, env),
                eval_chk(n, env),
                eval_chk(f, env),
            )
        # Eq
        case Eq(type_, left, right):
            return VEq(
                eval_chk(type_, env),
                eval_chk(left, env),
                eval_chk(right, env),
            )
        case Refl(type_, val):
            return VRefl(eval_chk(type_, env), eval_chk(val, env))
        case EqElim(type_, motive, refl_case, left, right, proof):
            return eq_elim(
                eval_chk(type_, env),
                eval_chk(motive, env),
                eval_chk(refl_case, env),
                eval_chk(left, env),
                eval_chk(right, env),
                eval_chk(proof, env),
            )
        case _:  # pragma: no cover
            raise EvalError(f"eval_inf: unhandled term {term!r}")


def eval_chk(term: CheckTerm, env: list[Value]) -> Value:
    """Evaluate a checkable term to a value."""
    match term:
        case Inf(t):
            return eval_inf(t, env)
        case Lam(body):
            e = env
            return VLam(lambda v, b=body, ev=e: eval_chk(b, [v] + ev))
        case _:  # pragma: no cover
            raise EvalError(f"eval_chk: unhandled term {term!r}")


def vapp(fn: Value, arg: Value) -> Value:
    """Apply a value to an argument value."""
    match fn:
        case VLam(f):
            return f(arg)
        case VNeutral(n):
            return VNeutral(NApp(n, arg))
        case _:  # pragma: no cover
            raise EvalError(f"vapp: not a function: {fn!r}")


# ---------------------------------------------------------------------------
# Eliminator reduction (iota rules)
# ---------------------------------------------------------------------------

def nat_elim(motive: Value, base: Value, step: Value, k: Value) -> Value:
    """Reduce natElim when the scrutinee is known."""
    match k:
        case VZero():
            return base
        case VSucc(pred):
            rec = nat_elim(motive, base, step, pred)
            # step : (k : Nat) -> P k -> P (Succ k)
            return vapp(vapp(step, pred), rec)
        case VNeutral(n):
            return VNeutral(NNatElim(motive, base, step, n))
        case _:  # pragma: no cover
            raise EvalError(f"nat_elim: not a Nat value: {k!r}")


def vec_elim(
    elem_type: Value,
    motive: Value,
    nil_case: Value,
    cons_case: Value,
    length: Value,
    vec: Value,
) -> Value:
    """Reduce vecElim when the vector scrutinee is known."""
    match vec:
        case VNil(_):
            return nil_case
        case VCons(_, n, h, t):
            rec = vec_elim(elem_type, motive, nil_case, cons_case, n, t)
            # cons_case n h t rec
            return vapp(vapp(vapp(vapp(cons_case, n), h), t), rec)
        case VNeutral(n):
            return VNeutral(NVecElim(elem_type, motive, nil_case, cons_case, length, n))
        case _:  # pragma: no cover
            raise EvalError(f"vec_elim: not a Vec value: {vec!r}")


def fin_elim(
    motive: Value,
    fzero_case: Value,
    fsucc_case: Value,
    n: Value,
    f: Value,
) -> Value:
    """Reduce finElim when the finite ordinal scrutinee is known."""
    match f:
        case VFZero(n_val):
            # fzero_case : (n : Nat) -> P (Succ n) (FZero n)
            return vapp(fzero_case, n_val)
        case VFSucc(n_val, x_val):
            rec = fin_elim(motive, fzero_case, fsucc_case, n_val, x_val)
            # fsucc_case : (n : Nat) -> (f : Fin n) -> P n f -> P (Succ n) (FSucc n f)
            return vapp(vapp(vapp(fsucc_case, n_val), x_val), rec)
        case VNeutral(neut):
            return VNeutral(NFinElim(motive, fzero_case, fsucc_case, n, neut))
        case _:  # pragma: no cover
            raise EvalError(f"fin_elim: not a Fin value: {f!r}")


def eq_elim(
    type_: Value,
    motive: Value,
    refl_case: Value,
    left: Value,
    right: Value,
    proof: Value,
) -> Value:
    """Reduce eqElim (J rule) when the proof is Refl."""
    match proof:
        case VRefl(_, x):
            # Only the refl case applies; J reduces to refl_case x
            return vapp(refl_case, x)
        case VNeutral(n):
            return VNeutral(NEqElim(type_, motive, refl_case, left, right, n))
        case _:  # pragma: no cover
            raise EvalError(f"eq_elim: not an Eq proof value: {proof!r}")
