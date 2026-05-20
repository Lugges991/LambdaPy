"""Bidirectional type checker for LambdaPi.

type_inf(i, ctx, term)  — synthesise the type of an InferTerm.
type_chk(i, ctx, term, ty) — check a CheckTerm against a given type Value.

The integer `i` counts how many Pi binders have been opened, used to create
fresh Local(i) names when checking under a binder.

Definitional equality is decided by quoting both sides to normal forms and
comparing with structural (==) equality on frozen dataclasses.
"""
from __future__ import annotations

from lambdapy.context import (
    Context, ContextEntry, HasKind, HasType, context_lookup,
)
from lambdapy.errors import TypeCheckError
from lambdapy.eval import (
    eval_chk, eval_inf, vapp,
    nat_elim, vec_elim, fin_elim, eq_elim,
)
from lambdapy.quote import quote0
from lambdapy.subst import subst_chk
from lambdapy.syntax import (
    Ann, App, Bound, CheckTerm, Cons, EqElim, Eq, Fin, FSucc, FZero,
    FinElim, Free, Inf, InferTerm, Lam, Local, Nat, NatElim, Nil, Pi,
    Refl, Star, Succ, Vec, VecElim, Zero,
    Value, VCons, VEq, VFin, VFSucc, VFZero, VLam, VNat, VNeutral,
    VNil, VPi, VRefl, VStar, VSucc, VVec, VZero,
    NFree,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def type_inf0(ctx: Context, term: InferTerm) -> Value:
    """Type-infer at top level (depth 0)."""
    return type_inf(0, ctx, term)


def type_inf(i: int, ctx: Context, term: InferTerm) -> Value:
    """Synthesise the type of an inferable term at binder depth i."""
    match term:

        # --- Annotation ---
        case Ann(expr, type_annot):
            # Evaluate the annotation and verify it is a valid type.
            ty_val = eval_chk(type_annot, [])
            _check_is_type(i, ctx, type_annot)
            type_chk(i, ctx, expr, ty_val)
            return ty_val

        # --- Universes ---
        case Star(n):
            return VStar(n + 1)  # Star(n) : Star(n+1)

        # --- Dependent product ---
        case Pi(domain, range_):
            j = _check_is_type(i, ctx, domain)
            local = Local(i)
            d_val = eval_chk(domain, [])
            new_ctx: Context = [(local, HasType(d_val))] + ctx
            # Substitute Free(Local(i)) for Bound(0) in the range so that
            # the range can be checked without raw Bound indices.
            range2 = subst_chk(0, Free(local), range_)
            k = _check_is_type(i + 1, new_ctx, range2)
            return VStar(max(j, k))

        # --- Variables ---
        case Free(name):
            entry = context_lookup(ctx, name)
            match entry:
                case HasType(t):
                    return t
                case HasKind():
                    return VStar(0)

        case Bound(_):
            # Should never be reached for well-scoped terms; elaboration
            # converts all Bound vars to Free(Local(...)) before type checking.
            raise TypeCheckError("Unexpected Bound variable in type_inf")  # pragma: no cover

        # --- Application ---
        case App(func, arg):
            func_ty = type_inf(i, ctx, func)
            match func_ty:
                case VPi(domain, range_fn):
                    type_chk(i, ctx, arg, domain)
                    return range_fn(eval_chk(arg, []))
                case _:
                    raise TypeCheckError(
                        f"Application to non-function type: {quote0(func_ty)!r}"
                    )

        # --- Nat ---
        case Nat():
            return VStar(0)

        case Zero():
            return VNat()

        case Succ(n):
            type_chk(i, ctx, n, VNat())
            return VNat()

        case NatElim(motive, base, step, k):
            # motive : Nat → * — we accept any universe
            type_chk(i, ctx, motive, VPi(VNat(), lambda _: VStar(1)))
            motive_v = eval_chk(motive, [])
            # base : motive Zero
            type_chk(i, ctx, base, vapp(motive_v, VZero()))
            # step : (k : Nat) → motive k → motive (Succ k)
            step_ty = VPi(VNat(), lambda kv: VPi(vapp(motive_v, kv),
                                                  lambda _: vapp(motive_v, VSucc(kv))))
            type_chk(i, ctx, step, step_ty)
            # k : Nat
            type_chk(i, ctx, k, VNat())
            return vapp(motive_v, eval_chk(k, []))

        # --- Vec ---
        case Vec(elem_type, length):
            _check_is_type(i, ctx, elem_type)
            type_chk(i, ctx, length, VNat())
            return VStar(0)

        case Nil(elem_type):
            _check_is_type(i, ctx, elem_type)
            a_v = eval_chk(elem_type, [])
            return VVec(a_v, VZero())

        case Cons(elem_type, length, head, tail):
            _check_is_type(i, ctx, elem_type)
            a_v = eval_chk(elem_type, [])
            type_chk(i, ctx, length, VNat())
            n_v = eval_chk(length, [])
            type_chk(i, ctx, head, a_v)
            type_chk(i, ctx, tail, VVec(a_v, n_v))
            return VVec(a_v, VSucc(n_v))

        case VecElim(elem_type, motive, nil_case, cons_case, length, vec):
            _check_is_type(i, ctx, elem_type)
            a_v = eval_chk(elem_type, [])
            # motive : (n : Nat) → Vec A n → *
            motive_ty = VPi(VNat(), lambda nv: VPi(VVec(a_v, nv), lambda _: VStar(1)))
            type_chk(i, ctx, motive, motive_ty)
            motive_v = eval_chk(motive, [])
            # nil_case : motive Zero (Nil A)
            type_chk(i, ctx, nil_case, vapp(vapp(motive_v, VZero()), VNil(a_v)))
            # cons_case : (n:Nat) → (h:A) → (t:Vec A n) →
            #             motive n t → motive (Succ n) (Cons A n h t)
            cons_ty = VPi(VNat(), lambda nv:
                          VPi(a_v, lambda hv:
                          VPi(VVec(a_v, nv), lambda tv:
                          VPi(vapp(vapp(motive_v, nv), tv),
                              lambda _: vapp(vapp(motive_v, VSucc(nv)),
                                             VCons(a_v, nv, hv, tv))))))
            type_chk(i, ctx, cons_case, cons_ty)
            # length : Nat
            type_chk(i, ctx, length, VNat())
            len_v = eval_chk(length, [])
            # vec : Vec A length
            type_chk(i, ctx, vec, VVec(a_v, len_v))
            vec_v = eval_chk(vec, [])
            return vapp(vapp(motive_v, len_v), vec_v)

        # --- Fin ---
        case Fin(n):
            type_chk(i, ctx, n, VNat())
            return VStar(0)

        case FZero(n):
            type_chk(i, ctx, n, VNat())
            n_v = eval_chk(n, [])
            return VFin(VSucc(n_v))

        case FSucc(n, x):
            type_chk(i, ctx, n, VNat())
            n_v = eval_chk(n, [])
            type_chk(i, ctx, x, VFin(n_v))
            return VFin(VSucc(n_v))

        case FinElim(motive, fzero_case, fsucc_case, n, f):
            # motive : (n : Nat) → Fin n → *
            type_chk(i, ctx, n, VNat())
            n_v = eval_chk(n, [])
            motive_ty = VPi(VNat(), lambda nv: VPi(VFin(nv), lambda _: VStar(1)))
            type_chk(i, ctx, motive, motive_ty)
            motive_v = eval_chk(motive, [])
            # fzero_case : (n : Nat) → motive (Succ n) (FZero n)
            fzero_ty = VPi(VNat(), lambda nv:
                           vapp(vapp(motive_v, VSucc(nv)), VFZero(nv)))
            type_chk(i, ctx, fzero_case, fzero_ty)
            # fsucc_case : (n : Nat) → (f : Fin n) →
            #              motive n f → motive (Succ n) (FSucc n f)
            fsucc_ty = VPi(VNat(), lambda nv:
                           VPi(VFin(nv), lambda fv:
                           VPi(vapp(vapp(motive_v, nv), fv),
                               lambda _: vapp(vapp(motive_v, VSucc(nv)),
                                              VFSucc(nv, fv)))))
            type_chk(i, ctx, fsucc_case, fsucc_ty)
            # f : Fin n
            type_chk(i, ctx, f, VFin(n_v))
            f_v = eval_chk(f, [])
            return vapp(vapp(motive_v, n_v), f_v)

        # --- Eq ---
        case Eq(type_, left, right):
            _check_is_type(i, ctx, type_)
            a_v = eval_chk(type_, [])
            type_chk(i, ctx, left, a_v)
            type_chk(i, ctx, right, a_v)
            return VStar(0)

        case Refl(type_, val):
            _check_is_type(i, ctx, type_)
            a_v = eval_chk(type_, [])
            type_chk(i, ctx, val, a_v)
            x_v = eval_chk(val, [])
            return VEq(a_v, x_v, x_v)

        case EqElim(type_, motive, refl_case, left, right, proof):
            _check_is_type(i, ctx, type_)
            a_v = eval_chk(type_, [])
            # motive : (x y : A) → Eq A x y → ∗₁
            # The motive returns a type in ∗₁ rather than ∗₀ so that motives may
            # range over either ordinary types (which inhabit ∗₀ ≤ ∗₁ by
            # cumulativity) or the universe ∗₀ itself (which lives in ∗₁).
            motive_ty = VPi(a_v, lambda xv:
                            VPi(a_v, lambda yv:
                            VPi(VEq(a_v, xv, yv), lambda _: VStar(1))))
            type_chk(i, ctx, motive, motive_ty)
            motive_v = eval_chk(motive, [])
            # refl_case : (z : A) → motive z z (Refl A z)
            refl_ty = VPi(a_v, lambda zv:
                          vapp(vapp(vapp(motive_v, zv), zv), VRefl(a_v, zv)))
            type_chk(i, ctx, refl_case, refl_ty)
            # left, right : A
            type_chk(i, ctx, left, a_v)
            l_v = eval_chk(left, [])
            type_chk(i, ctx, right, a_v)
            r_v = eval_chk(right, [])
            # proof : Eq A left right
            type_chk(i, ctx, proof, VEq(a_v, l_v, r_v))
            p_v = eval_chk(proof, [])
            return vapp(vapp(vapp(motive_v, l_v), r_v), p_v)

        case _:  # pragma: no cover
            raise TypeCheckError(f"type_inf: unhandled term {term!r}")


def type_chk(i: int, ctx: Context, term: CheckTerm, ty: Value) -> None:
    """Check a checkable term against a given type value at binder depth i."""
    match term, ty:
        case Inf(t), _:
            inferred = type_inf(i, ctx, t)
            if not _is_subtype(inferred, ty):
                raise TypeCheckError(
                        f"Type mismatch.\n"
                        f"  Expected: {quote0(ty)!r}\n"
                        f"  Got:      {quote0(inferred)!r}"
                    )

        case Lam(body), VPi(domain, range_fn):
            local = Local(i)
            new_ctx: Context = [(local, HasType(domain))] + ctx
            range_ty = range_fn(VNeutral(NFree(local)))
            # Substitute Free(Local(i)) for Bound(0) in the body so that
            # type_inf sees only Free variables, never raw Bound indices.
            body2 = subst_chk(0, Free(local), body)
            type_chk(i + 1, new_ctx, body2, range_ty)

        case Lam(_), _:
            raise TypeCheckError(
                f"Lambda cannot be checked against non-Pi type: {quote0(ty)!r}"
            )

        case _:  # pragma: no cover
            raise TypeCheckError(f"type_chk: unhandled case {term!r} vs {ty!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_is_type(i: int, ctx: Context, term: CheckTerm) -> int:
    """Check that term is a type (has type VStar(k) for some k).
    Returns the universe level k."""
    match term:
        case Inf(t):
            ty = type_inf(i, ctx, t)
            match ty:
                case VStar(k):
                    return k
                case _:
                    raise TypeCheckError(
                        f"Expected a type (universe), got {quote0(ty)!r}"
                    )
        case Lam(_):
            raise TypeCheckError("A lambda is not a type")
        case _:  # pragma: no cover
            raise TypeCheckError(f"_check_is_type: unhandled {term!r}")
            
# ---------------------------------------------------------------------------
# Universe-level-agnostic equality
# ---------------------------------------------------------------------------

def _is_subtype(sub: Value, sup: Value) -> bool:
    """Return True iff `sub` is a subtype of `sup` under cumulativity.

    Both sides are first quoted to normal forms so that HOAS closures
    inside Pi types are opened in lock-step under fresh probe binders.
    The remaining work is a structural traversal that propagates variance.
    """
    return _subtype_chk(0, quote0(sub), quote0(sup))


def _subtype_chk(i: int, t1: CheckTerm, t2: CheckTerm) -> bool:
    """Subtype check on CheckTerms at binder depth i."""
    # Fast path: definitional equality always implies subtyping.
    if t1 == t2:
        return True
    match t1, t2:
        case Inf(a), Inf(b):
            return _subtype_inf(i, a, b)
        case Lam(a), Lam(b):
            # Lambdas appear as types only when they are stuck (i.e. a
            # higher-kinded value applied to a neutral).  Two such
            # lambdas are subtypes iff their bodies are subtypes at the
            # next binder depth.
            return _subtype_chk(i + 1, a, b)
        case _:
            return False


def _subtype_inf(i: int, a: InferTerm, b: InferTerm) -> bool:
    """Subtype check on InferTerms at binder depth i."""
    if a == b:
        return True
    match a, b:
        # Universes: cumulativity.
        case Star(j), Star(k):
            return j <= k

        # Dependent function: contravariant domain, covariant codomain.
        case Pi(da, ra), Pi(db, rb):
            # Domain reverses direction (sub becomes sup and vice versa).
            # `ra` and `rb` are checked under one additional binder.
            return (_subtype_chk(i, db, da)
                    and _subtype_chk(i + 1, ra, rb))

        # Annotation: subtype on both the term and its annotation.
        case Ann(ea, ta), Ann(eb, tb):
            return (_subtype_chk(i, ea, eb)
                    and _subtype_chk(i, ta, tb))

        # Application: invariant — applied type families are invariant in
        # their arguments unless the language records variance.  We still
        # allow subtyping in the head position so e.g.  Vec * <: Vec *
        # when both heads quote to the same neutral.
        case App(fa, aa), App(fb, ab):
            return _subtype_inf(i, fa, fb) and _defeq_chk(aa, ab)

        # --- Invariant type formers --------------------------------------
        # For Vec, Eq, Fin and Nat, the carrier and indices are invariant.
        # Subtyping at these positions would change which canonical values
        # inhabit the type, breaking subject reduction.
        case Vec(ea, la), Vec(eb, lb):
            return _defeq_chk(ea, eb) and _defeq_chk(la, lb)
        case Eq(ta, la, ra), Eq(tb, lb, rb):
            return (_defeq_chk(ta, tb) and _defeq_chk(la, lb)
                    and _defeq_chk(ra, rb))
        case Fin(na), Fin(nb):
            return _defeq_chk(na, nb)
        case Nat(), Nat():
            return True

        # Variables — Bound indices must match exactly; free variables
        # compared by name.  No subtyping at this level.
        case Bound(j), Bound(k):
            return j == k
        case Free(na), Free(nb):
            return na == nb

        # Anything else falls back to definitional equality, recovering
        # reflexivity for term-level constructors that can appear inside
        # type expressions (e.g. eliminator targets).
        case _:
            return a == b


def _defeq_chk(t1: CheckTerm, t2: CheckTerm) -> bool:
    """Definitional equality (no cumulativity).  Used at invariant
    positions inside the subtype relation."""
    return t1 == t2
