"""Abstract syntax for LambdaPi: terms, values, and neutrals.

Design decisions:
- InferTerm / CheckTerm are distinct Python type aliases (bidirectional discipline).
- Values use HOAS: VLam and VPi hold Python callables (closures), so beta-reduction
  is just Python function application — no hand-rolled substitution at runtime.
- VLam and VPi use identity-based equality (eq=False) because their Callable fields
  are not structurally comparable.  All other Value / Neutral nodes are frozen=True.
- Definitional equality is checked via quotation: quote0(v1) == quote0(v2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Global:
    """A top-level name (from `let` or `assume`)."""
    name: str


@dataclass(frozen=True)
class Local:
    """A locally-introduced name, created when opening a Pi binder during type
    checking.  The integer is the current binder depth `i`."""
    index: int


@dataclass(frozen=True)
class Quote:
    """Used *only* inside `boundfree` during quotation to convert HOAS back to
    de Bruijn.  Never appears in user-written terms."""
    index: int


Name = Global | Local | Quote

# ---------------------------------------------------------------------------
# Check terms  (types are *inputs* — checked against a given type)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inf:
    """Embed an inferable term in a checkable position."""
    term: "InferTerm"


@dataclass(frozen=True)
class Lam:
    """Lambda abstraction.  The body has one extra de Bruijn variable in scope
    (Bound(0) refers to this lambda's argument)."""
    body: "CheckTerm"


CheckTerm = Inf | Lam

# ---------------------------------------------------------------------------
# Infer terms  (types are *outputs* — synthesised)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ann:
    """Type annotation  (expr : type)."""
    expr: CheckTerm
    type: CheckTerm


@dataclass(frozen=True)
class Star:
    """Universe at level n.  Star(n) : Star(n+1)."""
    level: int


@dataclass(frozen=True)
class Pi:
    """Dependent product  ∀x:domain.range.
    `range` has one extra de Bruijn variable in scope."""
    domain: CheckTerm
    range: CheckTerm


@dataclass(frozen=True)
class Bound:
    """De Bruijn index (0 = innermost binder)."""
    index: int


@dataclass(frozen=True)
class Free:
    """Free variable, referenced by name."""
    name: Name


@dataclass(frozen=True)
class App:
    """Application."""
    func: "InferTerm"
    arg: CheckTerm


# --- Nat ---

@dataclass(frozen=True)
class Nat:
    """The type of natural numbers."""


@dataclass(frozen=True)
class Zero:
    """The zero constructor."""


@dataclass(frozen=True)
class Succ:
    """The successor constructor."""
    n: CheckTerm


@dataclass(frozen=True)
class NatElim:
    """Natural number eliminator."""
    motive: CheckTerm
    base: CheckTerm
    step: CheckTerm
    k: CheckTerm


# --- Vec ---

@dataclass(frozen=True)
class Vec:
    """Length-indexed vector type."""
    elem_type: CheckTerm
    length: CheckTerm


@dataclass(frozen=True)
class Nil:
    """Empty vector."""
    elem_type: CheckTerm


@dataclass(frozen=True)
class Cons:
    """Vector cons cell."""
    elem_type: CheckTerm
    length: CheckTerm
    head: CheckTerm
    tail: CheckTerm


@dataclass(frozen=True)
class VecElim:
    """Vector eliminator."""
    elem_type: CheckTerm
    motive: CheckTerm
    nil_case: CheckTerm
    cons_case: CheckTerm
    length: CheckTerm
    vec: CheckTerm


# --- Fin ---

@dataclass(frozen=True)
class Fin:
    """Finite ordinal type:  Fin n has exactly n inhabitants."""
    n: CheckTerm


@dataclass(frozen=True)
class FZero:
    """First element of Fin (Succ n)."""
    n: CheckTerm


@dataclass(frozen=True)
class FSucc:
    """Successor element of Fin."""
    n: CheckTerm
    x: CheckTerm


@dataclass(frozen=True)
class FinElim:
    """Finite ordinal eliminator."""
    motive: CheckTerm
    fzero_case: CheckTerm
    fsucc_case: CheckTerm
    n: CheckTerm
    f: CheckTerm


# --- Eq (propositional equality / Martin-Löf identity type) ---

@dataclass(frozen=True)
class Eq:
    """Equality type  Eq A x y."""
    type: CheckTerm
    left: CheckTerm
    right: CheckTerm


@dataclass(frozen=True)
class Refl:
    """Reflexivity proof  Refl A x : Eq A x x."""
    type: CheckTerm
    val: CheckTerm


@dataclass(frozen=True)
class EqElim:
    """Equality eliminator (J rule)."""
    type: CheckTerm
    motive: CheckTerm
    refl_case: CheckTerm
    left: CheckTerm
    right: CheckTerm
    proof: CheckTerm


InferTerm = (
    Ann | Star | Pi | Bound | Free | App
    | Nat | Zero | Succ | NatElim
    | Vec | Nil | Cons | VecElim
    | Fin | FZero | FSucc | FinElim
    | Eq | Refl | EqElim
)

# ---------------------------------------------------------------------------
# Values  (semantic domain, produced by evaluation)
# ---------------------------------------------------------------------------


@dataclass
class VLam:
    """Lambda value.  HOAS: the body is a Python callable Value -> Value.
    Identity-equality: two closures are never structurally equal."""
    fn: Callable[["Value"], "Value"]

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class VPi:
    """Pi (dependent function) value.
    Identity-equality for the same reason as VLam."""
    domain: "Value"
    range_fn: Callable[["Value"], "Value"]

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=True)
class VStar:
    """Universe value."""
    level: int


@dataclass(frozen=True)
class VNeutral:
    """A stuck computation (neutral term)."""
    neutral: "Neutral"


@dataclass(frozen=True)
class VNat:
    """The Nat type as a value."""


@dataclass(frozen=True)
class VZero:
    """The Zero constructor value."""


@dataclass(frozen=True)
class VSucc:
    """The Succ constructor value."""
    pred: "Value"


@dataclass(frozen=True)
class VNil:
    """The Nil constructor value."""
    elem_type: "Value"


@dataclass(frozen=True)
class VCons:
    """The Cons constructor value."""
    elem_type: "Value"
    length: "Value"
    head: "Value"
    tail: "Value"


@dataclass(frozen=True)
class VVec:
    """The Vec type value."""
    elem_type: "Value"
    length: "Value"


@dataclass(frozen=True)
class VFin:
    """The Fin type value."""
    n: "Value"


@dataclass(frozen=True)
class VFZero:
    """FZero constructor value."""
    n: "Value"


@dataclass(frozen=True)
class VFSucc:
    """FSucc constructor value."""
    n: "Value"
    x: "Value"


@dataclass(frozen=True)
class VEq:
    """The Eq type value."""
    type: "Value"
    left: "Value"
    right: "Value"


@dataclass(frozen=True)
class VRefl:
    """The Refl constructor value."""
    type: "Value"
    val: "Value"


Value = (
    VLam | VPi | VStar | VNeutral
    | VNat | VZero | VSucc
    | VNil | VCons | VVec
    | VFin | VFZero | VFSucc
    | VEq | VRefl
)

# ---------------------------------------------------------------------------
# Neutrals  (stuck computations: a free variable applied to arguments)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NFree:
    """A free variable."""
    name: Name


@dataclass(frozen=True)
class NApp:
    """A neutral applied to a value argument."""
    func: "Neutral"
    arg: "Value"


@dataclass(frozen=True)
class NNatElim:
    """natElim stuck on a neutral natural number."""
    motive: "Value"
    base: "Value"
    step: "Value"
    k: "Neutral"


@dataclass(frozen=True)
class NVecElim:
    """vecElim stuck on a neutral vector."""
    elem_type: "Value"
    motive: "Value"
    nil_case: "Value"
    cons_case: "Value"
    length: "Value"
    vec: "Neutral"


@dataclass(frozen=True)
class NFinElim:
    """finElim stuck on a neutral finite ordinal."""
    motive: "Value"
    fzero_case: "Value"
    fsucc_case: "Value"
    n: "Value"
    f: "Neutral"


@dataclass(frozen=True)
class NEqElim:
    """eqElim stuck on a neutral equality proof."""
    type: "Value"
    motive: "Value"
    refl_case: "Value"
    left: "Value"
    right: "Value"
    proof: "Neutral"


Neutral = NFree | NApp | NNatElim | NVecElim | NFinElim | NEqElim
