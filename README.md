# LambdaPi-Py

A Python 3.11+ implementation of a dependently typed lambda calculus, following the paper:

> Löh, McBride & Swierstra — *"A Tutorial Implementation of a Dependently Typed Lambda Calculus"*  
> Fundamenta Informaticae, vol. 102, pp. 177–207, 2010.

It strives to be fully compatible with this haskell implementation: [LambdaPi](https://github.com/ilya-klyuchnikov/lambdapi)

## Features

- **Bidirectional type checker** — separate `InferTerm` / `CheckTerm` hierarchies enforce the infer/check discipline at the Python type level
- **Normalisation-by-evaluation (NbE)** — values use HOAS (Python closures); beta-reduction is plain function application, no hand-rolled substitution at runtime
- **Definitional equality** via quotation: `quote0(v1) == quote0(v2)` on frozen dataclasses
- **Universe hierarchy** `Type 0 : Type 1 : Type 2 : …` — avoids the `Type : Type` inconsistency
- **Built-in datatypes**: `Nat`, `Vec` (length-indexed), `Fin` (finite ordinals), `Eq` (propositional equality / J rule)
- **Surface syntax** parsed with Lark (Earley), elaborated to locally-nameless de Bruijn terms
- **CLI** with interactive REPL and batch file execution

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11, `lark >= 1.1`, `rich >= 13`.

## Quick Start

### REPL (read-eval-print loop)

```
$ lambdapy repl
LambdaPi REPL  (:help for commands)
λΠ> eval Zero
Zero : Nat
λΠ> eval Succ (Succ (Succ Zero)) : Nat
3 : Nat
λΠ> assume A : Type
Assumed: A : Type
λΠ> assume x : A
Assumed: x : A
λΠ> eval x
x : A
λΠ> eval Refl Nat Zero
Refl Nat Zero : Eq Nat Zero Zero
λΠ> :quit
Bye.
```

### Batch files

```bash
lambdapy run examples/02_nat.lp
# 2 : Nat
# 3 : Nat
```

### Programmatic use

```python
from lambdapy.parser import parse
from lambdapy.repl import fresh_state, execute

state = fresh_state()
for stmt in parse("assume n : Nat\neval Succ n"):
    state, output = execute(state, stmt)
    print(output)
# Assumed: n : Nat
# Succ n : Nat
```

## Surface Syntax

### Statements

```
assume name₁ name₂ ... : Type     -- introduce axioms
let name : Type = body             -- define a name
eval term                          -- evaluate and print with type
check term                         -- type-check and print type
```

### Terms

| Syntax | Meaning |
|--------|---------|
| `Type` / `Type 0` | Universe at level 0 |
| `Type 1`, `Type 2`, … | Higher universes |
| `\x -> body` | Lambda abstraction |
| `\x y z -> body` | Curried lambda |
| `f a b` | Left-associative application |
| `A -> B` | Non-dependent function type |
| `forall (x : A). B` | Dependent function type (Pi) |
| `forall (x : A) (y : B). C` | Nested Pi |
| `expr : Type` | Type annotation (makes a term inferable) |

### Built-in types and terms

| Term | Type |
|------|------|
| `Nat` | `Type` |
| `Zero` | `Nat` |
| `Succ n` | `Nat` |
| `NatElim P z s n` | `P n` |
| `Vec A n` | `Type` |
| `Nil A` | `Vec A Zero` |
| `Cons A n h t` | `Vec A (Succ n)` |
| `VecElim A P nil cons n v` | `P n v` |
| `Fin n` | `Type` |
| `FZero n` | `Fin (Succ n)` |
| `FSucc n x` | `Fin (Succ n)` |
| `FinElim P fz fs n f` | `P n f` |
| `Eq A x y` | `Type` |
| `Refl A x` | `Eq A x x` |
| `EqElim A P r x y p` | `P x y p` |

## Examples

### Natural number addition

```
-- plus m n = NatElim (\_.Nat) n (\k.\r. Succ r) m
eval NatElim (\_ -> Nat) (Succ Zero) (\k -> \r -> Succ r) (Succ (Succ Zero)) : Nat
-- 3 : Nat
```

### Polymorphic identity

```
eval (\A -> \x -> x) : forall (A : Type). A -> A
```

### Length-indexed vector

```
eval Cons Nat Zero Zero (Nil Nat)
-- Cons Nat Zero Zero (Nil Nat) : Vec Nat 1
```

### Propositional equality

```
eval Refl Nat Zero
-- Refl Nat Zero : Eq Nat Zero Zero

eval EqElim Nat (\x -> \y -> \p -> Nat) (\z -> Zero) Zero Zero (Refl Nat Zero) : Nat
-- Zero : Nat
```

## Architecture

```
lambdapy/
├── syntax.py      # AST: InferTerm, CheckTerm, Value, Neutral, Name
├── eval.py        # NbE evaluator: eval_inf, eval_chk, vapp, *_elim helpers
├── quote.py       # Quotation: quote0, boundfree — basis for definitional equality
├── subst.py       # Substitution: subst_inf, subst_chk (used by elaborator)
├── check.py       # Bidirectional type checker: type_inf, type_chk
├── context.py     # Context (association list of Name → HasType | HasKind)
├── errors.py      # TypeCheckError, EvalError, ParseError
├── grammar.lark   # Lark Earley grammar
├── parser.py      # Parser + elaborator (named binders → de Bruijn)
├── pretty.py      # Value/Term → human-readable string
├── prelude.py     # (reserved for future built-in definitions)
├── repl.py        # Interactive REPL (ReplState, execute, run_file)
└── cli.py         # CLI entry point (lambdapy repl/run/check)
```

### Key design decisions

**HOAS for values.**  `VLam(fn: Callable[[Value], Value])` and `VPi(domain, range_fn)` store Python closures. Beta-reduction is `fn(arg)` — no explicit substitution at runtime. Closures use `eq=False` (identity equality); all other value nodes are `frozen=True` (structural equality).

**Locally-nameless terms.**  `Bound(i)` for de Bruijn-indexed bound variables; `Free(Global(name))` for top-level names. The elaborator converts surface named binders to de Bruijn by maintaining a scope stack. The type checker opens Pi binders with fresh `Free(Local(i))` names via `subst_chk`, so `type_inf` never sees raw `Bound` indices.

**Bidirectional typing.**  `type_inf(i, ctx, t) -> Value` synthesises; `type_chk(i, ctx, t, ty) -> None` checks. `Lam` is only in `CheckTerm`; `Ann`, `Pi`, `App` are only in `InferTerm`. The Python union-type aliases enforce this at the type-checker level.

**Definitional equality.**  `quote0(v1) == quote0(v2)` converts both values to their normal forms as `CheckTerm` trees and uses structural dataclass equality. HOAS closures are normalised by applying them to fresh `VNeutral(NFree(Quote(i)))` probes.

**Universe hierarchy.**  `Star(n) : Star(n+1)`. Pi types have level `max(j, k)` where `j` and `k` are the universe levels of their domain and range. This rules out `Type : Type`.

## REPL Commands

| Command | Description |
|---------|-------------|
| `:help` | Show available commands |
| `:context` | List all assumptions and definitions |
| `:load FILE` | Load and execute a `.lp` file |
| `:reset` | Clear the context |
| `:quit` / `:q` | Exit |

## Running Tests

```bash
pytest tests/             # run all 136 tests
pytest tests/ -v          # verbose
pytest tests/ --cov=lambdapy --cov-report=term-missing
```

## Reference

- Löh, A., McBride, C., & Swierstra, W. (2010). [A tutorial implementation of a dependently typed lambda calculus](https://www.andres-loeh.de/LambdaPi/). *Fundamenta Informaticae*, 102(2), 177–207.
