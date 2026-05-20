# %% Imports
import io

from lambdapy.repl import fresh_state, execute, run_file
from lambdapy.parser import (
    parse, 
    _make_nat_elim_fn, 
    _make_vec_elim_fn, 
    _make_vec_elim_fn,
    _make_eq_elim_fn,
    make_fin_elim_type,
    _make_fin_elim_fn,
)
from lambdapy.errors import TypeCheckError, EvalError, ParseError
from lambdapy.check import type_chk, type_inf0
from lambdapy.syntax import *
from lambdapy.nameenv import with_name_env
from lambdapy.pretty import pretty_value, pretty_infer
from lambdapy.quote import quote0

# %%
# https://www.andres-loeh.de/LambdaPi/prelude.lp
# Changes applied
prelude = """\
let id    = (\ a x -> x) :: forall (a :: *) . a -> a
let const = (\ a b x y -> x) :: forall (a :: *) (b :: *) . a -> b -> a
let plus = natElim ( \ _ -> Nat -> Nat ) ( \ n -> n ) ( \ p rec n -> Succ (rec n) )
let pred = natElim ( \ _ -> Nat ) Zero ( \ np _rec -> np )
let natFold = ( \ m mz ms -> natElim ( \ _ -> m ) mz ( \ np rec -> ms rec ) ) :: forall (m :: *) . m -> (m -> m) -> Nat -> m
let nat1Elim = ( \ m m0 m1 ms -> natElim m m0 (\ p rec -> natElim (\ n -> m (Succ n)) m1 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> (forall (n :: Nat) . m (Succ n) -> m (Succ (Succ n))) -> forall (n :: Nat) . m n
let nat2Elim = ( \ m m0 m1 m2 ms -> nat1Elim m m0 m1 (\ p rec -> natElim (\ n -> m (Succ (Succ n))) m2 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> m (Succ (Succ Zero)) -> (forall (n :: Nat) . m (Succ (Succ n)) -> m (Succ (Succ (Succ n)))) -> forall (n :: Nat) . m n
let inc = natFold Nat (Succ Zero) Succ
let finNat = finElim (\ _ _ -> Nat) (\ _ -> Zero) (\ _ _ rec -> Succ rec)
let Unit = Fin (Succ Zero)
let U = FZero Zero
let unitElim = ( \ m mu -> finElim ( nat1Elim (\ n -> Fin n -> *) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) ) ( natElim (\ n -> natElim (\ n -> Fin (Succ n) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FZero n)) mu (\ _ _ -> U) ) ( \ n f _ -> finElim (\ n f -> natElim (\ n -> Fin (Succ n) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc n f)) (\ _ -> U) (\ _ _ _ -> U) n f ) (Succ Zero) ) :: forall (m :: Unit -> *) . m U -> forall (u :: Unit) . m u
let Void = Fin Zero
let voidElim = ( \ m -> finElim (natElim (\ n -> Fin n -> *) (\ x -> m x) (\ _ _ _ -> Unit)) (\ _ -> U) (\ _ _ _ -> U) Zero ) :: forall (m :: Void -> *) (v :: Void) . m v
let Bool = Fin (Succ (Succ Zero))
let False = FZero (Succ Zero)
let True  = FSucc (Succ Zero) (FZero Zero)
let boolElim = ( \ m mf mt -> finElim ( nat2Elim (\ n -> Fin n -> *) (\ _ -> Unit) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) ) ( nat1Elim ( \ n -> nat1Elim (\ n -> Fin (Succ n) -> *) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) n (FZero n)) U mf (\ _ _ -> U) ) ( \ n f _ -> finElim ( \ n f -> nat1Elim (\ n -> Fin (Succ n) -> *) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc n f) ) ( natElim ( \ n -> natElim (\ n -> Fin (Succ (Succ n)) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc (Succ n) (FZero n)) ) mt (\ _ _ -> U) ) ( \ n f _ -> finElim (\ n f -> natElim (\ n -> Fin (Succ (Succ n)) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc (Succ n) (FSucc n f))) (\ _ -> U) (\ _ _ _ -> U) n f ) n f ) (Succ (Succ Zero)) ) :: forall (m :: Bool -> *) . m False -> m True -> forall (b :: Bool) . m b
let not = boolElim (\ _ -> Bool) True False
let and = boolElim (\ _ -> Bool -> Bool) (\ _ -> False) (id Bool)
let or  = boolElim (\ _ -> Bool -> Bool) (id Bool) (\ _ -> True)
let iff = boolElim (\ _ -> Bool -> Bool) not (id Bool)
let xor = boolElim (\ _ -> Bool -> Bool) (id Bool) not
let even    = natFold Bool True not
let odd     = natFold Bool False not
let isZero  = natFold Bool True (\ _ -> False)
let isSucc  = natFold Bool False (\ _ -> True)
let natEq = natElim ( \ _ -> Nat -> Bool ) ( natElim ( \ _ -> Bool ) True ( \ np _ -> False ) ) ( \ mp rec_mp -> natElim ( \ _ -> Bool ) False ( \ np _ -> rec_mp np ))
let Prop = boolElim (\ _ -> *) Void Unit
let pNatEqRefl = natElim (\ n -> Prop (natEq n n)) U (\ np rec -> rec) :: forall (n :: Nat) . Prop (natEq n n)
let Not = (\ a -> a -> Void) :: * -> *
let leibniz = ( \ a b f -> eqElim a (\ x y eq_x_y -> Eq b (f x) (f y)) (\ x -> Refl b (f x)) ) :: forall (a :: *) (b :: Type 1) (f :: a -> b) (x :: a) (y :: a) . Eq a x y -> Eq b (f x) (f y)
let symm = ( \ a -> eqElim a (\ x y eq_x_y -> Eq a y x) (\ x -> Refl a x) ) :: forall (a :: *) (x :: a) (y :: a) . Eq a x y -> Eq a y x
let tran = ( \ a x y z eq_x_y -> eqElim a (\ x y eq_x_y -> forall (z :: a) . Eq a y z -> Eq a x z) (\ x z eq_x_z -> eq_x_z) x y eq_x_y z ) :: forall (a :: *) (x :: a) (y :: a) (z :: a) . Eq a x y -> Eq a y z -> Eq a x z
let apply = eqElim * (\ a b _ -> a -> b) (\ _ x -> x) :: forall (a :: *) (b :: *) (p :: Eq * a b) . a -> b
let p1IsNot0 = (\ p -> apply Unit Void (leibniz Nat * (natElim (\ _ -> *) Void (\ _ _ -> Unit)) (Succ Zero) Zero p) U) :: Not (Eq Nat (Succ Zero) Zero)
let p0IsNot1 = (\ p -> p1IsNot0 (symm Nat Zero (Succ Zero) p)) :: Not (Eq Nat Zero (Succ Zero))
let p0IsNoSucc = natElim ( \ n -> Not (Eq Nat Zero (Succ n)) ) p0IsNot1 ( \ np rec_np eq_0_SSnp -> rec_np (leibniz Nat Nat pred Zero (Succ (Succ np)) eq_0_SSnp) )
let replicate = ( natElim ( \ n -> forall (a :: *) . a -> Vec a n ) ( \ a _ -> Nil a ) ( \ np rec_np a x -> Cons a np x (rec_np a x) ) ) :: forall (n :: Nat) . forall (a :: *) . a -> Vec a n
let fromto = natElim ( \ n -> Vec Nat n ) ( Nil Nat ) ( \ np rec_np -> Cons Nat np np rec_np )
let append = ( \ a -> vecElim a (\ m _ -> forall (n :: Nat) . Vec a n -> Vec a (plus m n)) (\ _ v -> v) (\ m v vs rec n w -> Cons a (plus m n) v (rec n w))) ::  forall (a :: *) (m :: Nat) (v :: Vec a m) (n :: Nat) (w :: Vec a n). Vec a (plus m n)
let tailp = (\ a -> vecElim a ( \ m v -> forall (n :: Nat) . Eq Nat m (Succ n) -> Vec a n ) ( \ n eq_0_SuccN -> voidElim ( \ _ -> Vec a n ) ( p0IsNoSucc n eq_0_SuccN ) ) ( \ mp v vs rec_mp n eq_SuccMp_SuccN -> eqElim Nat (\ mp n e -> Vec a mp -> Vec a n) (\ _ v -> v) mp n (leibniz Nat Nat pred (Succ mp) (Succ n) eq_SuccMp_SuccN) vs)) :: forall (a :: *) (m :: Nat) . Vec a m -> forall (n :: Nat) . Eq Nat m (Succ n) -> Vec a n
let tail = (\ a n v -> tailp a (Succ n) v n (Refl Nat (Succ n))) :: forall (a :: *) (n :: Nat) . Vec a (Succ n) -> Vec a n
let at = (\ a -> vecElim a ( \ n v -> Fin n -> a ) ( \ f -> voidElim (\ _ -> a) f ) ( \ np v vs rec_np f_SuccNp -> finElim ( \ n _ -> Eq Nat n (Succ np) -> a ) ( \ n e -> v ) ( \ n f_N _ eq_SuccN_SuccNp -> rec_np (eqElim Nat (\ x y e -> Fin x -> Fin y) (\ _ f -> f) n np (leibniz Nat Nat pred (Succ n) (Succ np) eq_SuccN_SuccNp) f_N)) (Succ np) f_SuccNp (Refl Nat (Succ np)))) :: forall (a :: *) (n :: Nat) . Vec a n -> Fin n -> a
let head = (\ a n v -> at a (Succ n) v (FZero n)) :: forall (a :: *) (n :: Nat) . Vec a (Succ n) -> a
let map = (\ a b f -> vecElim a ( \ n _ -> Vec b n ) ( Nil b ) ( \ n x _ rec -> Cons b n (f x) rec )) :: forall (a :: *) (b :: *) (f :: a -> b) (n :: Nat) . Vec a n -> Vec b n
let p0PlusNisN = Refl Nat :: forall (n :: Nat) . Eq Nat (plus Zero n) n
let pNPlus0isN = natElim ( \ n -> Eq Nat (plus n Zero) n ) ( Refl Nat Zero ) ( \ np rec -> leibniz Nat Nat Succ (plus np Zero) np rec ) :: forall (n :: Nat) . Eq Nat (plus n Zero) n
assume (k1 :: Nat)
assume (k2 :: Nat)
assume (z :: Eq Nat k1 k2)
eval symm
let pp = Eq Nat (plus k1 Zero) k1
let zz = pNPlus0isN k1 :: pp
"""

# %% Execute statements
state = fresh_state()
for i, stmt in enumerate(parse(prelude)):
    try:
        state, output = execute(state, stmt)
        print(f"  [{i:2d}] OK: {output[:120]}")
    except (TypeCheckError, EvalError, ParseError) as exc:
        print(f"  [{i:2d}] ERROR: {type(exc).__name__}: {exc}")
        print(prelude.split("\n")[i])
    except Exception as exc:
        print(f"  [{i:2d}] UNEXPECTED ERROR: {type(exc).__name__}: {exc}")

# %% Reduction check/smoke test ensuring evaluations work on the unitElim / boolElim / append outputs
src = [
    "eval unitElim (\\ _ -> Nat) (Succ Zero) U\n",   # should reduce to Succ Zero
    "eval boolElim (\\ _ -> Nat) Zero (Succ Zero) True\n",  # should be Succ Zero
    "eval boolElim (\\ _ -> Nat) Zero (Succ Zero) False\n", # should be Zero
    "eval p1IsNot0 :: Eq Nat (Succ Zero) Zero -> Void\n"
    # append usage
    "eval append Nat (Succ Zero) (Cons Nat Zero Zero (Nil Nat)) (Succ Zero) (Cons Nat Zero (Succ Zero) (Nil Nat))\n",
]
for stmt in parse("".join(src)):
    state, out = execute(state, stmt)
    print(out)

# %% Run the script end-to-end through run_file
out = io.StringIO()
run_file("./examples/prelude_and_examples.lp", out)
print(out.getvalue())

# %% test apply
apply_src = """\
let apply = eqElim * (\\ a b _ -> a -> b) (\\ _ x -> x) :: forall (a :: *) (b :: *) (p :: Eq * a b) . a -> b
eval apply Nat Nat (Refl * Nat) Zero
"""

state = fresh_state()
for stmt in parse(apply_src):
    state, out = execute(state, stmt)
    print(out)
    
# %% test append
append_src = """\
let plus = (\\ m n -> natElim (\\ _ -> Nat) n (\\ _ r -> Succ r) m) :: forall (m :: Nat) (n :: Nat) . Nat
let append = ( \\ a -> vecElim a (\\ m _ -> forall (n :: Nat) . Vec a n -> Vec a (plus m n)) (\\ _ v -> v) (\\ m v vs rec n w -> Cons a (plus m n) v (rec n w))) ::  forall (a :: *) (m :: Nat) (v :: Vec a m) (n :: Nat) (w :: Vec a n). Vec a (plus m n)
"""

state = fresh_state()
for stmt in parse(append_src):
    state, out = execute(state, stmt)
    print(out)

# %% test leibniz
# leibniz :: forall (x :: *)
#                   (y :: *)
#                   (z :: forall z :: x . y)
#                   (a :: x)
#                   (b :: x)
#                   (c :: Eq x a b) .
#            Eq y (z a) (z b)
leipniz_src = """\
let leibniz = (\\ a b f x y p -> eqElim a (\\ x y _ -> Eq b (f x) (f y)) (\\ x -> Refl b (f x)) x y p) :: forall (a :: *) (b :: Type 1) (f :: a -> b) (x :: a) (y :: a) (p :: Eq a x y) . Eq b (f x) (f y)
let leibniz = ( \ a b f -> EqElim a (\ x y eq_x_y -> Eq b (f x) (f y)) (\ x -> Refl b (f x)) ) :: forall (a :: *) (b :: *) (f :: a -> b) (x :: a) (y :: a) . Eq a x y -> Eq b (f x) (f y)
let leibniz = ( \ a b f -> eqElim a (\ x y eq_x_y -> Eq b (f x) (f y)) (\ x -> Refl b (f x)) ) :: forall (a :: *) (b :: Type 1) (f :: a -> b) (x :: a) (y :: a) . Eq a x y -> Eq b (f x) (f y)
"""
# leibniz : forall (x : Type). forall (y : Type 1). forall (z : x -> y). forall (w : x). forall (u : x). Eq x w u -> Eq y (z w) (z u)

state = fresh_state()
for stmt in parse(leipniz_src):
    state, out = execute(state, stmt)
    print(out)


# %% test p1IsNot0
# defining leibniz with b at Type 1
p1_src = """\
let plus = (\\ m n -> natElim (\\ _ -> Nat) n (\\ _ r -> Succ r) m) :: forall (m :: Nat) (n :: Nat) . Nat
let Unit = ( Fin (Succ Zero) ) :: *
let U = ( FZero Zero ) :: Fin (Succ Zero)
let Void = ( Fin Zero ) :: *
let Not = (\\ a -> a -> Fin Zero) :: * -> *
let leibniz = ( \ a b f -> eqElim a (\ x y eq_x_y -> Eq b (f x) (f y)) (\ x -> Refl b (f x)) ) :: forall (a :: *) (b :: Type 1) (f :: a -> b) (x :: a) (y :: a) . Eq a x y -> Eq b (f x) (f y)
let apply = (eqElim * (\\ a b _ -> a -> b) (\\ _ x -> x)) :: forall (a :: *) (b :: *) (p :: Eq * a b) . a -> b
let p1IsNot0 = (\\ p -> apply Unit Void (leibniz Nat * (natElim (\\ _ -> *) Void (\\ _ _ -> Unit)) (Succ Zero) Zero p) U) :: Not (Eq Nat (Succ Zero) Zero)
"""

state = fresh_state()
for stmt in parse(p1_src):
    state, out = execute(state, stmt)
    print(out)

# %% Test nat1Elim
# let nat1Elim =
#   ( \ m m0 m1 ms -> natElim m m0
#                             (\ p rec -> natElim (\ n -> m (Succ n)) m1 ms p) )
#   :: forall (m :: Nat -> *) . m 0 -> m 1 ->
#      (forall n :: Nat . m (Succ n) -> m (Succ (Succ n))) ->
#      forall (n :: Nat) . m n
# nat1Elim :: forall (x :: forall x :: Nat . *)
#                    (y :: x 0)
#                    (z :: x 1)
#                    (a :: forall (a :: Nat) (b :: x (1 + a)) . x (2 + a))
#                    (b :: Nat) .
#             x b
nat1Elim_src = """\
let nat1Elim = ( \ m m0 m1 ms -> natElim m m0 (\ p rec -> natElim (\ n -> m (Succ n)) m1 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> (forall (n :: Nat) . m (Succ n) -> m (Succ (Succ n))) -> forall (n :: Nat) . m n
"""

state = fresh_state()
for stmt in parse(nat1Elim_src):
    state, out = execute(state, stmt)
    print(out)

# %% Test nat2Elim
# nat2Elim :: forall (x :: forall x :: Nat . *)
#                    (y :: x 0)
#                    (z :: x 1)
#                    (a :: x 2)
#                    (b :: forall (b :: Nat) (c :: x (2 + b)) . x (3 + b))
#                    (c :: Nat) .
#             x c
nat2Elim_src = """\
let nat1Elim = ( \ m m0 m1 ms -> natElim m m0 (\ p rec -> natElim (\ n -> m (Succ n)) m1 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> (forall (n :: Nat) . m (Succ n) -> m (Succ (Succ n))) -> forall (n :: Nat) . m n
let nat2Elim = ( \\ m m0 m1 m2 ms -> nat1Elim m m0 m1 (\\ p rec -> natElim (\\ n -> m (Succ (Succ n))) m2 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> m (Succ (Succ Zero)) -> (forall (n :: Nat) . m (Succ (Succ n)) -> m (Succ (Succ (Succ n)))) -> forall (n :: Nat) . m n
"""

state = fresh_state()
for stmt in parse(nat2Elim_src):
    state, out = execute(state, stmt)
    print(out)




# %% # Let's also run some evaluation tests to make sure the evaluator works correctly
test_stmts = [
    # Test plus
    "eval plus Zero Zero",
    "eval plus (Succ Zero) (Succ Zero)",
    "eval plus (Succ (Succ Zero)) (Succ (Succ (Succ Zero)))",
    # Test pred
    "eval pred Zero",
    "eval pred (Succ (Succ Zero))",
    # Test inc
    "eval inc Zero",
    "eval inc (Succ Zero)",
    # Test natFold
    "eval even Zero",  # should be True  (FSucc 1 (FZero 0))
    "eval even (Succ Zero)",   # should be False  (FSucc 1 (FZero 0))
    "eval even (Succ (Succ Zero))",  # should be True
    # Test not
    "eval not True",  # shopuld be False
    "eval not False",  # should be True
    # Test and
    "eval and True True",
    "eval and True False",
    # Test natEq
    "eval natEq Zero Zero",
    "eval natEq Zero (Succ Zero)",
    "eval natEq (Succ Zero) (Succ Zero)",
    # Test replicate
    "eval replicate (Succ (Succ Zero)) Nat Zero",
    # Test head 
    "eval head Nat (Succ Zero) (Cons Nat (Succ Zero) (Succ (Succ (Succ Zero))) (Cons Nat Zero (Succ (Succ Zero)) (Nil Nat)))",
]

for test in test_stmts:
    test_parsed = parse(test + "\n")
    for stmt in test_parsed:
        try:
            _, output = execute(state, stmt)
            print(f"  {test:75s} => {output}")
        except Exception as e:
            print(f"  {test:75s} => ERROR: {e}")
            
            
# %% test the equality proofs
more_tests = [
    # p0PlusNisN should reduce for concrete n
    "eval p0PlusNisN (Succ (Succ Zero))",
    # pNPlus0isN for concrete n  
    "eval pNPlus0isN (Succ (Succ Zero))",
    # fromto 3 = [2, 1, 0] ->  (Cons 2, Cons 1, Cons 0)
    "eval fromto (Succ (Succ (Succ Zero)))",
    # map 
    # map Succ [0, 1] -> [1, 2]
    "eval map Nat Nat Succ (Succ (Succ Zero)) (Cons Nat (Succ Zero) Zero (Cons Nat Zero (Succ Zero) (Nil Nat)))",
    # tail 
    # tail [0, 1] -> [1]
    "eval tail Nat (Succ Zero) (Cons Nat (Succ Zero) Zero (Cons Nat Zero (Succ Zero) (Nil Nat)))",
    # at (vector indexing)
    # at [3, 2] 0 = 3 -> (FZero 1 = index 0)
    # at [3, 2] 1 = 2 -> (FSucc 1 (FZero 0) = index 1)
    "eval at Nat (Succ (Succ Zero)) (Cons Nat (Succ Zero) (Succ (Succ (Succ Zero))) (Cons Nat Zero (Succ (Succ Zero)) (Nil Nat))) (FZero (Succ Zero))",
    "eval at Nat (Succ (Succ Zero)) (Cons Nat (Succ Zero) (Succ (Succ (Succ Zero))) (Cons Nat Zero (Succ (Succ Zero)) (Nil Nat))) (FSucc (Succ Zero) (FZero Zero))",
]

for test in more_tests:
    test_parsed = parse(test + "\n")
    for stmt in test_parsed:
        try:
            _, output = execute(state, stmt)
            print(f"  {test[:80]:80s} => {output}")
        except Exception as e:
            print(f"  {test[:80]:80s} => ERROR: {e}")
            
# %% Additional kernel issues
# Check 1: Does the type checker correctly handle Lam checked against non-Pi?
try:
    with with_name_env(state.definitions):
        type_chk(0, state.context, Lam(Inf(Bound(0))), VNat())
    print("FAIL: Lam against VNat should fail")
except Exception as e:
    print(f"OK: Lam against VNat correctly fails: {type(e).__name__}")


# %% Test parsing
correct_nat_elim_fn = _make_nat_elim_fn()

# verify this type-checks correctly
try:
    ty = type_inf0([], correct_nat_elim_fn)
    print(f"Type-checks! Type: {pretty_infer(quote0(ty).term, [])}" if isinstance(quote0(ty), Inf) else f"Type: {quote0(ty)}")
except TypeCheckError as e:
    print(f"Type check error: {e.msg[:200]}")


# test partial application
# Apply it to 3 args (motive, base, step) for the `plus` definition:
# motive = \_ -> Nat -> Nat
# base = \n -> n
# step = \p rec n -> Succ (rec n)
motive_term = Lam(Inf(Pi(Inf(Nat()), Inf(Nat()))))  # \_ -> Nat -> Nat
base_term = Lam(Inf(Bound(0)))  # \n -> n  
step_term = Lam(Lam(Lam(Inf(Succ(Inf(App(Bound(1), Inf(Bound(0)))))))))  # \p rec n -> Succ (rec n)

correct_nat_elim_fn = _make_nat_elim_fn()

partial_natElim = App(
    App(
        App(correct_nat_elim_fn, motive_term),
        base_term
    ),
    step_term
)

try:
    ty = type_inf0([], partial_natElim)
    print(f"Partial application type-checks!")
    print(f"  Type: {pretty_value(ty)}")
except TypeCheckError as e:
    print(f"Type check error: {e.msg[:400]}")

# partial application works and gives the expected type Nat -> Nat -> Nat for `plus`.

# %% finElim
correct_fin_elim_type = make_fin_elim_type()

print("finElim type:")
print(pretty_infer(correct_fin_elim_type, []))

# Let me verify finElim type-checks:
try:
    ty = type_inf0([], _make_fin_elim_fn())
    print(f"finElim type-checks!")
    print(f"  Type: {pretty_value(ty)}")
except TypeCheckError as e:
    print(f"Type check error: {e.msg[:400]}")
    
# %% vecElim