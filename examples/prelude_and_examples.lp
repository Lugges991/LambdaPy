-- Prelude and example definitions resolving Star(0)/Star(1) TypeCheckErrors.
-- Compatible with the patched check.py (subtyping) + patched parser.py (motive
-- codomains bumped to Type 1 in the natElim / vecElim / finElim wrappers).

-- Naturals
let plus = (natElim (\ _ -> Nat -> Nat) (\ n -> n) (\ p rec n -> Succ (rec n))) :: forall (m :: Nat) (n :: Nat) . Nat

-- Unit / Void / Bool
let Unit  = ( Fin (Succ Zero) ) :: *
let U     = ( FZero Zero ) :: Fin (Succ Zero)
let Void  = ( Fin Zero ) :: *
let Bool  = ( Fin (Succ (Succ Zero)) ) :: *
let False = ( FZero (Succ Zero) ) :: Fin (Succ (Succ Zero))
let True  = ( FSucc (Succ Zero) (FZero Zero) ) :: Fin (Succ (Succ Zero))
let Not   = (\ a -> a -> Void) :: * -> *

-- Leibniz, with b lifted to Type 1 so * itself can be passed at b's position.
-- Without this, p1IsNot0's `leibniz Nat * ...` fails the Star(0)/Star(1) check.
let leibniz = ( \ a b f -> eqElim a (\ x y eq_x_y -> Eq b (f x) (f y)) (\ x -> Refl b (f x)) ) :: forall (a :: *) (b :: Type 1) (f :: a -> b) (x :: a) (y :: a) . Eq a x y -> Eq b (f x) (f y)

-- apply: transports along an equality of types.
let apply = (eqElim * (\ a b _ -> a -> b) (\ _ x -> x)) :: forall (a :: *) (b :: *) (p :: Eq * a b) . a -> b

-- nat1Elim / nat2Elim, with motive codomain at Type 1 so motives like
-- `\\ n -> Fin n -> *` (Nat -> Star(1)) are admissible.
let nat1Elim = ( \ m m0 m1 ms -> natElim m m0 (\ p rec -> natElim (\ n -> m (Succ n)) m1 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> (forall (n :: Nat) . m (Succ n) -> m (Succ (Succ n))) -> forall (n :: Nat) . m n
let nat2Elim = ( \ m m0 m1 m2 ms -> nat1Elim m m0 m1 (\ p rec -> natElim (\ n -> m (Succ (Succ n))) m2 ms p) ) :: forall (m :: Nat -> Type 1) . m Zero -> m (Succ Zero) -> m (Succ (Succ Zero)) -> (forall (n :: Nat) . m (Succ (Succ n)) -> m (Succ (Succ (Succ n)))) -> forall (n :: Nat) . m n

-- The four target definitions from the prompt.

let unitElim = ( \ m mu -> finElim ( nat1Elim (\ n -> Fin n -> *) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) ) ( natElim (\ n -> natElim (\ n -> Fin (Succ n) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FZero n)) mu (\ _ _ -> U) ) ( \ n f _ -> finElim (\ n f -> natElim (\ n -> Fin (Succ n) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc n f)) (\ _ -> U) (\ _ _ _ -> U) n f ) (Succ Zero) ) :: forall (m :: Unit -> *) . m U -> forall (u :: Unit) . m u

let boolElim = ( \ m mf mt -> finElim ( nat2Elim (\ n -> Fin n -> *) (\ _ -> Unit) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) ) ( nat1Elim ( \ n -> nat1Elim (\ n -> Fin (Succ n) -> *) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) n (FZero n)) U mf (\ _ _ -> U) ) ( \ n f _ -> finElim ( \ n f -> nat1Elim (\ n -> Fin (Succ n) -> *) (\ _ -> Unit) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc n f) ) ( natElim ( \ n -> natElim (\ n -> Fin (Succ (Succ n)) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc (Succ n) (FZero n)) ) mt (\ _ _ -> U) ) ( \ n f _ -> finElim (\ n f -> natElim (\ n -> Fin (Succ (Succ n)) -> *) (\ x -> m x) (\ _ _ _ -> Unit) n (FSucc (Succ n) (FSucc n f))) (\ _ -> U) (\ _ _ _ -> U) n f ) n f ) (Succ (Succ Zero)) ) :: forall (m :: Bool -> *) . m False -> m True -> forall (b :: Bool) . m b

let p1IsNot0 = (\ p -> apply Unit Void (leibniz Nat * (natElim (\ _ -> *) Void (\ _ _ -> Unit)) (Succ Zero) Zero p) U) :: Not (Eq Nat (Succ Zero) Zero)

let append = ( \ a -> vecElim a (\ m _ -> forall (n :: Nat) . Vec a n -> Vec a (plus m n)) (\ _ v -> v) (\ m v vs rec n w -> Cons a (plus m n) v (rec n w))) ::  forall (a :: *) (m :: Nat) (v :: Vec a m) (n :: Nat) (w :: Vec a n). Vec a (plus m n)
