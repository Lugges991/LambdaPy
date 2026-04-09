assume A : Type
assume x : A
eval (\a -> \y -> y) : forall (a : Type). a -> a
