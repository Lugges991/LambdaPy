eval Succ (Succ Zero) : Nat
eval NatElim (\_ -> Nat) (Succ Zero) (\k -> \r -> Succ r) (Succ (Succ Zero)) : Nat
