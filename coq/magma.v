Require Import Lists.List.
Import ListNotations.
Require Import Unicode.Utf8.
Open Scope list_scope.
Open Scope type_scope.

Definition magma a := a → a → a.
Definition unit {a} (m : magma a) := { x:a | ∀ y, m x y = y ∧ m y x = y }.
Definition associative {a} (m : magma a) :=  
  ∀ x y z, m x (m y z) = m (m x y) z.
Definition divisible {a} (m : magma a) :=
  ∀ a b, ∃ x y, a * x = b ∧ y * a = b.

Definition quasigroup {a} := { m : magma a & divisible m }. 
Definition semigroup {a} := { m : magma a & associative m }. 
Definition unitalmagma {a} := { m : magma a & unit m }.

Definition loop {a} := { m : magma a & divisible m * unit m }. 
Definition inverseSemigroup {a} := { m : magma a & divisible m * associative m }. 
Definition monoid {a} := { m : magma a & associative m * unit m }. 

Definition group {a} := { m : magma a & 
  divisible m *  unit m * associative m }.

Definition join {a} (x y:list a) (xys: list a * list a) := let (xs, ys) := xys in (x++xs, y++ys).
Fixpoint split {a} (l : list a) : list a * list a := match l with
  | x :: y :: ls => join [x] [y] (split ls)
  | x :: ls => join [x] [] (split ls)
  | [] => ([], [])
  end. 

Fixpoint reduce {a} (n : nat) (l : list a) (m : unitalmagma) :=
  let f := projT1 m in match n with
  | 0 => fold_left f l (proj1_sig (projT2 m))
  | S n => let (l,r) := split l in f (reduce n l m) (reduce n r m)
  end.


