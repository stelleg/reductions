{-# LANGUAGE UnicodeSyntax #-}
--{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

import Prelude hiding (Semigroup(..), Monoid(..), id)

data Magma m = Magma {getMagma ∷ m → m → m}
data Unital (a ∷ *) (m ∷ Magma a) = Unital a
data Associative (a ∷ *) (m ∷ Magma a) = Associative -- ∀ a b c, (a · b) · c = a · (b · c)
data Divisible (a ∷ *) (m ∷ Magma a) = Divisible -- ∀ a b, ∃ x y, a · x = b ∧ y · a = b

type UnitalMagma (a ∷ *) (m ∷ Magma a) = Unital a
type Semigroup (a ∷ *) (m ∷ Magma a) = Associative a m
type Quasigroup (a ∷ *) (m ∷ Magma a) = Divisible a m

type Monoid (a ∷ *) (m ∷ Magma a) = (Unital a m, Associative a m)
type Loop (a ∷ *) (m ∷ Magma a) = (Unital a m, Associative a m)
type InverseSemigroup (a ∷ *) (m ∷ Magma a) = (Divisible a m, Associative a m)

type Group (a ∷ *) (m ∷ Magma a) = (Unital a m, Associative a m, Divisible a m)

sumMagma ∷ Num a ⇒ Magma a
sumMagma = Magma (+)

sumUnit ∷ Num a ⇒ Unital a sumMagma
sumUnit = Unital 0

productMagma ∷ Num a ⇒ Magma a
productMagma = Magma (*)

productUnit ∷ Num a ⇒ Unital a productMagma
productUnit = Unital 1

reduce ∷ Foldable t ⇒ ∀ (a ∷ *) (m ∷ Magma a). Unital a m → Unital a m → t a → a
reduce (Unital id) (Unital id2) xs = id --undefined --foldr (∘) id xs

-- Problem: we don't actually have dependent types, it just looks like
-- it. This should be impossible, as in the type definition of reduce,
-- we should have constrained the reduction to be over the same
-- magmas, but they are not. We also don't have a way to use m as a
-- term.
wrong = reduce sumUnit productUnit [1,2]
