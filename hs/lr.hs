data Entry a = 
    Empty 
  | Dummy a
  | Value a

data LR a = LR {
   l :: [Entry a],
   r :: [Entry a]
}

data Tree a =  
    Leaf a
  | Node (Tree a) (Tree a)

-- False = left, True = right
type Interval = (Tree, [Bool], [Bool])

init ∷ Interval a → LR a
init (t, ls, rs) = LR (repeat Empty) (map  
