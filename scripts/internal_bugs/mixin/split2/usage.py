from potions import GreenPotion, BrownPotion

b1 = GreenPotion(5)
b2 = BrownPotion(111)

b3 = b1 + b2
assert b3.volume == 116
assert type(b3) is BrownPotion

b4 = b1 * 3
assert b4.volume == 15
assert type(b4) is GreenPotion

b5 = b2 * 3
assert b5.volume == 333
assert type(b5) is BrownPotion

b6 = -b1
assert b6.volume == 5
assert type(b6) is BrownPotion
