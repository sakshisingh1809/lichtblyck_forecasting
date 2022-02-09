from zoo import Dog, Cat, Animal

a1 = Dog(34)
try:
    a2 = Dog(0.9)  # ValueError
except ValueError:
    pass
else:
    raise RuntimeError("Should have raised Exception!")

a3 = Cat(0.8)
try:
    a4 = Cat(25)  # ValueError
except ValueError:
    pass
else:
    raise RuntimeError("Should have raised Exception!")

a5 = Animal(80)  # can only be dog; should return dog.
assert type(a5) is Dog
a6 = Animal(0.7)  # can only be cat; should return cat.
assert type(a6) is Cat
a7 = Animal(10)  # can be both; should return dog.
assert type(a7) is Dog
try:
    a8 = Animal(400)
except NotImplementedError:
    pass
else:
    raise RuntimeError("Should have raised Exception!")
