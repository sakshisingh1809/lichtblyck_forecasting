from . import base, brown, green


def add_potion_instances(potion1, potion2):
    return brown.BrownPotion(potion1.volume + potion2.volume)


def ext_add(self, other):
    # Adding potions always returns a brown potion.
    if isinstance(other, base.Potion):
        return add_potion_instances(self, other)
    return brown.BrownPotion(self.volume + other)


def ext_mul(self, other):
    # Multiplying a potion with a number scales it.
    if isinstance(other, base.Potion):
        raise TypeError("Cannot multiply Potions")
    return self.__class__(self.volume * other)


def ext_neg(self):
    # Negating a potion changes its color but not its volume.
    if isinstance(self, green.GreenPotion):
        return brown.BrownPotion(self.volume)
    else:  # isinstance(self, BrownPotion):
        return green.GreenPotion(self.volume)


base.Potion.__add__ = ext_add
base.Potion.__mul__ = ext_mul
base.Potion.__neg__ = ext_neg
