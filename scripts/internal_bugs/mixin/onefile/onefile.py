from abc import ABC, abstractmethod


class PotionArithmatic:
    def __add__(self, other):
        # Adding potions always returns a brown potion.
        if isinstance(other, Potion):
            return BrownPotion(self.volume + other.volume)
        return BrownPotion(self.volume + other)

    def __mul__(self, other):
        # Multiplying a potion with a number scales it.
        if isinstance(other, Potion):
            raise TypeError("Cannot multiply Potions")
        return self.__class__(self.volume * other)

    def __neg__(self):
        # Negating a potion changes its color but not its volume.
        if isinstance(self, GreenPotion):
            return BrownPotion(self.volume)
        else:  # isinstance(self, BrownPotion):
            return GreenPotion(self.volume)


class Potion(ABC, PotionArithmatic):
    def __init__(self, volume: float):
        self.volume = volume

    __repr__ = lambda self: f"{self.__class__.__name__} with volume of {self.volume} l."

    @property
    @abstractmethod
    def color(self) -> str:
        ...


class GreenPotion(Potion):
    color = "green"


class BrownPotion(Potion):
    color = "brown"


if __name__ == "__main__":

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
