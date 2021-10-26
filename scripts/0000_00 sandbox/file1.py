import math

class Class:

    @classmethod
    def from_prodsum(cls, p, s):
        x = math.sqrt(s**2 / 4 - p) + s/2
        y = s - x
        return cls(x, y)

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def sum(self):
        return self.x + self.y

    @property
    def prod(self):
        return self.x * self.y


def class_from_2x2y(twicex, twicey):
    return Class(twicex/2, twicey/2)

setattr(Class, "from_2x2y", class_from_2x2y)