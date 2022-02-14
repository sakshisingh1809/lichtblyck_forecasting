from .animal import Animal


class Cat(Animal):
    def __init__(self, weight: float = 5):
        if not (0.5 < weight < 15):
            raise ValueError("No cat has this weight")
        self._weight = weight

    weight: float = property(lambda self: self._weight)
