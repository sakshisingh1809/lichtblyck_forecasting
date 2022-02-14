from .animal import Animal


class Dog(Animal):
    def __init__(self, weight: float = 5):
        if not (1 < weight < 90):
            raise ValueError("No dog has this weight")
        self._weight = weight

    weight: float = property(lambda self: self._weight)
