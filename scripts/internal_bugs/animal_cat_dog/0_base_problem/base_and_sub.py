from abc import ABC, abstractmethod


class Animal(ABC):
    def __new__(cls, weight: float):
        if cls is Animal:
            # Try to return subclass instance instead.
            for subcls in [Dog, Cat]:
                try:
                    return subcls(weight)
                except ValueError:
                    pass
            raise NotImplementedError("No appropriate subclass found.")
        return super().__new__(cls)

    @property
    @abstractmethod
    def weight(self) -> float:
        """weight of the animal in kg."""
        ...


class Dog(Animal):
    def __init__(self, weight: float = 5):
        if not (1 < weight < 90):
            raise ValueError("No dog has this weight")
        self._weight = weight

    weight: float = property(lambda self: self._weight)


class Cat(Animal):
    def __init__(self, weight: float = 5):
        if not (0.5 < weight < 15):
            raise ValueError("No cat has this weight")
        self._weight = weight

    weight: float = property(lambda self: self._weight)


if __name__ == "__main__":

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

# %%
