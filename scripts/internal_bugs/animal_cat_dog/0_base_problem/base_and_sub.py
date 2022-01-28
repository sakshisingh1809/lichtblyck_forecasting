from abc import ABC, abstractmethod
from typing import Callable


class Animal(ABC):
    def __new__(cls, name, weight):
        print(f"{cls.__name__}, __new__")
        try:
            return object.__new__(Cat(name, weight))
        except ValueError:
            pass
        try:
            return Dog(name, weight)
        except ValueError:
            pass

    # Methods to be implemented by subclasses.

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the animal."""
        ...

    @property
    @abstractmethod
    def weight(self) -> float:
        """weight of the animal in kg."""
        ...

    @abstractmethod
    def speak(self):
        """Say the typical animal words."""
        ...

    # Methods directly implemented by base class.

    def turn_into_cat(self):
        return Cat(self.name, self.weight)


class Cat(Animal):
    def __init__(self, name: str, weight: float = 5):
        if not (0.5 < weight < 15):
            raise ValueError("No cat has this weight")
        self._name, self._weight = name, weight

    name: str = property(lambda self: self._name)
    weight: float = property(lambda self: self._weight)
    speak: Callable = lambda self: print(f"{self.name} says 'miauw'")


class Dog(Animal):
    def __init__(self, name: str, weight: float = 5):
        print("Dog.__init__")
        if not (1 < weight < 90):
            raise ValueError("No dog has this weight")
        self._name, self._weight = name, weight

    name: str = property(lambda self: self._name)
    weight: float = property(lambda self: self._weight)
    speak = lambda self: print(f"{self.name} says 'woof'")


mrchompers = Dog("Mr. Chompers", 3)
mrchompers.speak()  # Mr. Chompers says 'woof'
mrchompers.turn_into_cat().speak()  # Mr. Chompers says 'miauw'
