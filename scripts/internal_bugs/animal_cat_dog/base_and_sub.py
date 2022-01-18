from abc import ABC, abstractmethod


class Animal(ABC):

    _registry = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls  # Add class to registry.

    # Methods to be implemented by subclass.

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the animal."""
        ...

    @abstractmethod
    def action(self):
        """Do the typical animal action."""
        ...

    # Methods directly implemented by base class.

    def turn_into_cat(self):
        return self._registry["Cat"](self.name)


class Cat(Animal):
    def __init__(self, name):
        self._name = name

    name = property(lambda self: self._name)
    action = lambda self: print(f"{self.name} says 'miauw'")


class Dog(Animal):
    def __init__(self, name):
        self._name = name

    name = property(lambda self: self._name)
    action = lambda self: print(f"{self.name} says 'woof'")


mrchompers = Dog("Mr. Chompers")
mrchompers.action()  # Mr. Chompers says 'woof'
mrchompers.turn_into_cat().action()  # Mr. Chompers says 'miauw'
