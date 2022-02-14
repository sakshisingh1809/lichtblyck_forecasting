from abc import abstractmethod, ABC
from typing import Iterable
from __future__ import annotations


class Entity(ABC):
    @abstractmethod
    def describe(self) -> str:
        ...

    @property
    @abstractmethod
    def firstname(self) -> str:
        ...

    def firstperson(self) -> Person:
        return Person(self.firstname)


class Person(Entity):
    def __init__(self, name: str):
        self.name = name

    def describe(self):
        return f"Hi I am {self.name}"

    firstname = property(lambda self: self.name)


class Group(Entity):
    def __init__(self, names: Iterable[str]):
        self.names = names

    def describe(self):
        return f'Hello we are {" and ".join(self.names)}.'

    firstname = property(lambda self: self.names[0])


p1 = Person("Alice")
g1 = Group(["Bob", "Chris"])
