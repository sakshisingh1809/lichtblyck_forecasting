from abc import abstractmethod, ABC
from typing import Iterable


class Entity(ABC):
    @abstractmethod
    def describe(self) -> str:
        ...


class Person(Entity):
    def __init__(self, name: str):
        self.name = name

    def describe(self):
        return f"Hi I am {self.name}"


class Group(Entity):
    def __init__(self, names: Iterable[str]):
        self.names = names

    def describe(self):
        return f'Hello we are {" and ".join(self.names)}.'


p1 = Person("Alice")
g1 = Group(["Bob", "Chris"])


def create_entity(name_or_names):
    if isinstance(name_or_names, str):
        return Person(name_or_names)
    else:
        return Group(name_or_names)


p2 = create_entity("Alice")
g2 = create_entity(["Bob", "Chris"])


p3 = Entity("Alice")
g3 = Entity(["Bob", "Chris"])

p3.describe()
