from abc import ABC, abstractmethod


class Potion(ABC):
    def __init__(self, volume: float):
        self.volume = volume

    __repr__ = lambda self: f"{self.__class__.__name__} with volume of {self.volume} l."

    @property
    @abstractmethod
    def color(self) -> str:
        ...


from . import arithmatic

arithmatic.apply()