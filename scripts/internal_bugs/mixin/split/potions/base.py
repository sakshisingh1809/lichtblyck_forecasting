from abc import ABC, abstractmethod

from .arithmatic import PotionArithmatic


class Potion(ABC, PotionArithmatic):
    def __init__(self, volume: float):
        self.volume = volume

    __repr__ = lambda self: f"{self.__class__.__name__} with volume of {self.volume} l."

    @property
    @abstractmethod
    def color(self) -> str:
        ...
