from abc import ABC, abstractmethod
import subs


class Animal(ABC):

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
