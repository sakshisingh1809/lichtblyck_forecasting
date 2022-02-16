from abc import ABC, abstractmethod


class Animal(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is Animal:
            # Try to return subclass instance instead.
            for subcls in [dog.Dog, cat.Cat]:
                try:
                    return subcls(*args, **kwargs)
                except ValueError:
                    pass
            raise NotImplementedError("No appropriate subclass found.")
        return super().__new__(cls)

    @property
    @abstractmethod
    def weight(self) -> float:
        """weight of the animal in kg."""
        ...


import dog  # from dog import Dog gives circular import error
import cat  # from cat import Cat gives circular import error
