from abc import ABC, abstractmethod


class Animal(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is Animal:
            # Try to return subclass instance instead.
            subclasses = {sc.__name__: sc for sc in Animal.__subclasses__()}
            for subcls in [subclasses["Dog"], subclasses["Cat"]]:
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
