from base import Animal


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


if __name__ == "__main__":
    mrchompers = Dog("Mr. Chompers")
    mrchompers.action()  # Mr. Chompers says 'woof'
    mrchompers.turn_into_cat().action()  # Mr. Chompers says 'miauw'


def turn_into_cat(animal: Animal):
    return Cat(animal.name)


Animal.turn_into_cat = turn_into_cat
