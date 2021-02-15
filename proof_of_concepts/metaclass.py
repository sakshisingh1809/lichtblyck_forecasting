# -*- coding: utf-8 -*-

import math


class Base:
    def __init__(self, val):
        self.val = val

    def newinst_addseven(self):
        return Base(self.val + 7)

    def newinst_timestwo(self):
        return Base(self.val * 2)

    # ...


class Child1(Base):
    @property
    def sqrt(self):
        return math.sqrt(self.val)


child1a = Child1(16)
child1a.val  # OK: 16
child1a.sqrt  # OK: 4.0

child1b = child1a.newinst_addseven()
child1b.val  # OK: 23
child1b.sqrt  # has no attribute 'sqrt'
type(child1b)  # __main__.Base


# Solution one:
Child1.newinst_addseven = lambda c: Child1(Base.newinst_addseven(c).val)
Child1.newinst_timestwo = lambda c: Child1(Base.newinst_timestwo(c).val)
# ...

child1c = child1a.newinst_addseven()
child1c.val  # OK: 23
child1c.sqrt  # OK: 4.796...
type(child1c)  # __main__.Child1

#%%

import math


class Base:
    def __init__(self, val):
        self.val = val

    def newinst_addseven(self):
        """Return new instance, with seven added to its value."""
        return Base(self.val + 7)

    def newinst_timestwo(self):
        """Return new instance, with its value multiplied by two."""
        return Base(self.val * 2)

    # ...


# Solotion two:


def force_child(fun):
    def wrapper(*args, **kwargs):
        result = fun(*args, **kwargs)
        if type(result) == Base:
            return Child2(result.val)
        return result

    return wrapper


class ChildMeta(type):
    def __new__(cls, name, bases, dct):
        print(f"** [__new__] name: {name}, dct: {dct}")
        child = super().__new__(cls, name, bases, dct)
        for base in bases:
            print(f"** base: {base}")
            for field_name, field in base.__dict__.items():
                print(f"** field_name: {field_name}, field: {field}")
                if callable(field):
                    print(f"** yes, callable {field_name}")
                    setattr(child, field_name, force_child(field))
        return child


class Child2(Base, metaclass=ChildMeta):
    @property
    def sqrt(self):
        return math.sqrt(self.val)


#%%
for meth in dir(Base):
    if (meth not in dir(Child2)) and not (
        meth.startswith("__") and meth.endswith("__")
    ):
        Child2.setattr(
            meth, lambda *args, **kwargs: getattr(Base, meth)(*args, **kwargs)
        )


#%%

child2a = Child2(16)
child2a.val  # OK: 16
child2a.sqrt  # OK: 4.0

Child2.newinst_addseven = force_child(Base.newinst_addseven)
child2b = child2a.newinst_addseven()
child2b.val  # OK: 23
child2b.sqrt  # has no attribute 'sqrt'
type(child2b)  # __main__.Base

#%%

import pandas as pd
import numpy as np


def pf_i(val):
    """Turn Series or DataFrame instance into PfSeries or PfFrame instance."""
    if type(val) is pd.DataFrame:
        val = PfFrame(val)
    return val


def pf_f(func):
    """Turn from Series-or-DataFrame-instance-returning function into
    PfSeries-or-PfFrame-instance-returning function."""

    def wrapper(*args, **kwargs):
        val = func(*args, **kwargs)
        return pf_i(val)

    return wrapper


# Wrap each method of pd.DataFrame
class PfMeta(type):
    def __new__(cls, name, bases, dct):
        klass = super().__new__(cls, name, bases, dct)
        for base in bases:
            for field_name, field in base.__dict__.items():
                if callable(field) and field_name != "__init__":
                    setattr(klass, field_name, pf_f(field))
        return klass


class PfFrame(pd.DataFrame, metaclass=PfMeta):
    """
    PortfolioFrame; thin wrapper around pandas DataFrame with additional information
    and functionality.
    """

    pass


df = pd.DataFrame(np.arange(16).reshape(8, 2), columns=list("ab"))
pfframe = PfFrame(np.arange(16).reshape(8, 2), columns=list("ab"))

breakpoint()
print(pfframe.a)

# elif type(val) is pd.Series:
#     val = PfSeries(val)
# class PfSeries(pd.Series, metaclass=PfMeta):
#     """
#     PortfolioSeries; thin wrapper around pandas Series with additional information
#     and functionality.
#     """
#     pass
