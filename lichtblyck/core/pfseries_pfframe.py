"""
Custom classes that are thin wrappers around the pandas objects.
"""

from __future__ import annotations
from . import attributes
import pandas as pd
import numpy as np
import functools


FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def pf(val):
    if type(val) is pd.DataFrame:
        val = PfFrame(val)
    elif type(val) is pd.Series:
        val = PfSeries(val)
    return val


def force_Pf(function):
    """Decorator to ensure a PfFrame (instead of a DataFrame) or a PfSeries (instead of a Series) is returned."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        val = function(*args, **kwargs)
        return pf(val)

    return wrapper


# # Currently not working. Goal: wrap each method of pd.DataFrame and pd.Series
class PfMeta(type):
    def __new__(cls, name, bases, dct):
        klass = super().__new__(cls, name, bases, dct)
        for base in bases:
            print(f"base: {base}")
            for field_name, field in base.__dict__.items():
                print(f"field_name: {field_name}, field: {field}")
                if callable(field):
                    print(f"yes, callable {field_name}")
                    setattr(klass, field_name, force_Pf(field))
                else:
                    setattr(klass, field_name, pf(field))
        return klass


class PfSeries(pd.Series):  # , metaclass=PfMeta):
    """
    PortfolioSeries; thin wrapper around pandas Series with additional information
    and functionality.
    """

    duration = property(force_Pf(attributes._duration))
    ts_right = property(force_Pf(attributes._ts_right))
    wavg = (
        attributes.wavg
    )  # no need to wrap; returns float or Series without time-index


class PfFrame(pd.DataFrame):  # , metaclass=PfMeta):
    """
    PortfolioFrame; thin wrapper around pandas DataFrame with additional information
    and functionality.
    """

    duration = property(force_Pf(attributes._duration))
    ts_right = property(force_Pf(attributes._ts_right))
    wavg = force_Pf(attributes.wavg)
