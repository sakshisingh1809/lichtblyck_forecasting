"""
Custom classes that are thin wrappers around the pandas objects.
"""

from __future__ import annotations
from . import attributes
import pandas as pd
import numpy as np
import functools


FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def force_Pf(function):
    """Decorator to ensure a PfFrame (instead of a DataFrame) or a PfSeries (instead of a Series) is returned."""

    def wrapper(*args, **kwargs):
        val = function(*args, **kwargs)
        if type(val) is pd.DataFrame:
            val = PfFrame(val)
        elif type(val) is pd.Series:
            val = PfSeries(val)
        return val

    return wrapper


# Currently not working. Goal: wrap each method of pd.DataFrame and pd.Series
# class PfMeta(type):
#     def __new__(cls, name, bases, dct):
#         klass = super().__new__(cls, name, bases, dct)
#         for base in bases:
#             print (f'base: {base}')
#             for field_name, field in base.__dict__.items():
#                 print (f'field_name: {field_name}, field: {field}')
#                 if callable(field):
#                     print(f'yes, callable {field_name}')
#                     setattr(klass, field_name, force_Pf(field))
#         return klass


class PfSeries(pd.Series):
    """
    PortfolioSeries; pandas series with additional functionality for getting
    duration [h] timeseries.
    """

    # @functools.wraps(pd.Series.__init__)  # keep original signature
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._entrycheck()

    # def _entrycheck(self):
    #     """Check passed values to see if all is OK."""
    #     # Index must have timezone and frequency information.
    #     if not self.index.tz:
    #         raise AttributeError("No timezone information is passed.")
    #     if not self.index.freq in FREQUENCIES:
    #         raise AttributeError(
    #             f".freq attribute must be one of {', '.join(FREQUENCIES)}."
    #         )
    #     # Set index name.
    #     self.index = self.index.rename("ts_left")

    duration = property(attributes._duration)
    ts_right = property(attributes._ts_right)


class PfFrame(pd.DataFrame):
    """
    PortfolioFrame; pandas dataframe with additional functionality for getting
    power [MW], price [Eur/MWh], quantity [MWh] and revenue [Eur] timeseries,
    as well as and right-bound timestamp and duration [h].

    Attributes
    ----------
    w, q, p, r : PfSeries
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries.
    ts_right, duration : pandas.Series
        Right timestamp and duration [h] of row.

    Notes
    -----
    Under the hood, only the `q` and `r` values are kept; the others are calculated
    whenever needed.
    """

    # @functools.wraps(pd.DataFrame.__init__)  # keep original signature
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._entrycheck()

    # def _entrycheck(self):
    #     """Check passed values to see if all is OK."""
    #     # Index must have timezone and frequency information.
    #     if not self.index.tz:
    #         raise AttributeError("No timezone information is passed.")
    #     if self.index.freq not in FREQUENCIES:
    #         raise AttributeError(
    #             f".freq attribute must be one of {', '.join(FREQUENCIES)}."
    #         )

    #     # Keep only q and r timeseries, and checking if redundant information is correct.
    #     if "q" not in self.columns:
    #         if "w" in self.columns:
    #             self["q"] = self["w"] * self.duration
    #         elif "r" in self.columns and "p" in self.columns:
    #             self["q"] = self["r"] / self["p"]
    #         else:
    #             raise AttributeError(
    #                 "`q` not passed as column and can't be calculated."
    #             )

    #     if "r" not in self.columns:
    #         if "p" in self.columns:
    #             self["r"] = self["q"] * self["p"]
    #         else:  # only MW data is provided, no financial value.
    #             self["r"] = np.nan

    #     if "w" in self.columns:
    #         if not np.allclose(self["q"], self["w"] * self.duration):  # redundant check
    #             raise ValueError("Passed values for `q` and `w` not compatible.")
    #         del self["w"]

    #     if "p" in self.columns:
    #         if not np.allclose(self["r"], self["p"] * self["q"], equal_nan=True):  # redundant check
    #             raise ValueError("Passed values for `q`, `p` and `r` not compatible.")
    #         del self["p"]

    #     # Set index name.
    #     self.index = self.index.rename("ts_left")

    # # Time series.
    # q = property(lambda self: PfSeries(self["q"]))
    # r = property(lambda self: PfSeries(self["r"]))
    # w = property(lambda self: PfSeries((self.q / self.duration).rename("w")))
    # p = property(lambda self: PfSeries((self.r / self.q).rename("p")))

    duration = property(attributes._duration)
    ts_right = property(attributes._ts_right)
