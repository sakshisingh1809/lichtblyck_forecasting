"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both.
"""

import functools
from typing import Dict, Iterable, Union
from __future__ import annotations
from .base import PfLine
from ..dunder_arithmatic import PfLineArithmatic
from ..hedge_functionality import PfLineHedge
from ..output_other import OtherOutput
from ..output_plot import PfLinePlotOutput
from ..output_text import PfLineTextOutput
from .prep import make_df
from ..utils import changefreq_sum
from ...tools.nits import name2unit, ureg
import pandas as pd
import numpy as np


class PfLine(
    PfLine,
    PfLineTextOutput,
    PfLinePlotOutput,
    OtherOutput,
    PfLineArithmatic,
    PfLineHedge,
):
    """Class to hold a related energy timeseries. This can be volume timeseries with q
    [MWh] and w [MW], a price timeseries with p [Eur/MWh] or both.

    Parameters
    ----------
    data : object
        Generally: object with one or more attributes or items `w`, `q`, `r`, `p`; all
        timeseries. Most commonly a DataFrame but may also be a dictionary or other
        PfLine object.

    Notes
    -----
    When kind == 'all', updating the PfLine means that we must choose how to recalculate
    the individual timeseries to keep the data consistent. In general, keeping the
    existing price is given priority. So, when multiplying the PfLine by 2, `w`, `q` and
    `r` are doubled, while `p` stays the same. And, when updating the volume (with
    `.set_w` or `.set_q`) the revenue is recalculated, and vice versa. Only when the
    price is updated, the existing volume is kept.'
    """

    pass
    # def __init__(self, data):
    #     # The only internal data of this class.
    #     self._df = make_df(data)

    # # Methods/Properties that return most important information.

    # @property
    # def index(self) -> pd.DatetimeIndex:
    #     """Left timestamp of time period corresponding to each data row."""
    #     return self._df.index

    # @property
    # def w(self) -> pd.Series:
    #     """Power timeseries [MW]."""
    #     if "q" in self._df:
    #         return pd.Series(self._df["q"] / self._df.index.duration, name="w").pint.to(
    #             "MW"
    #         )
    #     return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")

    # @property
    # def q(self) -> pd.Series:
    #     """Energy timeseries [MWh]."""
    #     if "q" in self._df:
    #         return self._df["q"]
    #     return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")

    # @property
    # def p(self) -> pd.Series:
    #     """Price timeseries [Eur/MWh]."""
    #     if "p" in self._df:
    #         return self._df["p"]
    #     if "q" in self._df and "r" in self._df:
    #         return pd.Series(self._df["r"] / self._df["q"], name="p").pint.to("Eur/MWh")
    #     return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")

    # @property
    # def r(self) -> pd.Series:
    #     """Revenue timeseries [Eur]."""
    #     if "r" in self._df:
    #         return self._df["r"]
    #     if "q" in self._df and "p" in self._df:
    #         return pd.Series(self._df["q"] * self._df["p"], name="r").pint.to("Eur")
    #     return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")

    # def df(self, cols: str = None) -> pd.DataFrame:
    #     """DataFrame for this PfLine.

    #     Parameters
    #     ----------
    #     cols : str, optional (default: all that are available)
    #         The columns to include in the dataframe.

    #     Returns
    #     -------
    #     pd.DataFrame
    #     """
    #     if cols is None:
    #         cols = self.available
    #     return pd.DataFrame({col: self[col] for col in cols})

    # @property
    # def kind(self) -> str:
    #     """Kind of data that is stored in the instance. Possible values:
    #     'q': volume data only; properties .q [MWh] and .w [MW] are available.
    #     'p': price data only; property .p [Eur/MWh] is available.
    #     'all': price and volume data; properties .q [MWh], .w [MW], .p [Eur/MWh], .r [Eur] are available.
    #     """
    #     if "q" in self._df:
    #         return "all" if ("r" in self._df or "p" in self._df) else "q"
    #     if "p" in self._df:
    #         return "p"
    #     raise ValueError("Unexpected value for ._df.")

    # # Methods/Properties that return new class instance.

    # def changefreq(self, freq: str = "MS") -> PfLine:
    #     """Resample the PfLine to a new frequency.

    #     Parameters
    #     ----------
    #     freq : str, optional
    #         The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS'
    #         (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.

    #     Returns
    #     -------
    #     PfLine
    #         Resampled at wanted frequency.
    #     """
    #     return PfLine(changefreq_sum(self.df(self.summable), freq))

    # # Dunder methods.

    # def __getitem__(self, name):
    #     return getattr(self, name)

    # def __eq__(self, other):
    #     if not isinstance(other, self.__class__):
    #         return False
    #     return self._df.equals(other._df)
