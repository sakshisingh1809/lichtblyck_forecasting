"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both; in all cases there is a single timeseries for each.
"""

from __future__ import annotations
from base import PFL

from lichtblyck.core.utils import changefreq_sum
from typing import Dict, Iterable, Union
import pandas as pd
import numpy as np


class SPFL(PFL):
    """Portfolio line without children. Has a single dataframe; .children is the empty
    dictionary.

    Parameters
    ----------
    data: Any
        Generally: object with one or more attributes or items `w`, `q`, `r`, `p`; all
        timeseries. Most commonly a DataFrame but may also be a dictionary or other
        PfLine object.
    """

    def __init__(self, data: Union[PFL, Dict, pd.DataFrame, pd.Series]):
        self._df = pd.DataFrame(
            np.random.rand(4, 4), index=[2022, 2023, 2024, 2025], columns=list("qwrp")
        )

    # Implementation of ABC methods.

    @property
    def children(self) -> Dict:
        return {}

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._df.index

    @property
    def w(self) -> pd.Series:
        if "q" not in self.available:
            return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")
        return pd.Series(self.q / self.index.duration, name="w").pint.to("MW")

    @property
    def q(self) -> pd.Series:
        if "q" not in self.available:
            return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")
        return self._df["q"]

    @property
    def p(self) -> pd.Series:
        if "p" not in self.available:
            return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")
        if self.kind == "p":
            return self._df["p"]
        return pd.Series(self.r / self.q, name="p").pint.to("Eur/MWh")

    @property
    def r(self) -> pd.Series:
        if "r" not in self.available:
            return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")
        return self._df["r"]

    @property
    def kind(self) -> str:
        if "q" in self._df:
            return "all" if ("r" in self._df or "p" in self._df) else "q"
        if "p" in self._df:
            return "p"
        raise ValueError("Unexpected value for ._df.")

    def df(self, cols: Iterable[str] = None) -> pd.DataFrame:
        if cols is None:
            cols = self.available
        return pd.DataFrame({col: self[col] for col in cols})

    def changefreq(self, freq: str = "MS") -> SPFL:
        return SPFL(changefreq_sum(self.df(self.summable), freq))

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)


class _LocIndexer:
    """Helper class to obtain subset of the PfLine index."""

    def __init__(self, spfl):
        self.spfl = spfl

    def __getitem__(self, arg) -> SPFL:
        new_df = self.spfl.df().loc[arg]
        return SPFL(new_df)
