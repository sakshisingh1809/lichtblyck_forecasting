"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import functools

from ...tools.nits import name2unit, ureg
from ..utils import changefreq_sum
from ..output_text import PfLineTextOutput
from ..output_plot import PfLinePlotOutput
from ..output_other import OtherOutput
from ..dunder_arithmatic import PfLineArithmatic
from ..hedge_functionality import PfLineHedge
from ..utils import changefreq_sum
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Union
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .pfline____archive import SinglePfLine

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where the volume q == 0 but the revenue r != 0, because this
#   occasionally arises for the sourced volume, e.g. after buying and selling the same
#   volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# It is unpractical to cater to both cases, as we'd need to constantly check which case
# we are dealing with, and it also raises questions without a 'natural' answer, e.g. when
# adding them, how is the result stored?
# The first case one is the most important one, and is therefore used. The second case
# must be handled by storing market prices seperately from volume data.


class PfLine(ABC):
    """Class to hold a related energy timeseries. This can be volume timeseries with q
    [MWh] and w [MW], a price timeseries with p [Eur/MWh] or both.

    Notes
    -----
    When kind == 'all', updating the PfLine means that we must choose how to recalculate
    the individual timeseries to keep the data consistent. In general, keeping the
    existing price is given priority. So, when multiplying the PfLine by 2, `w`, `q` and
    `r` are doubled, while `p` stays the same. And, when updating the volume (with
    `.set_w` or `.set_q`) the revenue is recalculated, and vice versa. Only when the
    price is updated, the existing volume is kept.'
    """

    # Abstract methods to be implemented by descendents.

    @property
    @abstractmethod
    def children(self) -> Dict[str, PfLine]:
        """Children of this instance, if any."""
        ...

    @property
    @abstractmethod
    def index(self) -> pd.DatetimeIndex:
        """Left timestamp of time period corresponding to each data row."""
        ...

    @property
    @abstractmethod
    def w(self) -> pd.Series:
        """Power timeseries [MW]."""
        ...

    @property
    @abstractmethod
    def q(self) -> pd.Series:
        """Energy timeseries [MWh]."""
        ...

    @property
    @abstractmethod
    def p(self) -> pd.Series:
        """Price timeseries [Eur/MWh]."""
        ...

    @property
    @abstractmethod
    def r(self) -> pd.Series:
        """Revenue timeseries [Eur]."""
        ...

    @property
    @abstractmethod
    def kind(self) -> str:
        """Kind of data that is stored in the instance. Possible values:
        - 'q': volume data only; properties .q [MWh] and .w [MW] are available.
        - 'p': price data only; property .p [Eur/MWh] is available.
        - 'all': price and volume data; properties .q [MWh], .w [MW], .p [Eur/MWh], .r [Eur] are available.
        """
        ...

    @abstractmethod
    def df(self, cols: Iterable[str] = None) -> pd.DataFrame:
        """DataFrame for this PfLine.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns to include in the dataframe.

        Returns
        -------
        pd.DataFrame
        """
        ...

    @abstractmethod
    def changefreq(self, freq: str = "MS") -> PfLine:
        """Resample the instance to a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.

        Returns
        -------
        Instance
            Resampled at wanted frequency.
        """
        ...

    # Methods implemented by the ABC itself.

    @property
    def summable(self) -> str:
        """Which attributes/colums of this PfLine can be added to those of other PfLines
        to get consistent/correct new PfLine."""
        return {"p": "p", "q": "q", "all": "qr"}[self.kind]

    @property
    def available(self) -> str:  # which time series have values
        """Attributes/columns that are available. One of {'wq', 'p', 'wqpr'}."""
        return {"p": "p", "q": "wq", "all": "wqpr"}[self.kind]

    def flatten(self) -> SinglePfLine:
        """Return flat instance, i.e., without children."""
        return SinglePfLine(self)

    @property
    def volume(self) -> SinglePfLine:
        """Return (flattened) volume-only PfLine."""
        return SinglePfLine(self.df("q"))

    @property
    def price(self) -> SinglePfLine:
        """Return (flattened) price-only PfLine."""
        return SinglePfLine(self.df("p"))

    def _set_col_val(
        self, col: str, val: Union[pd.Series, float, int, ureg.Quantity]
    ) -> SinglePfLine:
        """Set or update a timeseries and return the modified instance."""

        # Get pd.Series of other, in correct unit.
        if isinstance(val, float) or isinstance(val, int):
            val = pd.Series(val, self.index)
        elif isinstance(val, ureg.Quantity):
            val = pd.Series(val.magnitude, self.index).astype(f"pint[{val.u}]")

        if self.kind == "all" and col == "r":
            raise NotImplementedError(
                "Cannot set `r`; first select `.volume` or `.price` before applying `.set_r()`."
            )
        # Create pd.DataFrame.
        data = {col: val.astype(f"pint[{name2unit(col)}]")}
        if col in ["w", "q", "r"] and self.kind in ["p", "all"]:
            data["p"] = self["p"]
        elif col in ["p", "r"] and self.kind in ["q", "all"]:
            data["q"] = self["q"]
        df = pd.DataFrame(data)
        return SinglePfLine(df)

    def set_w(self, w: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update power timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("w", w)

    def set_q(self, q: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update energy timeseries [MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("q", q)

    def set_p(self, p: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update price timeseries [Eur/MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("p", p)

    def set_r(self, r: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update revenue timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("r", r)

    def set_volume(self, other: PfLine) -> SinglePfLine:
        """Set or update volume information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind != "q":
            raise ValueError(
                "Can only set volume from a PfLine instance with kind=='q'. Use .volume to obtain from given instance."
            )
        return self.set_q(other.q)

    def set_price(self, other: PfLine) -> SinglePfLine:
        """Set or update price information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind != "p":
            raise ValueError(
                "Can only set price from a PfLine instance with kind=='p'. Use .price to obtain from given instance."
            )
        return self.set_p(other.p)

    def __getitem__(self, name):
        return getattr(self, name)

    def __getattr__(self, name):
        if name in self.children:
            return self.children[name]
        raise AttributeError(f"No attribute {name}.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.df().equals(other.df())
