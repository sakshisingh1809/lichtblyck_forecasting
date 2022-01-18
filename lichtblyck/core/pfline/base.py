"""
Abstract Base Classes for PfLine and PfState.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping
import pandas as pd

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where, for some timestamp, the volume q == 0 but the revenue
#   r != 0, because this occasionally arises for the sourced volume, e.g. after buying
#   and selling the same volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# Both cases can be catered to. The first as a 'SinglePfLine', where the timeseries for
# q and r are used in the instance creation. The price is not defined at the timestamp in
# the example, but can be calculated for other timestamps, and downsampling is also still
# possble.
# The second is a bit more complex. It is possible as a 'MultiPfLine'. This has then 2
# 'SinglePfLine' instances as its children: one made from each of the timeseries for q
# and p.


class NDFrameLike(ABC):
    """Class that specifies which attributes from pandas Series and DataFrames must be
    implemented by descendents of this class."""

    # Abstract methods to be implemented by descendents.

    @property
    @abstractmethod
    def index(self) -> pd.DatetimeIndex:
        """Left timestamp of time period corresponding to each data row."""
        ...

    @abstractmethod
    def df(self, cols: Iterable[str] = None, flat: bool = True) -> pd.DataFrame:
        """DataFrame for this PfLine.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns to include in the dataframe.
        flat : bool, optional (default: True)
            - If True, include only aggregated timeseries (4 or less; 1 per dimension).
            - If False, include all timerseries and their (intermediate and final)
              aggregations.

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

    @property
    @abstractmethod
    def loc(self):
        """Create a new instance with a subset of the rows (selection by row label(s) or
        a boolean array.)"""
        ...

    @abstractmethod
    def __eq__(self, other) -> bool:
        ...


class PfLine(NDFrameLike, Mapping):
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
    def w(self) -> pd.Series:
        """(Flattened) power timeseries in [MW]."""
        ...

    @property
    @abstractmethod
    def q(self) -> pd.Series:
        """(Flattened) energy timeseries in [MWh]."""
        ...

    @property
    @abstractmethod
    def p(self) -> pd.Series:
        """(Flattened) price timeseries in [Eur/MWh]."""
        ...

    @property
    @abstractmethod
    def r(self) -> pd.Series:
        """(Flattened) revenue timeseries in [Eur]."""
        ...

    @property
    @abstractmethod
    def kind(self) -> str:
        """Kind of data that is stored in the instance. Possible values:
        - 'q': volume data only; properties .q [MWh] and .w [MW] are available.
        - 'p': price data only; property .p [Eur/MWh] is available.
        - 'all': price and volume data; properties .q [MWh], .w [MW], .p [Eur/MWh], .r
          [Eur] are available.
        """
        ...

    # Implemented directly at ABC.

    # Iterating over children.

    def items(self):
        return self.children.items()

    def __iter__(self):
        return iter(self.children.keys())

    def __len__(self):
        return len(self.children)

    def __getitem__(self, name):
        return getattr(self, name)

    # Dunder methods.

    def __getattr__(self, name):
        if name in self.children:
            return self.children[name]
        raise AttributeError(f"No such attribute '{name}'.")
