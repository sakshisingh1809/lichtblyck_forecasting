"""Base class for PfLine and PfState classes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd


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
    def df(self, cols: Iterable[str] = None, flatten: bool = True) -> pd.DataFrame:
        """DataFrame for this object.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns to include in the dataframe.
        flatten : bool, optional (default: True)
            - If True, include only aggregated timeseries (4 or less; 1 per dimension).
            - If False, include all timerseries and their (intermediate and final)
              aggregations.

        Returns
        -------
        pd.DataFrame
        """
        ...

    @abstractmethod
    def asfreq(self, freq: str = "MS") -> NDFrameLike:
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
