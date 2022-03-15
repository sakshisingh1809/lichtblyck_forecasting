"""
Module to create multi-pflines. These behave exactly as single-pflines, with all the
same methods. But which include several Pflines as children that can be accessed by
their name.
"""

from __future__ import annotations

from . import multi_helper
from .base import PfLine

from typing import Dict, Iterable, Mapping, Optional, Union
import pandas as pd
import numpy as np
import warnings


class MultiPfLine(PfLine):
    """Portfolio line with children, can be found in .children dictionary attribute
    and can also be accessed by name from class instance.

    Parameters
    ----------
    data: Any
        Generally: object with a mapping from strings to PfLine instances; most commonly a
        dictionary.
    """

    def __new__(cls, data):
        # Catch case where data is already a valid class instance.
        if isinstance(data, MultiPfLine):
            return data  # TODO: make copy instead
        # Otherwise, do normal thing.
        return super().__new__(cls, data)

    def __init__(self, data: Union[MultiPfLine, Mapping[str, PfLine]]):
        self._children = multi_helper.make_childrendict(data)

    # Implementation of ABC methods.

    @property
    def children(self) -> Dict[str, PfLine]:
        return self._children

    @property
    def index(self) -> pd.DatetimeIndex:
        return next(iter(self._children.values())).index

    @property
    def w(self) -> pd.Series:
        if self.kind == "p":
            return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")
        else:
            return pd.Series(self.q / self.index.duration, name="w").pint.to("MW")

    @property
    def q(self) -> pd.Series:
        if self.kind == "p":
            return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")
        elif (qp_children := self._qp_children) is not None:
            return qp_children["q"].q
        else:  # all children have a sensible timeseries for .q
            return sum(child.q for child in self._children.values()).rename("q")

    @property
    def p(self) -> pd.Series:
        if self.kind == "q":
            return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")
        elif (qp_children := self._qp_children) is not None:
            return qp_children["p"].p
        elif self.kind == "all":  # all children have .kind == 'all'
            return pd.Series(self.r / self.q, name="p").pint.to("Eur/MWh")
        else:  # self.kind == 'p', all children have a sensible timeseries for .p
            return sum(child.p for child in self._children.values()).rename("p")

    @property
    def r(self) -> pd.Series:
        if self.kind != "all":
            return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")
        elif (qp_children := self._qp_children) is not None:
            q, p = qp_children["q"].q, qp_children["p"].p
            return pd.Series(q * p, name="r").pint.to("Eur")
        else:  # all children have .kind == 'all'
            return sum(child.r for child in self._children.values()).rename("r")

    @property
    def kind(self) -> str:
        if self._heterogeneous_children:
            return "all"
        return next(iter(self._children.values())).kind

    def df(self, cols: Iterable[str] = None, flatten: bool = True) -> pd.DataFrame:
        if flatten:
            cols = self.available if cols is None else cols
            return pd.DataFrame({col: self[col] for col in cols})
        # One big dataframe. First: collect constituent dataframes.
        dfs = [self.df(cols, True)]
        dfdicts = [{n: c.df(cols, False)} for n, c in self._children.items()]
        dfs.extend([pd.concat(dfdict, axis=1) for dfdict in dfdicts])
        # Then: make all have same number of levels.
        n_target = max([df.columns.nlevels for df in dfs])
        for df in dfs:
            n_current = df.columns.nlevels
            keys = [""] * (n_target - n_current)
            oldcol = df.columns if n_current > 1 else [[item] for item in df.columns]
            df.columns = pd.MultiIndex.from_tuples(((*item, *keys) for item in oldcol))
        # Finally: put all together in big new dataframe.
        return pd.concat(dfs, axis=1)

    def asfreq(self, freq: str = "MS") -> MultiPfLine:
        if self._heterogeneous_children:
            warnings.warn(
                "This portfolio has its price and volume information stored in distinct child porfolios. The portfolio is flattened before changing its frequency."
            )
            return self.flatten().asfreq(freq)
        return MultiPfLine(
            {label: child.asfreq(freq) for label, child in self._children.items()}
        )

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._children == other._children

    def __bool__(self) -> bool:
        # True if any of the children are true.
        return any(self._children.keys())

    # Additional methods, unique to this class.

    @property
    def _heterogeneous_children(self) -> bool:
        """Return True if children are not all of same kind."""
        return bool(self._qp_children)

    @property
    def _qp_children(self) -> Optional[Dict]:
        """Helper method that returns the child providing the volume and the one providing the price."""
        qp_children = {child.kind: child for child in self._children.values()}
        if "q" in qp_children and "p" in qp_children:
            return qp_children


class _LocIndexer:
    """Helper class to obtain MultiPfLine instance, whose index is subset of original index."""

    def __init__(self, mpfl):
        self.mpfl = mpfl

    def __getitem__(self, arg) -> MultiPfLine:
        new_dict = {name: child.loc[arg] for name, child in self.mpfl.children.items()}
        return MultiPfLine(new_dict)
