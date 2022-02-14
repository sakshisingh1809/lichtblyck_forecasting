"""
Module to create multi-pflines. These behave exactly as single-pflines, with all the 
same methods. But which include several Pflines as children that can be accessed by 
their name.
"""

from __future__ import annotations
from .abc import PfLine
from .multi_helper import make_childrendict
from typing import Dict, Iterable, Mapping, Union
import pandas as pd


class MultiPfLine(PfLine):
    def __init__(self, data: Union[MultiPfLine, Mapping[str, PfLine]]):
        self._children = make_childrendict(data)

    # Implementation of ABC methods.

    @property
    def children(self) -> Dict[str, PfLine]:
        return self._children

    @property
    def index(self) -> pd.DatetimeIndex:
        return next(iter(self.children.values())).index

    @property
    def w(self) -> pd.Series:
        return sum(child.w for child in self.children.values())

    @property
    def q(self) -> pd.Series:
        return sum(child.q for child in self.children.values())

    @property
    def p(self) -> pd.Series:
        if self.kind == "p":
            return sum(child.p for child in self.children.values())
        return pd.Series(self.r / self.q, name="p").pint.to("Eur/MWh")

    @property
    def r(self) -> pd.Series:
        return sum(child.r for child in self.children.values())

    @property
    def kind(self) -> str:
        return next(iter(self.children.values())).kind

    def df(self, cols: Iterable[str] = None) -> pd.DataFrame:
        dfdict = {label: child.df(cols) for label, child in self.children.items()}
        return pd.concat(dfdict, axis=1)

    def changefreq(self, freq: str = "MS") -> MultiPfLine:
        return MultiPfLine(
            {label: child.changefreq(freq) for label, child in self.children.items()}
        )

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)


class _LocIndexer:
    """Helper class to obtain subset of the PfLine index."""

    def __init__(self, mpfl):
        self.mpfl = mpfl

    def __getitem__(self, arg) -> MultiPfLine:
        new_dict = {name: child.loc[arg] for name, child in self.mpfl.children.items()}
        return MultiPfLine(new_dict)
