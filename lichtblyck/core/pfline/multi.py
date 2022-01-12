"""Module to create multi-pflines. These behave exactly as single-pflines, with all the 
same methods. But which include several Pflines as children that can be accessed by 
their name."""


from typing import Dict, Iterable
from .abc import PfLine
import pandas as pd


def MultiPfLine(PfLineABC):
    def __init__(self, children: Dict[str, PfLine]):
        self._children = (
            children  # TODO: check if all same kind, reduce index to common index, etc.
        )

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
