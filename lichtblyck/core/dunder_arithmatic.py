"""
Module with mixins, to add arithmatic functionality to PfLine and PfState classes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine


class PfLineArithmatic:
    
    def __add__(self: PfLine, other):
        if not other:
            return self
        if self.kind == "p" and (isinstance(other, float) or isinstance(other, int)):
            return self.__class__(self.df("p") + other)
        if not isinstance(other, self.__class__):
            raise NotImplementedError("This addition is not defined.")
        if self.kind != other.kind:
            raise ValueError("Cannot add portfolio lines of unequal kind.")
        if self.index.freq != other.index.freq:
            raise NotImplementedError("Cannot add portfolio lines of unequal frequency.")
        # Get addition and keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        dfs = [pfl.df(pfl._summable) for pfl in [self, other]]
        df = sum(dfs).dropna().resample(self.index.freq).asfreq()
        return self.__class__(df)

    def __radd__(self: PfLine, other):
        return self + other

    def __sub__(self: PfLine, other):
        return self + -1 * other if other else self

    def __rsub__(self : PfLine, other):
        return -1 * self + other

    def __mul__(self:PfLine, other):
        if isinstance(other, float) or isinstance(other, int):
            # multiply price p (kind == 'p'), volume q (kind == 'q'), or volume q and revenue r (kind == 'all').
            return self.__class__(self._df * other)
        if not isinstance(other, self.__class__):
            raise NotImplementedError("This multiplication is not defined.")
        if self.index.freq != other.index.freq:
            raise NotImplementedError("Cannot multiply portfolio lines of unequal frequency.")
        if self.kind == "p" and other.kind == "q":
            df = pd.DataFrame({"q": other.q, "p": self.p})
        elif self.kind == "q" and other.kind == "p":
            df = pd.DataFrame({"q": self.q, "p": other.p})
        else:
            raise ValueError("Can only multiply volume with price information.")
        # Keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        df = df.dropna().resample(self.index.freq).asfreq()
        return self.__class__(df)

    def __rmul__(self: PfLine, other):
        return self * other

    def __neg__(self: PfLine):
        return self * -1

    def __truediv__(self:PfLine, other):
        if not isinstance(other, float) and not isinstance(other, int):
            raise NotImplementedError("This division is not defined.")
        return self * (1 / other)



class PfStateArithmatic:

    def __add__(self: PfState, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError("This addition is not defined.")
        offtakevolume = self.offtake.volume + other.offtake.volume
        unsourcedprice = (self.unsourced + other.unsourced).price  # weighted average
        sourced = self.sourced + other.sourced
        return self.__class__(offtakevolume, unsourcedprice, sourced)

    def __radd__(self: PfState, other):
        return self + other

    def __sub__(self: PfState, other):
        return self + -1 * other

    def __rsub__(self: PfState, other):
        return -1 * self + other

    def __mul__(self: PfState, other):
        if not isinstance(other, float) and not isinstance(other, int):
            raise NotImplementedError("This multiplication is not defined.")
        offtakevolume = self.offtake.volume * other
        unsourcedprice = self.unsourcedprice
        sourced = self.sourced * other
        return self.__class__(offtakevolume, unsourcedprice, sourced)

    def __rmul__(self: PfState, other):
        return self * other

    def __neg__(self: PfState):
        return self * -1

    def __truediv__(self: PfState, other):
        if not isinstance(other, float) and not isinstance(other, int):
            raise NotImplementedError("This division is not defined.")
        return self * (1 / other)