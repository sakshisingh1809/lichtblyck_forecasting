"""
Module with mixins, to add arithmatic functionality to PfLine and PfState classes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd
from ..tools.nits import ureg, Q_, unit2name

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine


def is_dimensionless_value(value):
    try:
        _ = dimensionless_value(value)
        return True
    except ValueError:
        return False


def dimensionless_value(value):
    if isinstance(value, float) or isinstance(value, int):
        return value
    if isinstance(value, Q_) and not (Q_.dimensionality):
        return value.magnitude
    raise ValueError("Not a single value, or not dimensionless.")


def is_dimensionless_series(value):
    try:
        _ = dimensionless_series(value)
        return True
    except ValueError:
        return False


def dimensionless_series(value):
    if isinstance(value, pd.Series):
        try:
            p = value.pint
        except AttributeError:
            return value
        if not p.dimensionality:
            return p.magnitude
    raise ValueError("Not a pandas Series, or not dimensionless.")


class PfLineArithmatic:
    def __add__(self: PfLine, other) -> PfLine:
        if not other:
            return self
        # Other is single value (dimensionality unknown).
        elif (
            isinstance(other, float) or isinstance(other, int) or isinstance(other, Q_)
        ):
            if self.kind == "p":  # cast single value to price
                return self.__class__({"p": self.p + Q_(other, "Eur/MWh")})
            else:
                raise NotImplementedError(
                    "Single value can only be added to portfolio line with price information."
                )
        # Other is a series (dimensionality unknown).
        elif isinstance(other, pd.Series):
            if self.index.freq != other.index.freq:
                raise NotImplementedError("Cannot add timeseries of unequal frequency.")
            if self.kind == "p":  # cast series to price
                return self.__class__({"p": self.p + other.astype("pint[Eur/MWh]")})
            else:  # ...or with a dimension
                raise NotImplementedError(
                    "Series can only be added to portfolio line with price information."
                )
        # Other is a PfLine.
        elif isinstance(other, self.__class__):
            if self.kind != other.kind:
                raise NotImplementedError("Cannot add portfolio lines of unequal kind.")
            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot add portfolio lines of unequal frequency."
                )
            # Get addition and keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
            dfs = [pfl.df(pfl._summable) for pfl in [self, other]]
            df = sum(dfs).dropna().resample(self.index.freq).asfreq()
        else:
            raise NotImplementedError("This addition is not defined.")
        return self.__class__(df)

    __radd__ = __add__

    def __sub__(self: PfLine, other):
        return self + -1 * other if other else self  # defer to mul and add

    def __rsub__(self: PfLine, other):
        return other + -1 * self  # defer to mul and add

    def __mul__(self: PfLine, other) -> PfLine:
        if self.kind == "all":
            raise NotImplementedError(
                "Cannot multiply a portfolio line that has both price and volume information."
            )
        # Other is single value...
        elif is_dimensionless_value(other):  # ...without a dimension
            # Scale the price p (kind == 'p') or the volume q (kind == 'q'), returning PfLine of same kind.
            return self.__class__(self._df * dimensionless_value(other))
        elif isinstance(other, Q_):  # ... or with a dimension
            if self.kind == "p":  # price must be multiplied by *power*
                df = pd.DataFrame({"w": other.to("MW"), "p": self.p})
            elif self.kind == "q":  # volume must be multiplied by price
                df = pd.DataFrame({"q": self.q, "p": other.to("Eur/MWh")})
            else:
                NotImplementedError("Can only multiply volume with price information.")
        # Other is a series...
        elif isinstance(other, pd.Series):
            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot multiply timeseries of unequal frequency."
                )
            if is_dimensionless_series(other):  # ...without a dimension
                # Scale the price p (kind == 'p') or the volume q (kind == 'q') with the values, returning a PfLine of same kind.
                df = self._df * dimensionless_series(other)
            else:  # ...or with a dimension
                raise NotImplementedError(
                    "To multiply with a series that has a unit, first turn it into a portfolio line."
                )
        # Other is a PfLine.
        elif isinstance(other, self.__class__):
            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot multiply portfolio lines of unequal frequency."
                )
            if self.kind == "p" and other.kind == "q":
                df = pd.DataFrame({"q": other.q, "p": self.p})
            elif self.kind == "q" and other.kind == "p":
                df = pd.DataFrame({"q": self.q, "p": other.p})
            else:
                raise NotImplementedError(
                    "Can only multiply volume with price information."
                )
        else:
            raise NotImplementedError("This multiplication is not defined.")
        # Keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        df = df.dropna().resample(self.index.freq).asfreq()
        return self.__class__(df)

    __rmul__ = __mul__

    def __neg__(self: PfLine):
        return self * -1  # defer to mul

    def __truediv__(self: PfLine, other):
        # Other is single value (dimensionality unknown) or a series (dimensionality unknown).
        if (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, Q_)
            or isinstance(other, pd.Series)
        ):
            return self * (1 / other)  # defer to mul
        # Other is a PfLine.
        elif isinstance(other, self.__class__):
            if self.kind != other.kind or self.kind == "all":
                raise NotImplementedError(
                    "Can only divide portfolio lines if both contain only price or both contain only volume information."
                )
            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot divide portfolio lines of unequal frequency."
                )
            if self.kind == "p":
                s = self.p / other.p
            else:  # self.kind == "q"
                s = self.q / other.q
            return s.pint.magnitude.rename("fraction")
        else:
            raise NotImplementedError("This division is not defined.")


class PfStateArithmatic:
    def __add__(self: PfState, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError("This addition is not defined.")
        offtakevolume = self.offtake.volume + other.offtake.volume
        unsourcedprice = (self.unsourced + other.unsourced).price  # weighted average
        sourced = self.sourced + other.sourced
        return self.__class__(offtakevolume, unsourcedprice, sourced)

    __radd__ = __add__

    def __sub__(self: PfState, other):
        return self + -1 * other if other else self  # defer to mul and add

    def __rsub__(self: PfState, other):
        return other + -1 * self  # defer to mul and add

    def __mul__(self: PfState, other):

        
        if not isinstance(other, float) and not isinstance(other, int):
            raise NotImplementedError("This multiplication is not defined.")
        offtakevolume = self.offtake.volume * other
        unsourcedprice = self.unsourcedprice
        sourced = self.sourced * other
        return self.__class__(offtakevolume, unsourcedprice, sourced)

    __rmul__ = __mul__

    def __neg__(self: PfState):
        return self * -1  # defer to mul

    def __truediv__(self: PfState, other):
        # Other is single value (dimensionality unknown) or a series (dimensionality unknown).
        if (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, Q_)
            or isinstance(other, pd.Series)
        ):
            return self * (1 / other)  # defer to mul
        raise NotImplementedError("This division is not defined.")
