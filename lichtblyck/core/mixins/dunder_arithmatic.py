"""
Module with mixins, to add arithmatic functionality to PfLine and PfState classes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import pandas as pd
from ...tools.nits import ureg, Q_, unit2name
from ...tools.frames import wavg

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfstate import PfState
    from ..pfline import PfLine


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


# Compatibility:
#
# General
#
# Physically true:
# unitA + unitA = unitA
# unitA * dimensionless = unitA
# unitA / dimensionless = unitA
# dimensionless / unitA = 1/unitA
#
# In addition, accepted as true:
# unitA + dimensionless = unitA
# Eur/MWh * MWh -> all-PfLine
# Eur/MWh * MW -> all-PfLine
#
#
# Implementation
#
# Before anything else: turn 'other' into p-PfLine or q-PfLine if possible, or else into
# a pd.Series. So, if other is single quantity, or pd.Series, in Eur/MWh, MW, or MWh,
# this is turned into p-PfLine or q-PfLine.
#                 other
#                 Eur/MWh                                          => p-PfLine
#                 MW, MWh                                          => q-PfLine
#                 other unit or dimensionless                      => pd.Series
# Values and Series in other units are considered under 'dimension', below.
#
# self            other                                               return
# -------------------------------------------------------------------------------
# p-PfLine      + p-PfLine, dimensionless                           = p-PfLine
#               + q-PfLine, all-PfLine, dimension                   = error
#               - anything                                          => + and neg
#               * dimensionless                                     = p-PfLine
#               * q-PfLine                                          = all-PfLine
#               * p-PfLine, all-PfLine, dimension                   = error
#               / dimensionless, dimension                          => *
#               / p-PfLine                                          = pd.Series
#               / q-PfLine, all-PfLine                              = error
# q-PfLine      + p-PfLine                                          => see above
#               + q-PfLine                                          = q-PfLine
#               + dimensionless, dimension, all-PfLine              = error
#               - anything                                          => + and neg
#               * p-PfLine                                          => see above
#               * dimensionless                                     = q-PfLine
#               * q-PfLine, all-PfLine, dimension                   = error
#               / dimensionless, dimension                          => *
#               / q-PfLine                                          = pd.Series
#               / p-PfLine, all-PfLine                              = error
# all-PfLine    + p-PfLine, q-PfLine                                => see above
#               + all-PfLine                                        = all-PfLine
#               + dimensionless, dimension                          = error
#               - anything                                          => * and neg
#               * p-PfLine, q-PfLine                                => see above
#               * dimensionless                                     = all-PfLine (keep p)
#               * dimension, all-PfLine                             = error
#               / dimensionless, dimension, p-PfLine, q-PfLine, all-PfLine = error


class PfLineArithmatic:
    def _prep_other(self: PfLine, other) -> Union[pd.Series, PfLine]:
        """Turn `other` into PfLine if possible. If not, turn into (normal or unit-aware)
        Series."""
        if isinstance(other, int) or isinstance(other, float):
            return pd.Series(other, self.index)
        elif isinstance(other, Q_):
            s = pd.Series(other.magnitude, self.index).astype(f"pint[{other.units}]")
            return self._prep_other(s)
        elif isinstance(other, pd.Series):
            if not hasattr(other, "pint"):
                return other  # has no unit
            try:
                name = unit2name(other.pint.units)
            except ValueError:
                return other  # has unit, but unknown
            if name not in ["p", "q", "w"]:
                return other  # has know unit, but none from which PfLine can be made
            return self.__class__({name: other})
        elif isinstance(other, self.__class__):
            return other
        raise TypeError(f"Cannot handle inputs of this type ({type(other)}).")

    def __add__(self: PfLine, other) -> PfLine:
        if not other:
            return self
        other = self._prep_other(other)  # other is now a PfLine or Series.

        # Other is a PfLine.
        if isinstance(other, self.__class__):
            if self.kind != other.kind:
                raise NotImplementedError("Cannot add portfolio lines of unequal kind.")
            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot add portfolio lines of unequal frequency."
                )
            # Get addition and keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
            dfs = [pfl.df(pfl.summable) for pfl in [self, other]]
            df = sum(dfs).dropna().resample(self.index.freq).asfreq()
            return self.__class__(df)

        # Other is a Series (but not containing [power], [energy] or [price]).
        elif isinstance(other, pd.Series):
            if not hasattr(other, "pint"):  # no unit information
                if self.index.freq != other.index.freq:
                    raise NotImplementedError(
                        "Cannot add timeseries of unequal frequency."
                    )
                if self.kind != "p":
                    raise NotImplementedError(
                        "Value(s) without unit can only be added to portfolio line with price information."
                    )
                # Cast to price, keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
                df = pd.DataFrame({"p": self.p + other.astype("pint[Eur/MWh]")})
                df = df.dropna().resample(self.index.freq).asfreq()
                return self.__class__(df)
        raise NotImplementedError("This addition is not defined.")

    __radd__ = __add__

    def __sub__(self: PfLine, other):
        return self + -other if other else self  # defer to mul and neg

    def __rsub__(self: PfLine, other):
        return other + -self  # defer to mul and neg

    def __mul__(self: PfLine, other) -> PfLine:
        if self.kind == "all":
            raise NotImplementedError(
                "Cannot multiply PfLine containing volume and price information."
            )
        other = self._prep_other(other)  # other is now a PfLine or Series.

        # Other is a PfLine.
        if isinstance(other, self.__class__):

            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot multiply portfolio lines of unequal frequency."
                )
            elif self.kind == "p" and other.kind == "q":
                df = pd.DataFrame({"q": other.q, "p": self.p})
            elif self.kind == "q" and other.kind == "p":
                df = pd.DataFrame({"q": self.q, "p": other.p})
            else:
                raise NotImplementedError(
                    "Can only multiply volume with price information."
                )
            # Keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
            df = df.dropna().resample(self.index.freq).asfreq()
            return self.__class__(df)

        # Other is a Series (but not containing [power], [energy] or [price]).
        elif isinstance(other, pd.Series):
            if not hasattr(other, "pint"):  # no unit information
                if self.index.freq != other.index.freq:
                    raise NotImplementedError(
                        "Cannot multiply timeseries of unequal frequency."
                    )
                # Scale the price p (kind == 'p') or the volume q (kind == 'q'), returning PfLine of same kind.
                df = self._df.mul(other, axis=0)  # multiplication with index-alignment
                df = df.dropna().resample(self.index.freq).asfreq()
                return self.__class__(df)
        raise NotImplementedError("This multiplication is not defined.")

    __rmul__ = __mul__

    def __neg__(self: PfLine):
        # invert price (kind == 'p'), volume (kind == 'q') or volume and revenue (kind == 'all')
        return self.__class__((-self._df.pint.dequantify()).pint.quantify())

    def __truediv__(self: PfLine, other):
        other = self._prep_other(other)  # other is now a PfLine or Series.

        # Other is a PfLine.
        if isinstance(other, self.__class__):

            if self.kind != other.kind or self.kind == "all":
                raise NotImplementedError(
                    "Can only divide portfolio lines if both contain price-only or both contain volume-only information."
                )
            if self.index.freq != other.index.freq:
                raise NotImplementedError(
                    "Cannot divide portfolio lines of unequal frequency."
                )
            if self.kind == "p":
                s = self.p / other.p
            else:  # self.kind == "q"
                s = self.q / other.q
            s = s.dropna().resample(self.index.freq).asfreq()
            return s.pint.magnitude.rename("fraction")

        # Other is a Series (but not containing [power], [energy] or [price]).
        elif isinstance(other, pd.Series):
            return self * (1 / other)  # defer to mul
        raise NotImplementedError("This division is not defined.")


class PfStateArithmatic:
    def _prep_other(self: PfState, other) -> Union[PfState, pd.Series]:
        """If not a PfState instance, turn `other` into Series."""
        if isinstance(other, int) or isinstance(other, float):
            return pd.Series(other, self.index)
        elif isinstance(other, Q_):
            return pd.Series(other.magnitude, self.index).astype(f"pint[{other.units}]")
        elif isinstance(other, pd.Series) or isinstance(other, self.__class__):
            return other
        raise TypeError(f"Cannot handle inputs of this type ({type(other)}).")

    def __add__(self: PfState, other):
        if not other:
            return self
        if not isinstance(other, self.__class__):
            raise NotImplementedError("This addition is not defined.")
        offtakevolume = self.offtake.volume + other.offtake.volume

        values = pd.DataFrame(
            {"s": self.unsourcedprice.p, "o": other.unsourcedprice.p}
        ).astype(float)
        weights = pd.DataFrame({"s": self.unsourced.q, "o": other.unsourced.q}).astype(
            float
        )
        unsourcedprice = wavg(values, weights, axis=1).rename("p")

        sourced = self.sourced + other.sourced
        return self.__class__(offtakevolume, unsourcedprice, sourced)

    __radd__ = __add__

    def __sub__(self: PfState, other):
        return self + -other if other else self  # defer to mul and neg

    def __rsub__(self: PfState, other):
        return other + -self  # defer to mul and neg

    def __mul__(self: PfState, other):

        other = self._prep_other(other)
        if not isinstance(other, pd.Series):
            raise NotImplementedError("This multiplication is not defined.")
        # Scale up volumes (and revenues), leave prices unchanged.
        offtakevolume = self.offtake.volume * other
        unsourcedprice = self.unsourcedprice
        sourced = self.sourced * other
        return self.__class__(offtakevolume, unsourcedprice, sourced)

    __rmul__ = __mul__

    def __neg__(self: PfState):
        # invert volumes and revenues, leave prices unchanged.
        return self.__class__(-self.offtake.volume, self.unsourcedprice, -self.sourced)

    def __truediv__(self: PfState, other):
        other = self._prep_other(other)
        if not isinstance(other, pd.Series):
            raise NotImplementedError("This division is not defined.")
        return self * (1 / other)  # defer to mul
