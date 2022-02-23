"""Enable arithmatic with PfLine classes."""

from __future__ import annotations

from . import base, single, multi
from ...tools.nits import Q_, unit2name

from typing import TYPE_CHECKING, Union
import pandas as pd

if TYPE_CHECKING:  # needed to avoid circular imports
    from .base import PfLine

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


def _prep_data(value, ref: PfLine) -> Union[pd.Series, PfLine]:
    """Turn ``value`` into PfLine if possible. If not, turn into (normal or unit-aware) Series."""

    # Already a PfLine.
    if isinstance(value, base.PfLine):
        return value

    # Series.
    if isinstance(value, pd.Series):
        if not hasattr(value, "pint"):  # has no unit
            return value

        try:
            name = unit2name(value.pint.units)
        except ValueError:
            return value  # has unit, but unknown

        if name not in ["p", "q", "w"]:
            return value  # has known unit, but none from which PfLine can be made

        return single.SinglePfLine({name: value})

    # Just a single value.
    if isinstance(value, int) or isinstance(value, float):
        s = pd.Series(value, ref.index)
        return _prep_data(s, ref)
    elif isinstance(value, Q_):
        s = pd.Series(value.magnitude, ref.index).astype(f"pint[{value.units}]")
        return _prep_data(s, ref)

    raise TypeError(f"Cannot handle inputs of this type; got {type(value)}.")


def _flatten(fn):
    """Flatten the portfoliolines before calling the wrapped function."""

    def wrapper(*pflines):
        pflines = [pfl.flatten() for pfl in pflines]
        return fn(*pflines)

    return wrapper


def _add_pflines(pfl1: PfLine, pfl2: PfLine):
    """Add two pflines."""
    if pfl1.kind != pfl2.kind:
        raise NotImplementedError("Cannot add portfolio lines of unequal kind.")

    if isinstance(pfl1, multi.MultiPfLine) and isinstance(pfl2, multi.MultiPfLine):
        # If BOTH are MultiPfLines, collect children and add those with same name.
        names = set([*pfl1.children.keys(), *pfl2.children.keys()])
        children = {}
        for name in names:
            child1, child2 = pfl1.children.get(name), pfl2.children.get(name)
            if child1 is not None and child2 is not None:
                children[name] = child1 + child2
            elif child1 is not None:
                children[name] = child1
            else:
                children[name] = child2
        return multi.MultiPfLine(children)

    else:  # at least one of them is a SinglePfLine.
        # Get addition and keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        dfs = [pfl.df(pfl.summable) for pfl in [pfl1, pfl2]]
        df = sum(dfs).dropna().resample(pfl1.index.freq).asfreq()
        return single.SinglePfLine(df)


@_flatten  # TODO: Decide if this should return a Single or Multi PfLine
def _multiply_pflines(pfl1: PfLine, pfl2: PfLine):
    """Multiply two pflines."""
    if set([pfl1.kind, pfl2.kind]) != {"p", "q"}:
        raise NotImplementedError("Can only multiply volume with price information.")

    if pfl1.kind == "p":
        data = {"q": pfl2.q, "p": pfl1.p}
    else:  # pfl1.kind == "q":
        data = {"q": pfl1.q, "p": pfl2.p}
    return single.SinglePfLine(data)


def _add_pfline_and_dimensionlessseries(pfl: PfLine, s: pd.Series):
    """Add pfline and dimensionless series."""
    if not isinstance(pfl, single.SinglePfLine) or pfl.kind != "p":
        raise NotImplementedError(
            "Value(s) without unit can only be added to a single portfolio line with price information."
        )

    # Cast to price, keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
    df = pd.DataFrame({"p": pfl.p + s.astype("pint[Eur/MWh]")})
    df = df.dropna().resample(pfl.index.freq).asfreq()
    return single.SinglePfLine(df)


def _multiply_pfline_and_dimensionlessseries(pfl: PfLine, s: pd.Series):
    """Multiply pfline and dimensionless series."""
    if not isinstance(pfl, single.SinglePfLine) or pfl.kind == "all":
        raise NotImplementedError(
            "Value(s) without unit can only be multiplied with a single portfolio line with price or volume information."
        )

    # Scale the price p (kind == 'p') or the volume q (kind == 'q'), returning PfLine of same kind.
    df = pfl.df(pfl.summable).mul(s, axis=0)  # multiplication with index-alignment
    df = df.dropna().resample(pfl.index.freq).asfreq()
    return single.SinglePfLine(df)


def _divide_pflines(pfl1: PfLine, pfl2: PfLine) -> pd.Series:
    """Divide two pflines."""
    if pfl1.kind != pfl2.kind or pfl1.kind == "all":
        raise NotImplementedError(
            "Can only divide portfolio lines if both contain price-only or both contain volume-only information."
        )

    if pfl1.kind == "p":
        s = pfl1.p / pfl2.p
    else:  # self.kind == "q"
        s = pfl1.q / pfl2.q
    s = s.dropna().resample(pfl1.index.freq).asfreq()
    return s.rename("fraction")  # pint[dimensionless]


def _assert_freq_compatibility(o1, o2):
    if o1.index.freq != o2.index.freq:
        raise NotImplementedError(
            "Cannot do arithmatic with timeseries of unequal frequency."
        )


class PfLineArithmatic:
    METHODS = ["neg", "add", "radd", "sub", "rsub", "mul", "rmul", "truediv"]

    def __neg__(self: PfLine):
        # invert price (kind == 'p'), volume (kind == 'q') or volume and revenue (kind == 'all')
        flat = self.flatten()
        df = (-flat.df(flat.summable).pint.dequantify()).pint.quantify()
        return single.SinglePfLine(df)

    def __add__(self: PfLine, other) -> PfLine:
        if not other:
            return self

        other = _prep_data(other, self)  # other is now a PfLine or Series.
        _assert_freq_compatibility(self, other)

        # Other is a PfLine.
        if isinstance(other, base.PfLine):
            return _add_pflines(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            if not hasattr(other, "pint"):  # no unit information
                return _add_pfline_and_dimensionlessseries(self, other)

        raise NotImplementedError("This addition is not defined.")

    __radd__ = __add__

    def __sub__(self: PfLine, other):
        return self + -other if other else self  # defer to mul and neg

    def __rsub__(self: PfLine, other):
        return other + -self  # defer to mul and neg

    def __mul__(self: PfLine, other) -> PfLine:

        other = _prep_data(other, self)  # other is now a PfLine or Series.
        _assert_freq_compatibility(self, other)

        # Other is a PfLine.
        if isinstance(other, base.PfLine):
            return _multiply_pflines(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            if not hasattr(other, "pint"):  # no unit information
                return _multiply_pfline_and_dimensionlessseries(self, other)

        raise NotImplementedError("This multiplication is not defined.")

    __rmul__ = __mul__

    def __truediv__(self: PfLine, other):
        other = _prep_data(other, self)  # other is now a PfLine or Series.
        _assert_freq_compatibility(self, other)

        # Other is a PfLine.
        if isinstance(other, base.PfLine):
            return _divide_pflines(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return self * (1 / other)  # defer to mul

        raise NotImplementedError("This division is not defined.")


def apply():
    for attr in PfLineArithmatic.METHODS:
        attrname = f"__{attr}__"
        setattr(base.PfLine, attrname, getattr(PfLineArithmatic, attrname))
