"""Add arithmatic to PfLine classes."""

from __future__ import annotations
from ...tools.nits import Q_, unit2name
from ...tools.frames import wavg
from .. import pfline
from . import pfstate
from typing import TYPE_CHECKING, Union
import pandas as pd

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfstate import PfState
    from ..pfline import PfLine

# pfline = pfstate = None


def _prep_data(value, ref: PfState) -> Union[pd.Series, PfLine, PfState]:
    """Turn ``value`` into PfLine or PfState if possible. If not, turn into (normal or unit-aware) Series."""

    # Already a PfState.
    if isinstance(value, pfstate.PfState):
        return value

    # Already a PfLine.
    if isinstance(value, pfline.PfLine):
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
            return value  # has know unit, but none from which PfLine can be made

        return pfline.SinglePfLine({name: value})

    # Just a single value.
    if isinstance(value, int) or isinstance(value, float):
        s = pd.Series(value, ref.index)
        return _prep_data(s, ref)
    elif isinstance(value, Q_):
        s = pd.Series(value.magnitude, ref.index).astype(f"pint[{value.units}]")
        return _prep_data(s, ref)

    raise TypeError(f"Cannot handle inputs of this type; got {type(value)}.")


def _add_pfstates(pfs1: pfstate.PfState, pfs2: pfstate.PfState) -> pfstate.PfState:
    """Add two pfstates."""
    offtakevolume = pfs1.offtake.volume + pfs2.offtake.volume

    values = pd.DataFrame(
        {"s": pfs1.unsourcedprice.p, "o": pfs2.unsourcedprice.p}
    ).astype(float)
    weights = pd.DataFrame({"s": pfs1.unsourced.q, "o": pfs2.unsourced.q}).astype(float)
    unsourcedprice = wavg(values, weights, axis=1).rename("p")

    sourced = pfs1.sourced + pfs2.sourced
    return pfstate.PfState(offtakevolume, unsourcedprice, sourced)


def _multiply_pfstate_and_series(pfs: pfstate.PfState, s: pd.Series) -> pfstate.PfState:
    """Multiply pfstate and Series."""
    # Scale up volumes (and revenues), leave prices unchanged.
    offtakevolume = pfs.offtake.volume * s
    unsourcedprice = pfs.unsourcedprice
    sourced = pfs.sourced * s
    return pfstate.PfState(offtakevolume, unsourcedprice, sourced)


def _divide_pfstates(pfs1: pfstate.PfState, pfs2: pfstate.PfState) -> pd.DataFrame:
    """Divide two pfstates."""
    series = {}
    for part in [
        ("offtake", "volume"),
        ("sourced", "volume"),
        ("sourced", "price"),
        ("unsourced", "price"),
    ]:
        series[part] = pfs1[part[0]][part[1]] / pfs2[part[0]][part[1]]
    return pd.DataFrame(series)


def _assert_freq_compatibility(o1, o2):
    if o1.index.freq != o2.index.freq:
        raise NotImplementedError(
            "Cannot do arithmatic with timeseries of unequal frequency."
        )


class PfStateArithmatic:

    METHODS = ["neg", "add", "radd", "sub", "rsub", "mul", "rmul", "truediv"]

    def __neg__(self: PfState):
        # invert volumes and revenues, leave prices unchanged.
        return self.__class__(-self.offtake.volume, self.unsourcedprice, -self.sourced)

    def __add__(self: PfState, other):
        if not other:
            return self

        other = _prep_data(other, self)  # other is now a PfState, PfLine, or Series.
        _assert_freq_compatibility(self, other)

        # Other is a PfState.
        if isinstance(other, pfstate.PfState):
            return _add_pfstates(self, other)

        raise NotImplementedError("This addition is not defined.")

    __radd__ = __add__

    def __sub__(self: PfState, other):
        return self + -other if other else self  # defer to mul and neg

    def __rsub__(self: PfState, other):
        return other + -self  # defer to mul and neg

    def __mul__(self: PfState, other):

        other = _prep_data(other, self)  # other is now a PfState, PfLine, or Series.
        _assert_freq_compatibility(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return _multiply_pfstate_and_series(self, other)

        raise NotImplementedError("This multiplication is not defined.")

    __rmul__ = __mul__

    def __truediv__(self: PfState, other):
        other = _prep_data(other, self)  # other is now a PfState, PfLine, or Series.
        _assert_freq_compatibility(self, other)

        # Other is a PfState.
        if isinstance(other, pfstate.PfState):
            return _divide_pfstates(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return self * (1 / other)  # defer to mul

        raise NotImplementedError("This division is not defined.")


def apply():
    for attr in PfStateArithmatic.METHODS:
        attrname = f"__{attr}__"
        setattr(pfstate.PfState, attrname, getattr(PfStateArithmatic, attrname))
