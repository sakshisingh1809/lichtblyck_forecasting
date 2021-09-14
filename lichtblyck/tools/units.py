"""
Module for working with physical units.
"""


from __future__ import annotations
from typing import Optional, Tuple, Union, Dict, FrozenSet
from enum import Enum
from dataclasses import dataclass
import functools


class UF(Enum):  # UnitFactor
    MILLI = 0.001
    ONE = 1
    THOUSAND = 1_000
    MILLION = 1_000_000

    __add__ = __radd__ = __sub__ = __rsub__ = lambda self, other: NotImplemented

    def __mul__(self, other):
        if type(other) is not UF:
            return self * UF(other)
        return UF(self.value * other.value)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) is not UF:
            return self / UF(other)
        return UF(self.value / other.value)

    def __rtruediv__(self, other):
        return UF(other) / self


class BU(Enum):  # BaseUnit
    # Values: `symbol` (abbreviation), `col` (customary column name)
    NONE = "", ""
    HOUR = "h", "duration"
    MEGAWATT = "MW", "w"
    MEGAWATTHOUR = "MWh", "q"
    EURO_PER_MEGAWATTHOUR = "Eur/MWh", "p"
    EURO = "Eur", "r"
    DEGREE_CELSIUS = "degC", "t"

    @functools.lru_cache(1)
    def multiplyresults(self,) -> Dict[FrozenSet[BU], BU]:  # {{A, B}: A*B}
        return {
            frozenset((BU.MEGAWATT, BU.HOUR)): BU.MEGAWATTHOUR,
            frozenset((BU.MEGAWATTHOUR, BU.EURO_PER_MEGAWATTHOUR)): BU.EURO,
            frozenset((BU.NONE, self)): self,
        }

    @classmethod
    def _missing_(cls, val: str):  # can be any item of a valid enum value
        if not isinstance(val, str):
            return
        val = val.upper().replace("_", "/").replace("°", "DEG")
        for member in cls:
            for v in member.value:
                if v.upper() == val:
                    return member

    symbol: str = property(lambda self: self.value[0])
    col: str = property(lambda self: self.value[1])

    def __len__(self):
        return len(str(self))

    def __str__(self):
        return self.symbol

    # Mathematical operations.

    def __add__(self, other):  # can only add values with equal units
        return self if self is other else NotImplemented

    __radd__ = __sub__ = __rsub__ = __add__

    def __mul__(self, other):
        for multipliers, product in self.multiplyresults().items():
            if multipliers == {self, other}:
                return product
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        for multipliers, product in self.multiplyresults().items():
            if self is product and other in multipliers:
                return next(iter(multipliers - {other}))
        return NotImplemented

    def __rtruediv__(self, other):
        return (other / self) if type(other) is Unit else NotImplemented


class Unit(Enum):
    # symbol, unitfactor, baseunit
    NONE = "", UF.ONE, BU.NONE
    HOUR = "h", UF.ONE, BU.HOUR
    KILOWATT = "kW", UF.MILLI, BU.MEGAWATT
    MEGAWATT = "MW", UF.ONE, BU.MEGAWATT
    GIGAWATT = "GW", UF.THOUSAND, BU.MEGAWATT
    KILOWATTHOUR = "kWh", UF.MILLI, BU.MEGAWATTHOUR
    MEGAWATTHOUR = "MWh", UF.ONE, BU.MEGAWATTHOUR
    GIGAWATTHOUR = "GWh", UF.THOUSAND, BU.MEGAWATTHOUR
    TERRAWATTHOUR = "TWh", UF.MILLION, BU.MEGAWATTHOUR
    EURO_PER_MEGAWATTHOUR = "Eur/MWh", UF.ONE, BU.EURO_PER_MEGAWATTHOUR
    EURO = "Eur", UF.ONE, BU.EURO
    KILOEURO = "kEur", UF.THOUSAND, BU.EURO
    MEGAEURO = "MEur", UF.MILLION, BU.EURO
    DEGREE_CELSIUS = "degC", UF.ONE, BU.DEGREE_CELSIUS

    @classmethod
    def _missing_(cls, value: Union[str, BU, Tuple]):
        if isinstance(value, str):
            value = value.upper().replace("_", "/").replace("°", "DEG")
            for member in cls:
                if member.symbol.upper() == value:
                    return member
        else:
            if isinstance(value, BU):
                uf, bu = UF.ONE, value
            elif isinstance(value, Tuple):
                uf, bu = value
            for member in cls:
                if member.uf is uf and member.bu is bu:
                    return member

    symbol: str = property(lambda self: self.value[0])
    uf: UF = property(lambda self: self.value[1])
    bu: BU = property(lambda self: self.value[2])

    factor: float = property(lambda self: self.uf.value)
    baseunit_symbol: str = property(lambda self: self.bu.symbol)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return f"{self.uf.name} * {self.bu.name}"

    # Mathematical operations.

    def __add__(self, other):
        if self is other:
            return self  # can only add equal units (factor must also be same)
        return NotImplemented

    __radd__ = __sub__ = __rsub__ = __add__

    def __mul__(self, other):
        if type(other) is Unit:
            return Unit((self.uf * other.uf, self.bu * other.bu))
        if type(other) is UF or isinstance(other, float) or isinstance(other, int):
            return Unit((self.uf * other, self.bu))
        if type(other) is BU:
            return Unit((self.uf, self.bu * other))
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) is Unit:
            return Unit((self.uf / other.uf, self.bu / other.bu))
        if type(other) is UF or isinstance(other, float) or isinstance(other, int):
            return Unit((self.uf / other, self.bu))
        if type(other) is BU:
            return Unit((self.uf, self.bu / other))
        return NotImplemented

    def __rtruediv__(self, other):
        if type(other) is Unit:
            return other / self
        if type(other) is UF or isinstance(other, float) or isinstance(other, int):
            return Unit((other / self.uf, 1 / self.bu))
        if type(other) is BU:
            return Unit((1 / self.uf, other / self.bu))
        return NotImplemented
