"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios, at a 
certain moment in time (e.g., at the current moment, without any historic data).
"""

from __future__ import annotations


from .pfstate_helper import make_pflines
from .. import utils
from ..base import NDFrameLike
from ..pfline import PfLine, SinglePfLine, MultiPfLine
from ..mixins import PfStateText, PfStatePlot, OtherOutput
from typing import Optional, Iterable, Union
import pandas as pd
import warnings


class PfState(NDFrameLike, PfStateText, PfStatePlot, OtherOutput):
    """Class to hold timeseries information of an energy portfolio, at a specific moment.

    Parameters
    ----------
    offtakevolume, unsourcedprice, sourced : PfLine
        `offtakevolume` may also be passed as pd.Series with name `q` or `w`.
        `unsourcedprice` may also be passed as pd.Series.
        `sourced` is optional; if non is specified, assume no sourcing has taken place.

    Attributes
    ----------
    offtake : PfLine ('q')
        Offtake. Volumes are <0 for all timestamps (see 'Notes' below).
    sourced : PfLine ('all')
        Procurement. Volumes (and normally, revenues) are >0 for all timestamps (see
        'Notes' below).
    unsourced : PfLine ('all')
        Procurement/trade that is still necessary until delivery. Volumes (and normally,
        revenues) are >0 if more volume must be bought, <0 if volume must be sold for a
        given timestamp (see 'Notes' below). NB: if volume for a timestamp is 0, its
        price is undefined (NaN) - to get the market prices in this portfolio, use the
        property `.unsourcedprice` instead.
    unsourcedprice : PfLine ('p')
        Prices of the unsourced volume.
    netposition : PfLine ('all')
        Net portfolio positions. Convenience property for users with a "traders' view".
        Does not follow sign conventions (see 'Notes' below); volumes are <0 if
        portfolio is short and >0 if long. Identical to `.unsourced`, but with sign
        change for volumes and revenues (but not prices).
    procurement : PfLine ('all')
        The expected costs needed to source the offtake volume; the sum of the sourced
        and unsourced positions.

    index : pandas.DateTimeIndex
        Left timestamp of row.

    Notes
    -----
    Sign conventions:
    . Volumes (`q`, `w`): >0 if volume flows into the portfolio.
    . Revenues (`r`): >0 if money flows out of the portfolio (i.e., costs).
    . Prices (`p`): normally positive.
    """

    @classmethod
    def from_series(
        cls,
        *,
        pu: pd.Series,
        qo: Optional[pd.Series] = None,
        qs: Optional[pd.Series] = None,
        rs: Optional[pd.Series] = None,
        wo: Optional[pd.Series] = None,
        ws: Optional[pd.Series] = None,
        ps: Optional[pd.Series] = None,
    ):
        """Create Portfolio instance from timeseries.

        Parameters
        ----------
        unsourced prices:
            `pu` [Eur/MWh]
        offtake volume: one of
            `qo` [MWh]
            `wo` [MW]
        sourced volume and revenue: two of
            (`qs` [MWh] or `ws` [MW])
            `rs` [Eur]
            `ps` [Eur/MWh]
            If no volume has been sourced, all 4 sourced timeseries may be None.

        Returns
        -------
        PfState
        """
        if ws is not None or qs is not None or rs is not None or ps is not None:
            sourced = PfLine({"w": ws, "q": qs, "r": rs, "p": ps})
        else:
            sourced = None
        return cls(PfLine({"q": qo, "w": wo}), PfLine({"p": pu}), sourced)

    def __init__(
        self,
        offtakevolume: Union[PfLine, pd.Series],
        unsourcedprice: Union[PfLine, pd.Series],
        sourced: Optional[PfLine],
    ):
        # The only internal data of this class is stored as PfLines.
        self._offtakevolume, self._unsourcedprice, self._sourced = make_pflines(
            offtakevolume, unsourcedprice, sourced
        )

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._offtakevolume.index

    @property
    def offtake(self) -> PfLine:
        return self._offtakevolume

    @property
    def sourced(self) -> PfLine:
        return self._sourced

    @property
    def unsourced(self) -> PfLine:
        sourcedvolume = 0 if self._sourced is None else self._sourced.volume
        return -(self._offtakevolume + sourcedvolume) * self._unsourcedprice

    @property
    def unsourcedprice(self) -> PfLine:
        return self._unsourcedprice

    @property
    def netposition(self) -> PfLine:
        return -self.unsourced

    @property
    def pnl_cost(self):
        return MultiPfLine({"sourced": self.sourced, "unsourced": self.unsourced})

    @property
    def sourcedfraction(self) -> pd.Series:
        return -self._sourced.volume / self._offtakevolume

    @property
    def unsourcedfraction(self) -> pd.Series:
        return 1 - self.sourcedfraction

    def df(self, flatten: bool = False, *args, **kwargs) -> pd.DataFrame:
        """DataFrame for this PfState.

        Returns
        -------
        pd.DataFrame
        """
        dfs = []
        for part in ("offtake", "pnl_cost", "sourced", "unsourced"):
            fl = True if part == "pnl_cost" else flatten  # always flatten pnl_cost
            dfin = self[part].df(flatten=fl)
            dfs.append(utils.add_header(dfin, part))
        return utils.concat(dfs, axis=1)

    # Methods that return new class instance.

    def set_offtakevolume(self, offtakevolume: PfLine) -> PfState:
        warnings.warn(
            "This changes the unsourced volume and causes inaccuracies in its price, if the portfolio has a frequency that is longer than the spot market."
        )
        return PfState(offtakevolume, self._unsourcedprice, self._sourced)

    def set_unsourcedprice(self, unsourcedprice: PfLine) -> PfState:
        return PfState(self._offtakevolume, unsourcedprice, self._sourced)

    def set_sourced(self, sourced: PfLine) -> PfState:
        warnings.warn(
            "This changes the unsourced volume and causes inaccuracies in its price, if the portfolio has a frequency that is longer than the spot market."
        )
        return PfState(self._offtakevolume, self._unsourcedprice, sourced)

    def add_sourced(self, add_sourced: PfLine) -> PfState:
        return self.set_sourced(self.sourced + add_sourced)

    def asfreq(self, freq: str = "MS") -> PfState:
        """Resample the Portfolio to a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.

        Returns
        -------
        PfState
            Resampled at wanted frequency.
        """
        # pu resampling is most important, so that prices are correctly weighted.
        offtakevolume = self.offtake.asfreq(freq).volume
        unsourcedprice = self.unsourced.asfreq(freq).price  # ensures weighted avg
        sourced = self.sourced.asfreq(freq)
        return PfState(offtakevolume, unsourcedprice, sourced)

    # Dunder methods.

    def __getitem__(self, name):
        if hasattr(self, name):
            return getattr(self, name)

    def __eq__(self, other):
        if not isinstance(other, PfState):
            return False
        return all(
            [self[part] == other[part] for part in ["offtake", "unsourced", "sourced"]]
        )

    def __bool__(self):
        return True

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    # Additional methods, unique to this class.


class _LocIndexer:
    """Helper class to obtain PfState instance, whose index is subset of original index."""

    def __init__(self, pfs):
        self.pfs = pfs

    def __getitem__(self, arg) -> PfState:
        offtakevolume = self.pfs.offtake.volume.loc[arg]
        unsourcedprice = self.pfs.unsourcedprice.loc[arg]
        sourced = self.pfs.sourced.loc[arg]
        return PfState(offtakevolume, unsourcedprice, sourced)


from . import enable_arithmatic, enable_hedging

enable_arithmatic.apply()
enable_hedging.apply()
