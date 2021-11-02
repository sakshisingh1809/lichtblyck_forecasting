"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios, at a 
certain moment in time (e.g., at the current moment, without any historic data).
"""

from __future__ import annotations
from .pfline import PfLine
from .output_text import PfStateTextOutput
from .output_plot import PfStatePlotOutput
from .dunder_arithmatic import PfStateArithmatic
from ..prices import convert, hedge
from typing import Optional, Iterable, Union
import pandas as pd
import warnings


def _make_pflines(offtakevolume, unsourcedprice, sourced) -> Iterable[PfLine]:
    """Take offtake, unsourced, sourced information. Do some data massaging and return
    3 PfLines: for offtake volume, unsourced price, and sourced price and volume."""

    # Make sure unsourced and offtake are specified.
    if offtakevolume is None or unsourcedprice is None:
        raise ValueError("Must specify offtake volume and unsourced prices.")

    # Offtake volume.
    if isinstance(offtakevolume, pd.Series) or isinstance(offtakevolume, pd.DataFrame):
        offtakevolume = PfLine(offtakevolume)  # using column names or series names
    if isinstance(offtakevolume, PfLine):
        if offtakevolume.kind == "p":
            raise ValueError("Must specify offtake volume.")
        elif offtakevolume.kind == "all":
            warnings.warn("Offtake also contains price infomation; this is discarded.")
            offtakevolume = offtakevolume.volume

    # Unsourced prices.
    if isinstance(unsourcedprice, pd.Series):
        if unsourcedprice.name in "qwr":
            ValueError("Name implies this is not a price timeseries.")
        elif unsourcedprice.name != "p":
            warnings.warn("Will assume prices, even though series name is not 'p'.")
            unsourcedprice.name = "p"
        unsourcedprice = PfLine(unsourcedprice)
    elif isinstance(unsourcedprice, pd.DataFrame):
        unsourcedprice = PfLine(unsourcedprice)  # using column names or series names

    if isinstance(unsourcedprice, PfLine):
        if unsourcedprice.kind == "q":
            raise ValueError("Must specify unsourced prices.")
        elif unsourcedprice.kind == "all":
            warnings.warn(
                "Unsourced also contains volume infomation; this is discarded."
            )
            unsourcedprice = unsourcedprice.price

    # Sourced volume and prices.
    if sourced is None:
        i = offtakevolume.index.union(unsourcedprice.index)  # largest possible index
        sourced = PfLine(pd.DataFrame({"q": 0, "r": 0}, i))

    # Do checks on indices. Lengths may differ, but frequency should be equal.
    indices = [
        obj.index for obj in (offtakevolume, unsourcedprice, sourced) if obj is not None
    ]
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")

    return offtakevolume, unsourcedprice, sourced


class PfState(PfStateTextOutput, PfStatePlotOutput, PfStateArithmatic):
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
    pnl_costs : PfLine ('all')
        The expected costs needed to source the offtake volume; the sum of the sourced 
        and unsourced positions.
        
    # index : pandas.DateTimeIndex
    #     Left timestamp of row.

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
        pu: pd.Series,
        qo: Optional[pd.Series],
        qs: Optional[pd.Series],
        rs: Optional[pd.Series],
        wo: Optional[pd.Series],
        ws: Optional[pd.Series],
        ps: Optional[pd.Series],
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
        if ws or qs or rs or ps:
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
        self._offtakevolume, self._unsourcedprice, self._sourced = _make_pflines(
            offtakevolume, unsourcedprice, sourced
        )

    @property
    def offtake(self) -> PfLine:
        return self._offtakevolume

    @property
    def sourced(self) -> PfLine:
        return self._sourced

    @property
    def unsourced(self) -> PfLine:
        return -(self._offtakevolume + self._sourced.volume) * self._unsourcedprice

    @property
    def unsourcedprice(self) -> PfLine:
        return self._unsourcedprice

    @property
    def netposition(self) -> PfLine:
        return -self.unsourced

    @property
    def pnl_cost(self):
        return self.sourced + self.unsourced

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

    def hedge_at_unsourcedprice(
        self, freq: str = "MS", how: str = "vol", bpo: bool = False
    ) -> PfState:
        """Hedge and source the unsourced volume, at unsourced prices in the portfolio,
        so that the portfolio is fully hedged.

        Parameters
        ----------
        freq : str, optional. By default "MS".
            Grouping frequency. One of {'D', 'MS', 'QS', 'AS'} for hedging at day, 
            month, quarter, or year level. ('D' not allowed for bpo==True.)
        how : {'vol' (default), 'val'}
            Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
        bpo : bool, optional. By default False.
            Set to True to split hedge into peak and offpeak values. (Only sensible
            for power portfolio with .freq=='H' or shorter, and a value for `freq` of
            'MS' or longer.)

        Returns
        -------
        PfState
            Which is fully hedge at time scales of `freq` or longer.
        """
        pass  # TODO

    def changefreq(self, freq: str = "MS") -> PfState:
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
        offtakevolume = self.offtake.changefreq(freq).volume
        unsourcedprice = self.unsourced.changefreq(freq).price  # important for wavg.
        sourced = self.sourced.changefreq(freq)
        return PfState(offtakevolume, unsourcedprice, sourced)

    # Dunder methods.

    def __getitem__(self, name):
        return getattr(self, name)

    # def __len__(self):
    #     return len(self.index)

    # def __bool__(self):
    #     return len(self) != 0

    def __eq__(self, other):
        if not isinstance(other, PfState):
            return False
        return all(
            [self[part] == other[part] for part in ["offtake", "unsourced", "sourced"]]
        )
