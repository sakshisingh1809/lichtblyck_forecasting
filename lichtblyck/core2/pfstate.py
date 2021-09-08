"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios, at a 
certain moment in time (e.g., at the current moment, without any historic data).
"""

from __future__ import annotations
from .pfline import PfLine
from typing import Optional, Iterable, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


def _make_sourced(qs: Optional[pd.Series], rs: Optional[pd.Series]) -> Optional[PfLine]:
    """Turn sourced volume `qs` and revenue (cost) `rs` into a portfolio frame."""
    if qs is None and rs is None:
        return None
    elif qs is not None and rs is not None:
        return PfLine(pd.DataFrame({"q": qs, "r": rs}))
    else:
        raise ValueError("Must specifty both sourced volume and revenue, or none.")


def _make_pflines(offtakevolume, unsourcedprice, sourced) -> Iterable[PfLine]:
    """Take offtake, unsourced, sourced information. Do some data massaging and return
    3 PfLines: for offtake volume, unsourced price, and sourced price and volume."""

    # Make sure unsourced and offtake are specified.
    if not offtakevolume or not unsourcedprice:
        raise ValueError("Must specify offtake volume and unsourced prices.")

    # Offtake volume.
    if isinstance(offtakevolume, Union[pd.Series, pd.DataFrame]):
        offtakevolume = PfLine(offtakevolume)  # using column names or series names
    if isinstance(offtakevolume, PfLine):
        if offtakevolume.kind == "p":
            raise ValueError("Must specify offtake volume.")
        elif offtakevolume.kind == "all":
            warnings.warn("offtake also contains price infomation; this is discarded.")
            offtakevolume = offtakevolume.volume

    # Unsourced prices.
    if isinstance(unsourcedprice, Union[pd.Series, pd.DataFrame]):
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
    if not sourced:
        i = offtakevolume.index.union(unsourcedprice.index)  # largest possible index
        sourced = PfLine(pd.DataFrame({"q": 0, "r": 0}, i))

    # Do checks on indices. Lengths may differ, but frequency should be equal.
    indices = [
        obj.index for obj in (offtakevolume, unsourcedprice, sourced) if obj is not None
    ]
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")

    return offtakevolume, unsourcedprice, sourced


class PfState:
    """Class to hold timeseries information of an energy portfolio, at a specific moment. 

    Parameters
    ----------
    offtakevolume, unsourcedprices, sourced : PfLine
        Offtake volume may also be passed as pd.Series with name `q` or `w`. Unsourced
        prices may also be passed as pd.Series. 

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
    # ts_right, duration : pandas.Series
    #     Right timestamp, and duration [h] of row.

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
        ps: Optional[pd.Series]
    ):
        """Create Portfolio instance from pd.Series instances for offtake volume (`qo`
        [MWh]), unsourced prices (`pu` [Eur/MWh]), sourced volume (`qs` [MWh]) and 
        sourced revenue (`rs` [Eur])."""
        if ws is None 
        ws, qs, rs, ps =
        return cls(PfLine({"q": qo, "w": wo}), PfLine({"p": pu}), _make_sourced(qs, rs))

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
    def pnl_costs(self):
        return self.sourced + self.unsourced

    # Methods.

    def plot_to_ax(
        self, ax: plt.Axes, part: str = "offtake", col: str = None, **kwargs
    ):
        """Plot a timeseries of the Portfolio to a specific axes.
        
        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        part : str, optional
            The part to plot. One of {'offtake' (default), 'sourced', 'unsourced', 
            'netposition', 'pnl_costs'}.
        col : str, optional
            The column to plot. Default: plot volume `w` [MW] (if available) or else
            price `p` [Eur/MWh].
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        pass  # TODO

    def plot(self, cols: str = "wp") -> plt.Figure:
        """Plot one or more timeseries of the Portfolio.
        
        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume `w` [MW] and price `p` [Eur/MWh] 
            (if available).

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        pass  # TODO

    # Methods that return new Portfolio instance.

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

    def changefreq(self, freq: str = "MS") -> PfState:
        """Resample the Portfolio to a new frequency.
        
        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS' 
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.
        
        Returns
        -------
        Portfolio
            Resampled at wanted frequency.
        """
        # pu resampling is most important, so that prices are correctly weighted.
        offtakevolume = self.offtake.changefreq(freq).volume
        unsourcedprice = self.unsourced.changefreq(freq).price  # important for wavg.
        sourced = self.sourced.changefreq(freq)
        return PfState(offtakevolume, unsourcedprice, sourced)