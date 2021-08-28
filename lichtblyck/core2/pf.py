"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios.
"""

from .pfline import PfLine
import pandas as pd


def _make_dict(qo=None, sourced=None, marketprices=None):
    """From data, create a dictionary with keys `offtake`, `sourced`, `marketprices`.
    Also, do some data verification."""
    # First, turn all into PfLines (or None).
    offtake = PfLine(offtake) if offtake is not None else None
    sourced = PfLine(sourced) if sourced is not None else None
    marketprices = PfLine(marketprices) if marketprices is not None else None

    # Do checks on indices. Lengths may differ, but frequency should be equal.
    indices = [obj.index for obj in (offtake, sourced, marketprices) if obj is not None]
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("Passed timeseries have unequal frequency; resample first.")

    # Offtake and market prices must be given.
    if offtake is None or marketprices is None:
        raise ValueError("At least `offtake` and `marketprices` must be given.")
    # Sourced volume is assumed 0 if not given.
    if sourced is None:
        i = offtake.index.union(marketprices.index)  # largest possible index
        sourced = PfLine(pd.DataFrame({"q": 0, "r": 0}, i))
    # Do checks on kind.
    if offtake.kind != "q":
        raise ValueError("For `offtake`, only volume information must be given.")
    if marketprices.kind != "p":
        raise ValueError("For `marketprices`, only price information must be given.")
    elif sourced.kind != "all":
        raise ValueError("For `sourced`, volume and price information must be given.")

    return {"offtake": offtake, "sourced": sourced, "marketprices": marketprices}


class Portfolio:
    """Class to hold timeseries information of an energy portfolio. 

    Parameters
    ----------
    offtake, sourced, marketprices : PfLine
        See below for expected characteristics.

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
        given timestamp (see 'Notes' below). Prices are the market prices.
    netposition : PfLine ('all')
        Net portfolio positions. Convenience property for users with a "traders' view"
        that does not follow sign conventions (see 'Notes' below); volumes are <0 if 
        portfolio is short and >0 if long. Identical to `.unsourced`, but with sign 
        change for volumes and revenues (but not prices).
    marketprices : PfLine ('p')
        Current market prices, e.g. a price-forward-curve (PFC).

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


    def __init__(
        self,offtake=None, sourced=None, marketprices=None
    ):
        data = _make_dict(offtake, sourced, marketprices)
        self._offtake = data["offtake"]
        self._sourced = data["sourced"]
        self._marketprices = data["marketprices"]

    @property
    def offtake(self) -> PfLine:
        return self._offtake

    @property
    def sourced(self) -> PfLine:
        return self._sourced

    @property
    def unsourced(self) -> PfLine:
        return PfLine(
            {"q": -self._offtake.q - self._sourced.q, "p": self._marketprices.p}
        )

    @property
    def marketprices(self) -> PfLine:
        return self._marketprices

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
            The part to plot. One of {'offtake' (default), 'sourced', 'unsourced', etc.}.
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

    # Methods that return new PfLine instance.

    set_offtake = lambda self, val: self._set_key_val("offtake", val)
    set_sourced = lambda self, val: self._set_key_val("sourced", val)
    set_marketprices = lambda self, val: self._set_key_val("marketprices", val)

    def _set_key_val(self, key, val) -> Portfolio:
        """Set or update a PfLine and return the modified Portfolio."""
        kwargs = {
            "offtake": self.offtake,
            "sourced": self.sourced,
            "marketprices": self.marketprices,
        }
        kwargs[key] = val
        return Portfolio(**kwargs)

    def changefreq(self, freq: str = "MS") -> Portfolio:
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
        offtake = self.offtake.changefreq(freq)
        sourced = self.sourced.changefreq(freq)
        marketprices = self.unsourced.changefreq(freq).p
        return Portfolio(offtake, sourced, marketprices)
