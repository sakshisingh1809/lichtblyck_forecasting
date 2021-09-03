"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios, at a 
certain moment in time (e.g., at the current moment, without any historic data).
"""

from .pfline import PfLine
from typing import Optional
import pandas as pd
import numpy as np
import warnings

# Complete set of PfLines:
# offtake has volume. Price is not regarded.
# sourced has volume and price.
# unsourced has volume and price.

def _validate_consistency(
    offtake=Optional[PfLine], sourced=Optional[PfLine], unsourced=Optional[PfLine]
):
    if not offtake or not sourced or not unsourced:
        return
    if not all(['q' in obj.available for obj in [offtake, sourced, unsourced]]):
        return
    if (abs((offtake.volume + sourced.volume + unsourced.volume).q) < 0.001).all():
       return
    raise ValueError('Offtake, sourced and unsourced volumes not consistent.')

def _calculate_and_validate_offtake(
    offtake=Optional[PfLine], sourced=Optional[PfLine], unsourced=Optional[PfLine]
) -> PfLine:
    """Check characteristics of PfLine representing offtake. If values are missing, cal-
    culate from sourced and unsourced PfLines, if possible."""
    if not offtake:
        offtake = -(sourced.volume + unsourced.volume)
    if offtake.kind not in ['q', 'all']:
        raise ValueError("Offtake volume not known.")
    if not (offtake.q <= 0.001).all():
        raise ValueError("Offtake volume must be <= 0 for all timestamps.")
    return offtake

def _calculate_and_validate_sourced(
    offtake=Optional[PfLine], sourced=Optional[PfLine], unsourced=Optional[PfLine]
) -> PfLine:
    """Check characteristics of PfLine representing sourced volume. If values are
    missing, calculate from offtake and unsourced PfLines, if possible."""
    if not sourced:
        sourced = -(offtake.volume + unsourced.volume)
    if sourced.kind == 'q' and (abs(sourced.q) < 0.001).all():
        sourced = sourced.set_r(0) # assume that revenue is 0 if volume is 0.
    if sourced.kind != 'all':
        raise ValueError("Sourced volume and/or price not known.")
    if not (sourced.q >= -0.001).all():
        raise ValueError("Sourced volume must be >= 0 for all timestamps.")
    return sourced

def _calculate_and_validate_unsourced(
    offtake=Optional[PfLine], sourced=Optional[PfLine], unsourced=Optional[PfLine]
) -> PfLine:
    """Check characteristics of PfLine representing unsourced volume. If values are
    missing, calculate from offtake and sourced PfLines, if possible."""
    if not unsourced:
        unsourced = -(offtake.volume + sourced.volume)
    if unsourced.kind == 'q' and (abs(sourced.q) < 0.001).all():
        unsourced = unsourced.set_r(0)
    if unsourced.kind not in ['p', 'all']:
        raise ValueError("Cannot calculate revenue/price of unsourced volume.")
    return unsourced


def _from_series(qo, pu, qs, rs) -> Iterable[pd.Series]:
    """Take offtake volume `qo`, market prices `pu` (= price of unsourced volume), 
    sourced volume `qs`, sourced revenue `rs` (= cost of sourced volume). Do some 
    data verification and return same 4 pd.Series."""

    # First, turn all into PfLines.
    qo_pfl = PfLine({"q": qo})
    pu_pfl = PfLine({"p": pu})
    if qs is not None and rs is not None:
        s_pfl = PfLine(pd.DataFrame({"q": qs, "r": rs}))
    elif qs is None and rs is None:  # Sourced volume assumed 0 if not passed.
        i = qo_pfl.index.union(pu_pfl.index)  # largest possible index
        s_pfl = PfLine(pd.DataFrame({"q": 0, "r": 0}, i))
    else:
        raise ValueError("Must specifty both sourced volume and revenue, or none.")

    # Do checks on indices. Lengths may differ, but frequency should be equal.
    indices = [obj.index for obj in (qo_pfl, pu_pfl, s_pfl) if obj is not None]
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("Passed timeseries have unequal frequency; resample first.")

    return qo_pfl.q, pu_pfl.p, s_pfl.q, s_pfl.r


def _from_pflines(offtake, unsourced, sourced) -> Iterable[pd.Series]:
    """Take offtake, unsourced, sourced PfLine instances. Do some date verification and
    return 4 pd.Series (qo, pu, qs, rs)."""

    # If unsourced is not specified, we have no way of finding the prices.
    if not unsourced:
        if offtake and sourced and offtake.volume != -sourced.volume:
            raise ValueError("Unsourced volume, but no prices.")

    # If all are specified, we need to check consistency.
    if offtake and unsourced and sourced:
        if "q" in offtake.available and "q" in unsourced.available:
            pass

    # Sourced.
    if sourced is None:
        qs = rs = 0
    else:
        qs, rs = sourced.q, sourced.r

    # Offtake volume.
    if offtake is not None:
        if offtake.kind != "q":
            raise ValueError("Offtake should only contain volume.")
        qo = offtake.q
        if unsourced.kind == "all" and not np.allclose(qo, -unsourced.q - qs):
            raise ValueError("Inconsistent information for offtake volume.")
    elif unsourced.kind == "all":
        qo = -unsourced.q - qs
    else:
        raise ValueError("No offtake volume information.")

    # Unsourced prices.
    if unsourced.kind in ["p", "all"]:
        pu = unsourced.p
    else:
        raise ValueError("No price information for the unsourced volume.")

    return qo, pu, qs, rs


class PfState:
    """Class to hold timeseries information of an energy portfolio, at a specific moment. 

    Parameters
    ----------
    qo, pu, qs (optional), rs (optional) : pd.Series
        Offtake volume `qo`, price of unsourced volume `pu` (= market price), 
        sourced volume `qs`, sourced revenue `rs` (= cost of sourced volume).

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
        given timestamp (see 'Notes' below).
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
    def from_pflines(
        cls,
        offtake: Optional[PfLine],
        unsourced: Optional[PfLine],
        sourced: Optional[PfLine],
    ):
        """Create Portfolio instance from several PfLine instances (all optional)."""
        return cls(*_from_pflines(offtake, unsourced, sourced))

    def __init__(
        self, qo: pd.Series, pu: pd.Series, qs: pd.Series = None, rs: pd.Series = None,
    ):
        # The only internal data of this class.
        self._qo, self._pu, self._qs, self._rs = _from_series(qo, pu, qs, rs)

    @property
    def offtake(self) -> PfLine:
        return PfLine({"q": self._qo})

    @property
    def sourced(self) -> PfLine:
        return PfLine({"q": self._qs, "r": self._rs})

    @property
    def unsourced(self) -> PfLine:
        return PfLine({"q": -(self._qo + self._qs), "p": self._pu})

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

    def set_offtakevolume(self, qo: pd.Series) -> PfState:
        warnings.warn(
            "This changes the unsourced volume and causes inaccuracies in its price, if the portfolio has a frequency that is longer than the spot market."
        )
        return PfState(qo, self._pu, self._qs, self._rs)

    def set_unsourcedprices(self, pu: pd.Series) -> PfState:
        return PfState(self._qo, pu, self._qs, self._rs)

    def set_sourced(self, qs: pd.Series, rs: pd.Series) -> PfState:
        warnings.warn(
            "This changes the unsourced volume and causes inaccuracies in its price, if the portfolio has a frequency that is longer than the spot market."
        )
        return PfState(self._qo, self._pu, qs, rs)

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
        qo = self.offtake.changefreq(freq).q
        pu = self.unsourced.changefreq(freq).p
        sourced = self.sourced.changefreq(freq)
        qs, rs = sourced.q, sourced.r
        return PfState(qo, pu, qs, rs)
