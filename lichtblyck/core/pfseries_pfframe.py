"""
Custom classes that are thin wrappers around the pandas objects.
"""

import pandas as pd
from . import attributes


FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def force_Pf(function):
    """Decorator to ensure a PfFrame (instead of a DataFrame) or a PfSeries (instead of a Series) is returned."""

    def wrapper(*args, **kwargs):
        val = function(*args, **kwargs)
        if type(val) is pd.DataFrame:
            val = PfFrame(val)
        elif type(val) is pd.Series:
            val = PfSeries(val)
        return val

    return wrapper


class PfFrame(pd.DataFrame):
    """
    PortfolioFrame; pandas dataframe with additional functionality for getting
    power [MW], price [Eur/MWh], quantity [MWh], revenue [Eur] and duration [h]
    timeseries.
    """

    # Time series.
    w = property(force_Pf(attributes._power))
    p = property(force_Pf(attributes._price))
    q = property(force_Pf(attributes._quantity))
    r = property(force_Pf(attributes._revenue))
    duration = property(attributes._duration)
    ts_right = property(attributes._ts_right)

    # Resample and aggregate.
    def changefreq(self, freq: str = "MS"):
        """
        Resample and aggregate the DataFrame at a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
            for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
            quarterhour; None to aggregate over the entire time period. The default is 'MS'.

        Returns
        -------
        PfFrame
            Same data at different timescale.
        """
        # By default, resampling labels are sometimes right-bound. Change to make left-bound.
        if freq == "M" or freq == "A" or freq == "Q":
            freq += "S"
        if freq is not None and freq not in FREQUENCIES:
            raise ValueError(
                "Parameter `freq` must be None or one of {"
                + ",".join(FREQUENCIES)
                + "}."
            )

        # Don't resample, just aggregate.
        if freq is None:
            duration = self.duration.sum()
            q = self.q.sum()
            r = self.r.sum()
            return pd.Series({"w": q / duration, "q": q, "p": r / q, "r": r})

        # Empty frame.
        if len(self) == 0:
            return PfFrame(self.resample(freq).mean())

        diff = FREQUENCIES.index(freq) - FREQUENCIES.index(self.index.freq)

        # Nothing more needed; dataframe already in desired frequency.
        if diff == 0:
            return self

        # Must downsample.
        elif diff < 0:
            pf = self.resample(freq).apply(aggpf)
            # Discard rows in new dataframe that are only partially present in original dataframe.
            mask1 = pf.index >= self.index[0]
            mask2 = pf.index + pf.index.freq <= self.index[-1] + self.index.freq
            pf = pf[mask1 & mask2]
            return PfFrame(pf)

        # Must upsample.
        else:
            # Keep only w and p because these are averages that can be copied over to each child row.
            pf = PfFrame({"w": self.w, "p": self.p}, self.index)
            # Workaround to avoid missing final values: first, add additional row...
            pf.loc[pf.index[-1] + pf.index.freq] = [None, None]
            pf = pf.resample(freq).ffill()  # ... then do upsampling ...
            pf = pf.iloc[:-1]  # ... and then remove final row.
            return PfFrame(pf)

        # TODO: change/customize the columns in the returned dataframe.

    @force_Pf  # If possible, return PfFrame or PfSeries instead of DataFrame or Series.
    def __getattr__(self, name):
        return super().__getattr__(name)

    # def __getitem__(self, name):
    # print ('getitem ' + name)
    # if name == 'r':
    # return self.r
    # return super().__getitem__(name)


class PfSeries(pd.Series):
    """
    PortfolioSeries; pandas series with additional functionality for getting
    duration [h] timeseries.
    """

    duration = property(attributes._duration)
    ts_right = property(attributes._ts_right)


def aggpf(pf: PfFrame) -> pd.Series:
    """
    Aggregation function for PfFrames.

    Parameters
    ----------
    pf : PfFrame
        Dataframe with (at least) 2 of the following columns: (w or q), p, r.

    Returns
    -------
    pd.Series
        The aggregated series with the aggregated values for w and p.
    """
    if not isinstance(pf, PfFrame):
        pf = PfFrame(pf)
    duration = pf.duration.sum()
    q = pf.q.sum()
    r = pf.r.sum()
    return pd.Series({"w": q / duration, "p": r / q})
