# -*- coding: utf-8 -*-
"""
Module to read price data from disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lichtblyck as lb
from sqlalchemy import create_engine
from typing import Tuple, Union, Iterable
import eikon

# eikon.set_app_key('e3993065d33b4e65b742130b0aa04528e90e1338')

SPOTPRICEFILE = "lichtblyck/prices/sourcedata/prices_mid2000_mid2020.xlsx"
FRONTMONTHPRICEFILE = "lichtblyck/prices/sourcedata/prices_mid2000_mid2020.xlsx"

# From Belvis
# SPOTPRICEFILE = 'lichtblyck/prices/sourcedata/spot.tsv'
# def spot() -> pd.Series:
#     """Return spot price timeseries."""
#     data = pd.read_csv(SPOTPRICEFILE, header=None, sep='\t',
#                        names=['date', 'time', 'price', 'anmerkungen', 'empty'])
#     data['ts_right'] = pd.to_datetime(data["date"] + " " + data["time"], format="%d.%m.%Y %H:%M:%S")
#     #Replace missing values with NaN, convert others to float:
#     data['p_spot'] = data['price'].apply(lambda x: np.NaN if x == '---' else float(x.replace(',', '.')))
#     spot = tools.set_ts_index(data, 'ts_right', 'right')
#     spot = spot.resample('H').asfreq()
#     return spot['p_spot']


def spot() -> pd.Series:
    """Return spot price timeseries."""
    data = pd.read_excel(
        SPOTPRICEFILE, sheet_name="spot_static", header=0, usecols="B:AC", index_col=-1
    )
    data = data.drop(data.columns[-2:], axis=1)
    dls_values = data[
        "DLS"
    ].dropna()  # prices for second 02:00-03:00 hour on last Sun in Oct.
    dls_values.index += pd.Timedelta(hours=2)
    data = data.drop("DLS", axis=1)
    data.columns = pd.to_timedelta([c[:5] + ":00" for c in data.columns])
    spot = data.stack()
    spot.index = spot.index.get_level_values(0) + spot.index.get_level_values(1)
    # Add repeated hour on last Sun in Oct.
    for ts in [
        i + pd.offsets.LastWeekOfMonth(weekday=6) + pd.Timedelta(hours=2)
        for i in pd.date_range(spot.index[0], spot.index[-1], freq="MS")
        if i.month == 10
    ]:
        try:
            to_insert = pd.Series(dls_values[ts], [ts])
        except KeyError:
            to_insert = pd.Series(spot[ts], [ts])
        spot = pd.concat([spot[:ts], to_insert, spot[ts:][1:]])
    spot = lb.set_ts_index(spot).rename("p_spot")
    spot = spot.resample("H").asfreq()
    return spot


def futures(
    prod: str = "y", earliest_trading=None, earliest_delivery=None
) -> pd.DataFrame:
    """
    Return futures prices timeseries.

    Arguments:
        prod: product to return the prices of. One of {'y', 'q', 'm'}.
        earliest_trading: only return prices after this date. (default: all)
        earliest_delivery: only return prices for products whose delivery is
            on or after this date. (default: all)

    Returns:
        Dataframe with EEX futures prices. Index: Multiindex (with levels 0:
            product {'y', 'q', 'm'}, 1: time stamp of delivery start, 2: date
            of trading). Columns: p_base (base price), p_peak (peak price),
            p_offpeak (offpeak price). ts_right_deliv (time stamp of delivery
            end), trade_before_deliv (timedelta between trade and delivery
            start), basehours, peakhours, offpeakhours in delivery period.
    """
    engine = create_engine("mssql+pymssql://sqlbiprod/__Staging")

    try:
        dur_months = {"m": 1, "q": 3, "y": 12}[prod]
    except KeyError:
        raise ValueError("Argument 'prod' must be in {'y', 'q', 'm'}.")

    # Get values from server...
    where = f"WHERE Typ='{prod.upper()}'"
    if earliest_trading:
        where += f" AND EEX_Handelstag > '{earliest_trading}'"
    if earliest_delivery:
        where += f" AND Lieferzeitraum > '{earliest_delivery}'"
    df = pd.read_sql_query(
        f"SELECT * FROM dbo.Fakten_EW_EEX_Preise_fkunden {where}", engine
    )
    breakpoint()
    # ...massage to get correct format...
    df = df.rename(
        columns={
            "Typ": "deliv_prod",
            "Peak_Preis": "p_peak",
            "Base_Preis": "p_base",
            "EEX_Handelstag": "ts_left_trade",
            "Lieferzeitraum": "ts_left_deliv",
        }
    )
    df = df.drop("ID", axis=1)
    df["ts_left_trade"] = pd.to_datetime(df["ts_left_trade"]).dt.tz_localize(
        "Europe/Berlin"
    )
    df["ts_left_deliv"] = pd.to_datetime(df["ts_left_deliv"]).dt.tz_localize(
        "Europe/Berlin"
    )
    df["ts_right_deliv"] = df["ts_left_deliv"] + pd.offsets.DateOffset(
        months=dur_months
    )
    df["trade_before_deliv"] = df["ts_left_deliv"] - df["ts_left_trade"]
    # ...get peak and base and offpeak hours...
    h = df[["ts_left_deliv", "ts_right_deliv"]].drop_duplicates()
    bpo = hours_bpo(h["ts_left_deliv"], h["ts_right_deliv"])
    h["basehours"], h["peakhours"], h["offpeakhours"] = bpo
    df = pd.merge(df, h, on=["ts_left_deliv", "ts_right_deliv"])
    # ...and use to calculate offpeak prices.
    df["p_offpeak"] = p_offpeak(
        df["p_base"], df["p_peak"], df["basehours"], df["peakhours"]
    )
    # Finally, return in correct row and column order.
    # TODO: add frequency to index, if possible
    df = df.set_index(["deliv_prod", "ts_left_deliv", "ts_left_trade"]).sort_index()
    return df[
        [
            "ts_right_deliv",
            "trade_before_deliv",
            "p_base",
            "p_peak",
            "p_offpeak",
            "basehours",
            "peakhours",
            "offpeakhours",
        ]
    ]


def frontmonth():
    """
    Return futures prices timeseries of frontmonth (M1).

    Returns:
        Dataframe with EEX futures prices. Index: trading day. Columns:
            p_base (base price), p_peak (peak price), p_offpeak (offpeak
            price). ts_left_deliv, ts_right_deliv (timestamps of delivery),
            trade_before_deliv (timedelta between trade and delivery
            start), basehours, peakhours, offpeakhours in delivery period.
    """
    # Get values...
    b = pd.read_excel(
        FRONTMONTHPRICEFILE,
        sheet_name="futures_static",
        usecols=[4, 8, 9],
        header=0,
        names=["p_base", "ts_left_trade", "ts_left_deliv"],
    )
    p = pd.read_excel(
        FRONTMONTHPRICEFILE,
        sheet_name="futures_static",
        usecols=[16, 20, 21],
        header=0,
        names=["p_peak", "ts_left_trade", "ts_left_deliv"],
    )
    df = p.merge(b, how="inner", on=["ts_left_trade", "ts_left_deliv"])
    df["ts_left_trade"] = df["ts_left_trade"].dt.tz_localize("Europe/Berlin")
    df["ts_left_deliv"] = df["ts_left_deliv"].dt.tz_localize("Europe/Berlin")
    df["ts_right_deliv"] = df["ts_left_deliv"] + pd.offsets.DateOffset(months=1)
    df["trade_before_deliv"] = df["ts_left_deliv"] - df["ts_left_trade"]
    # ...get peak and base and offpeak hours...
    h = df[["ts_left_deliv", "ts_right_deliv"]].drop_duplicates().reset_index(drop=True)
    bpo = hours_bpo(h["ts_left_deliv"], h["ts_right_deliv"])
    h["basehours"], h["peakhours"], h["offpeakhours"] = bpo
    df = df.merge(h, how="left", on=["ts_left_deliv", "ts_right_deliv"])
    # ...and use to calculate offpeak prices.
    df["p_offpeak"] = p_offpeak(
        df["p_base"], df["p_peak"], df["basehours"], df["peakhours"]
    )
    # Finally, return in correct row and column order.
    df = df.set_index("ts_left_trade").sort_index()
    return df[
        [
            "ts_left_deliv",
            "ts_right_deliv",
            "trade_before_deliv",
            "p_base",
            "p_peak",
            "p_offpeak",
            "basehours",
            "peakhours",
            "offpeakhours",
        ]
    ]


def hours_bpo(ts_left, ts_right) -> float:
    """Return number of base, peak and offpeak hours in interval [ts_left, ts_right).
    Timestamps must coincide with day start."""
    if isinstance(ts_left, Iterable):
        return np.vectorize(hours_bpo)(ts_left, ts_right)
    days = pd.date_range(ts_left, ts_right, freq="D", closed="left")
    base = (
        ts_right - ts_left
    ).total_seconds() / 3600  # to correctly deal with summer-/wintertime changeover
    fullweeks, individual = divmod(len(days), 7)
    peak = fullweeks * 60.0
    for day in days[:individual]:
        if day.isoweekday() < 6:
            peak += 12
    return base, peak, base - peak


def p_offpeak(p_base, p_peak, basehours, peakhours) -> float:
    """Return offpeak price from base and peak price and number of base and peak hours in period."""
    if isinstance(p_base, Iterable):
        return np.vectorize(p_offpeak)(p_base, p_peak, basehours, peakhours)
    return (p_base * basehours - p_peak * peakhours) / (basehours - peakhours)


def is_peak_hour(ts) -> bool:
    """Return True if timestamp 'ts' is a peak hour. More precisely: if 'ts'
    lies in one of the (left-closed) time intervals that define the peak hour
    periods."""
    if isinstance(ts, Iterable):
        return np.vectorize(is_peak_hour)(ts)
    return ts.hour >= 8 and ts.hour < 20 and ts.isoweekday() < 6


def _p_bpo(s: pd.Series) -> pd.Series:
    """
    Aggregate price series to base, peak and offpeak prices.

    Arguments:
        s: price timeseries with (left-bound) DatetimeIndex.

    Returns:
        Series with prices. Index: p_base, p_peak, p_offpeak.
    """
    grouped = s.groupby(is_peak_hour).mean()
    return pd.Series(
        {"p_base": s.mean(), "p_peak": grouped[True], "p_offpeak": grouped[False]}
    )


def p_bpo(s: pd.Series, freq: str = None) -> pd.DataFrame:
    """
    Aggregate price series to base, peak and offpeak prices. Grouped by time
    interval (if specified).

    Arguments:
        s: price timeseries with (left-bound) DatetimeIndex.
        freq: grouping frequency. One of {'MS', 'QS', 'YS'} for month, quarter,
            or year prices. None (default) for no grouping.

    Returns:
        If grouped: Dataframe with base, peak and offpeak prices (as columns).
            Index: time stamp of delivery start. Columns: p_base, p_peak,
            p_offpeak.
        If not grouped: Series with base, peak and offpeak prices (as index).
            Index: p_base, p_peak, p_offpeak.
    """
    if freq is None:
        return _p_bpo(s)
    if freq not in ("MS", "QS", "YS"):
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'YS'}.")
    return s.groupby(pd.Grouper(freq=freq)).apply(_p_bpo).unstack()


def p_bpo_long(s: pd.Series, freq: str = None) -> pd.Series:
    """
    Transform price series to peak and offpeak prices, by calculating the mean
    for each.

    Arguments:
        s: price timeseries with (left-bound) DatetimeIndex.
        freq: grouping frequency. One of {'MS', 'QS', 'YS'} for month, quarter,
            or year prices. None (default) for uniform peak and offpeak values
            throughout timeseries.

    Returns:
        Price timeseries where each peak hour within the provided frequency
        has the same value. Idem for offpeak hours. Index: as original series.
        Values: mean value for time period.
    """
    if freq is None:
        group_function = lambda ts: is_peak_hour(ts)
    elif freq == "MS":
        group_function = lambda ts: (ts.year, ts.month, is_peak_hour(ts))
    elif freq == "QS":
        group_function = lambda ts: (ts.year, ts.quarter, is_peak_hour(ts))
    elif freq == "YS":
        group_function = lambda ts: (ts.year, is_peak_hour(ts))
    else:
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'YS'}.")
    return s.groupby(group_function).transform(np.mean)


def _w_hedge(w: Union[pd.Series, Iterable], p: Union[pd.Series, Iterable]) -> pd.Series:
    """
    Make value hedge of power timeseries, for given price timeseries.

    Arguments:
        w: power timeseries.
        p: price timeseries.

    Returns:
        Series with powers. Index: w_peak, w_offpeak.
    """
    grouped = (
        pd.DataFrame({"w": w, "p": p})
        .groupby(is_peak_hour)
        .apply(lambda df: lb.wavg(df["w"], df["p"]))
    )
    return pd.Series({"w_peak": grouped[True], "w_offpeak": grouped[False]})


def w_hedge(
    w: Union[pd.Series, Iterable], p: Union[pd.Series, Iterable], freq: str = None
) -> pd.Series:
    """
    Make value hedge of power timeseries, for given price timeseries.

    Arguments:
        w: power timeseries.
        p: price timeseries.
        freq: grouping frequency. One of {'MS', 'QS', 'YS'} for month, quarter,
            or year prices. None (default) for uniform peak and offpeak values
            throughout timeseries.

    Returns:
        If grouped: Dataframe with peak and offpeak powers (as columns).
            Index: time stamp of delivery start. Columns: w_peak,
            w_offpeak.
        If not grouped: Series with peak and offpeak powers (as index).
            Index: w_peak, w_offpeak.
    """
    if freq is None:
        return _w_hedge(w, p)
    if freq not in ("MS", "QS", "YS"):
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'YS'}.")
    return (
        pd.DataFrame({"w": w, "p": p})
        .groupby(pd.Grouper(freq=freq))
        .apply(lambda df: _w_hedge(df["w"], df["p"]))
    )


def w_hedge_long(
    w: Union[pd.Series, Iterable], p: Union[pd.Series, Iterable], freq: str = None
) -> pd.Series:
    """
    Transform power and price timeseries into new power timeseries with value
        hedge.

    Arguments:
        w: power timeseries.
        p: price timeseries.
        freq: grouping frequency. One of {'MS', 'QS', 'YS'} for month, quarter,
            or year prices. None (default) for uniform peak and offpeak values
            throughout timeseries.

    Returns:
        Power timeseries where each peak hour within the provided frequency
        has the same value. Idem for offpeak hours. Index: as original series.
        Values: power that has same monetary value as w in each time period.
    """
    if freq is None:
        group_function = lambda ts: is_peak_hour(ts)
    elif freq == "MS":
        group_function = lambda ts: (ts.year, ts.month, is_peak_hour(ts))
    elif freq == "QS":
        group_function = lambda ts: (ts.year, ts.quarter, is_peak_hour(ts))
    elif freq == "YS":
        group_function = lambda ts: (ts.year, is_peak_hour(ts))
    else:
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'YS'}.")
    apply_function = lambda df: pd.Series(
        [lb.wavg(df["w"], df["p"])] * len(df), df.index
    )
    s = (
        pd.DataFrame({"w": w, "p": p})
        .groupby(group_function)
        .apply(apply_function)
        .droplevel(0)
        .rename("w_hedge")
    )
    if w.index.freq is None:
        return s.sort_index()
    else:
        return s.resample(w.index.freq).asfreq()


def vola(
    df: Union[pd.Series, pd.DataFrame], window: int = 100
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate volatility in [fraction/year] from price time series/dataframe.

    Arguments:
        df: Series or Dataframe with price values indexed by (trading day) timestamps.
        window: number of observations for volatility estimate.

    Returns:
        Series or Dataframe with volatility calculated with rolling window.
    """
    df = df.apply(np.log).diff()  # change as fraction
    volas = df.rolling(window).std()  # volatility per average timedelta
    av_years = df.rolling(window).apply(
        lambda s: ((s.index[-1] - s.index[0]) / window).total_seconds()
        / 3600
        / 24
        / 365.24
    )
    volas /= av_years.apply(np.sqrt)  # volatility per year
    return volas.dropna()
