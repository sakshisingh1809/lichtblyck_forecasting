"""
Module to read montel price data from disk.
"""

from . import utils
from .convert import offpeak
from ..tools.frames import set_ts_index
from typing import Dict
from pathlib import Path
import functools
import pandas as pd
import numpy as np


_MONTELFILEPATH = Path(__file__).parent / "sourcedata" / "prices_montel.xlsm"


def _excel_gas(
    period_type: str = "m", period_start: int = 1, market_code: str = "ncg"
) -> Dict:
    """
    Get the location of gas data in the montel excel file.

    Parameters
    ----------
    period_type : {'d' (day), 'm' (month, default), 'q' (quarter), 's' (season), 'a' (year)}
        Duration of the product.
    period_start : int
        1 = next/coming (full) period (default), 2 = period after that, etc.
        Ignored for period_type == 'd'.
    market_code : {'ncg' (netconnect-germany, default), 'gpl' (gaspool)}

    Example
    -------
    period_type, period_start, market_code == ('q', 1, 'gpl') to get prices for
    front-quarter product in the gpl market area.
    """
    period_type = period_type.lower()[0]
    kwargs = {"io": _MONTELFILEPATH, "header": 0}
    market_code = market_code.lower()

    if period_type == "d":  # day prices from day-ahead market
        if market_code == "gpl":
            startcol = 5
        elif market_code == "ncg":
            startcol = 15
        else:
            raise ValueError("Invalid value for parameter `market_code`.")
        kwargs.update(
            {"sheet_name": "gas_spot_stat", "usecols": startcol + np.array([1, 2])}
        )
    else:
        if period_type == "m":
            sheet_name = "gas_M_stat"
            max_anticipation = 6
        elif period_type == "q":
            sheet_name = "gas_Q_stat"
            max_anticipation = 7
        elif period_type == "s":
            sheet_name = "gas_S_stat"
            max_anticipation = 4
        elif period_type == "a":
            sheet_name = "gas_A_stat"
            max_anticipation = 6
        else:
            raise ValueError("Invalid value for parameter `period_type`.")

        if period_start > max_anticipation:
            raise NotImplementedError(
                f"Currently, only frontperiod (1) to ({max_anticipation}) (inclusive) are implemented."
            )

        if market_code == "gpl":
            startcol = 0
        elif market_code == "ncg":
            startcol = 6
        else:
            raise ValueError("Invalid value for parameter `market_code`.")
        startcol += (period_start - 1) * 12

        kwargs.update(
            {"sheet_name": sheet_name, "usecols": startcol + np.array([1, 2])}
        )
    return kwargs


def _excel_power(
    period_type: str = "M", period_start: int = 1, product_code: str = "base"
) -> Dict:
    """
    Get the location of power data in the montel excel file.

    Parameters
    ----------
    period_type : {'h' (hourly, from day-ahead auction), 'm' (month, default), 'q' (quarter), 'a' (year)}
        Duration of the product.
    period_start : int
        1 = next/coming (full) period (default), 2 = period after that, etc.
        Ignored for period_type == 'h'.
    product_code : {'base' (default), 'peak'}
        Ignored for period_type == 'h'.

    Example
    -------
    period_type, period_start, product_code == ('q', 1, 'peak') to get prices for
    peak band delivery in the next quarter.
    """
    period_type = period_type.lower()[0]
    kwargs = {"io": _MONTELFILEPATH, "header": 0}
    product_code = product_code.lower()

    if period_type == "h":  # hourly prices from day-ahead market
        kwargs.update({"sheet_name": "pwr_spot_stat", "usecols": "B:AC"})
    else:
        if period_type == "m":
            sheet_name = "pwr_M_stat"
            max_anticipation = 9
        elif period_type == "q":
            sheet_name = "pwr_Q_stat"
            max_anticipation = 10
        elif period_type == "a":
            sheet_name = "pwr_A_stat"
            max_anticipation = 6
        else:
            raise ValueError("Invalid value for parameter `period_type`.")

        if period_start > max_anticipation:
            raise NotImplementedError(
                f"Currently, only frontperiod (1) to ({max_anticipation}) (inclusive) are implemented."
            )

        if product_code == "base":
            startcol = 0
        elif product_code == "peak":
            startcol = 6
        else:
            raise ValueError("Invalid value for parameter `product_code`.")
        startcol += (period_start - 1) * 12

        kwargs.update(
            {"sheet_name": sheet_name, "usecols": startcol + np.array([1, 2])}
        )
    return kwargs


def power_spot() -> pd.Series:
    """
    Power spot price timeseries with hourly prices.

    Parameters
    ----------
    None

    Returns
    -------
    pd.Series
    """
    data = pd.read_excel(**_excel_power("h"))
    data = data.set_index("Date").drop(["Base", "Peak (09-20)"], axis=1)
    dls_values = data["DLS"].dropna()  # prices for second 02:00-03:00 hour last Oct Sun
    dls_values.index += pd.Timedelta(hours=2)
    data = data.drop("DLS", axis=1)
    data.columns = pd.to_timedelta([c[:5] + ":00" for c in data.columns])
    spot = data.stack()
    spot.index = spot.index.get_level_values(0) + spot.index.get_level_values(1)
    # Add repeated hour on last Sun in Oct.
    st2wt_changeover = [
        i + pd.offsets.LastWeekOfMonth(weekday=6) + pd.Timedelta(hours=2)
        for i in pd.date_range(spot.index[0], spot.index[-1], freq="MS")
        if i.month == 10
    ]
    for ts in st2wt_changeover:
        try:
            to_insert = pd.Series(dls_values[ts], [ts])
        except KeyError:
            to_insert = pd.Series(spot[ts], [ts])
        spot = pd.concat([spot[:ts], to_insert, spot[ts:][1:]])
    spot = set_ts_index(spot).rename("p")
    spot = spot.resample("H").asfreq()
    return spot


def gas_spot(market_code: str = "ncg") -> pd.Series:
    """
    Spot price timeseries with daily prices.

    Parameters
    ----------
    market_code : {'ncg' (netconnect-germany, default), 'gpl' (gaspool)}

    Returns
    -------
    pd.Series
    """
    data = pd.read_excel(**_excel_gas("da", market_code=market_code))
    data = set_ts_index(data.dropna(), data.columns[0], continuous=False)
    s = data.iloc[:, 0]  # turn one-column df into series
    s.index = s.index.ts_right  # shift up one, so delivery (not trade) day is shown.
    spot = set_ts_index(s).rename("p")
    return spot


def _power_futures(period_type: str = "m", period_start: int = 1) -> pd.DataFrame:
    """
    Get power futures prices.

    Parameters
    ----------
    period_type : {'m' (month, default), 'q' (quarter), 'a' (year)}
        Duration of the product.
    period_start : int
        1 = next/coming (full) period (ie., "frontyear", "frontmonth" etc) (default),
        2 = period after that, etc.

    Returns
    -------
    pd.DataFrame
        with power futures prices. Index: trading day. Columns: p_base, p_peak,
        p_offpeak (prices), ts_left, ts_right (timestamps of delivery),
        anticipation (timedelta between trade and delivery start), basehours,
        peakhours, offpeakhours in delivery period.
    """
    # Get values...
    b = pd.read_excel(**_excel_power(period_type, period_start, "base"))
    p = pd.read_excel(**_excel_power(period_type, period_start, "peak"))
    for df in (b, p):
        df.dropna(inplace=True)
        df.columns = ["ts_left_trade", "p"]
    b = set_ts_index(b, "ts_left_trade", continuous=False)
    p = set_ts_index(p, "ts_left_trade", continuous=False)
    # ...put into one object...
    df = p.merge(
        b,
        how="inner",
        left_index=True,
        right_index=True,
        suffixes=("_peak", "_base"),
    )
    # ...add some additional information...
    @functools.lru_cache
    def deliv_f(ts):
        return utils.ts_leftright(ts, period_type, period_start)

    df["ts_left"], df["ts_right"] = zip(*df.index.map(deliv_f))
    df["anticipation"] = df["ts_left"] - df.index
    # ...get number of peak and base and offpeak hours...
    # h = df[["ts_left", "ts_right"]].drop_duplicates().reset_index(drop=True)
    # bpo = np.vectorize(duration_bpo)(h["ts_left"], h["ts_right"])
    # h["basehours"], h["peakhours"], h["offpeakhours"] = bpo
    # df = df.reset_index().merge(h, how="left").set_index(df.index.names)
    # ...and use to calculate offpeak prices.
    df["p_offpeak"] = df.apply(
        lambda row: offpeak(
            row["p_base"], row["p_peak"], row["ts_left"], row["ts_right"]
        ),
        axis=1,
    )
    # Finally, return in correct row and column order.
    df.index = df.index.rename("ts_left_trade")
    return df[
        [
            "ts_left",
            "ts_right",
            "anticipation",
            "p_base",
            "p_peak",
            "p_offpeak",
            # "basehours",
            # "peakhours",
            # "offpeakhours",
        ]
    ]


def power_futures(period_type: str = "m") -> pd.DataFrame:
    """
    Power futures prices, indexed by delivery period and trading day.

    Parameters
    ----------
    period_type : {'m' (month, default), 'q' (quarter), 'a' (year)}
        Duration of the product.

    Returns
    -------
    pd.DataFrame
        with power futures prices. Multiindex with levels 0 (start of delivery period)
        and 1 (trading day). Columns: p_base, p_peak, p_offpeak (prices), ts_right
        (end of delivery period), anticipation (timedelta between trade and delivery
        start), basehours, peakhours, offpeakhours in delivery period.
    """
    # Get all trading data and put in big dataframe.
    pieces = {}
    for period_start in range(1, 10):
        try:
            pieces[period_start] = _power_futures(period_type, period_start)
        except NotImplementedError:
            pass
    fut = pd.concat(pieces.values()).reset_index()
    # Set index: values and frequency.
    fut.index = pd.MultiIndex.from_frame(fut[["ts_left", "ts_left_trade"]])
    fut = fut.drop(columns=["ts_left_trade", "ts_left"]).sort_index()
    fut.index.levels[0].freq = pd.infer_freq(fut.index.levels[0])
    return fut


def _gas_futures(
    period_type: str = "m", period_start: int = 1, market_code: str = "ncg"
) -> pd.DataFrame:
    """
    Gas futures prices, indexed by trading day.

    Parameters
    ----------
    period_type : {'m' (month, default), 'q' (quarter), 's' (season), 'a' (year)}
        Duration of the product.
    period_start : int
        1 = next/coming (full) period (ie., "frontyear", "frontmonth" etc) (default)
        2 = period after that, etc.
    market_code : {'ncg' (netconnect-germany, default), 'gpl' (gaspool)}

    Returns
    -------
    pd.DataFrame
        with gas futures prices. Index: trading day. Columns: p (price), ts_left,
        ts_right (timestamps of delivery), anticipation (timedelta
        between trade and delivery start), hours in delivery period.
    """
    # Get values...
    df = pd.read_excel(**_excel_gas(period_type, period_start, market_code))
    df.dropna(inplace=True)
    df.columns = ["ts_left_trade", "p"]
    df = set_ts_index(df, "ts_left_trade", continuous=False)
    # ...add some additional information...
    @functools.lru_cache
    def deliv_f(ts):
        return utils.ts_leftright(ts, period_type, period_start)

    df["ts_left"], df["ts_right"] = zip(*df.index.map(deliv_f))
    df["anticipation"] = df["ts_left"] - df.index
    df["hours"] = (df["ts_right"] - df["ts_left"]).apply(
        lambda dt: dt.total_seconds() / 3600
    )
    # Finally, return in correct row and column order.
    df.index = df.index.rename("ts_left_trade")
    return df[["ts_left", "ts_right", "anticipation", "p", "hours"]]


def gas_futures(period_type: str = "m", market_code: str = "ncg") -> pd.DataFrame:
    """
    Gas futures prices, indexed by delivery period and trading day.

    Parameters
    ----------
    period_type : {'m' (month, default), 'q' (quarter), 's' (season), 'a' (year)}
        Duration of the product.
    market_code : {'ncg' (netconnect-germany, default), 'gpl' (gaspool)}

    Returns
    -------
    pd.DataFrame
        with gas futures prices. Multiindex with levels 0 (start of delivery period) and
        1 (trading day). Columns: p (price), ts_right (end of delivery period),
        anticipation (timedelta between trade and delivery start), hours in delivery
        period.
    """
    # Get all trading data and put in big dataframe.
    pieces = {}
    for period_start in range(1, 10):
        try:
            pieces[period_start] = _gas_futures(period_type, period_start, market_code)
        except NotImplementedError:
            pass
    fut = pd.concat(pieces.values()).reset_index()
    # Set index: values and frequency.
    fut.index = pd.MultiIndex.from_frame(fut[["ts_left", "ts_left_trade"]])
    fut = fut.drop(columns=["ts_left_trade", "ts_left"]).sort_index()
    fut.index.levels[0].freq = pd.infer_freq(fut.index.levels[0])
    return fut


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

# def power_futures(
#     prod: str = "y", earliest_trading=None, earliest_delivery=None
# ) -> pd.DataFrame:
#     """
#     Return futures prices timeseries.

#     Arguments:
#         prod: product to return the prices of. One of {'y', 'q', 'm'}.
#         earliest_trading: only return prices after this date. (default: all)
#         earliest_delivery: only return prices for products whose delivery is
#             on or after this date. (default: all)

#     Returns:
#         Dataframe with EEX futures prices. Index: Multiindex (with levels 0:
#             product {'y', 'q', 'm'}, 1: time stamp of delivery start, 2: date
#             of trading). Columns: p_base (base price), p_peak (peak price),
#             p_offpeak (offpeak price). ts_right_deliv (time stamp of delivery
#             end), anticipation (timedelta between trade and delivery
#             start), basehours, peakhours, offpeakhours in delivery period.
#     """
#     engine = create_engine("mssql+pymssql://sqlbiprod/__Staging")

#     try:
#         dur_months = {"m": 1, "q": 3, "y": 12}[prod]
#     except KeyError:
#         raise ValueError("Argument 'prod' must be in {'y', 'q', 'm'}.")

#     # Get values from server...
#     where = f"WHERE Typ='{prod.upper()}'"
#     if earliest_trading:
#         where += f" AND EEX_Handelstag > '{earliest_trading}'"
#     if earliest_delivery:
#         where += f" AND Lieferzeitraum > '{earliest_delivery}'"
#     df = pd.read_sql_query(
#         f"SELECT * FROM dbo.Fakten_EW_EEX_Preise_fkunden {where}", engine
#     )
#     # ...massage to get correct format...
#     df = df.rename(
#         columns={
#             "Typ": "deliv_prod",
#             "Peak_Preis": "p_peak",
#             "Base_Preis": "p_base",
#             "EEX_Handelstag": "ts_left_trade",
#             "Lieferzeitraum": "ts_left_deliv",
#         }
#     )
#     df = df.drop("ID", axis=1)
#     df["ts_left_trade"] = pd.to_datetime(df["ts_left_trade"]).dt.tz_localize(
#         "Europe/Berlin"
#     )
#     df["ts_left_deliv"] = pd.to_datetime(df["ts_left_deliv"]).dt.tz_localize(
#         "Europe/Berlin"
#     )
#     df["ts_right_deliv"] = df["ts_left_deliv"] + pd.offsets.DateOffset(
#         months=dur_months
#     )
#     df["anticipation"] = df["ts_left_deliv"] - df["ts_left_trade"]
#     # ...get peak and base and offpeak hours...
#     h = df[["ts_left_deliv", "ts_right_deliv"]].drop_duplicates()
#     bpo = hours_bpo(h["ts_left_deliv"], h["ts_right_deliv"])
#     h["basehours"], h["peakhours"], h["offpeakhours"] = bpo
#     df = pd.merge(df, h, on=["ts_left_deliv", "ts_right_deliv"])
#     # ...and use to calculate offpeak prices.
#     df["p_offpeak"] = p_offpeak(
#         df["p_base"], df["p_peak"], df["basehours"], df["peakhours"]
#     )
#     # Finally, return in correct row and column order.
#     # TODO: add frequency to index, if possible
#     df = df.set_index(["deliv_prod", "ts_left_deliv", "ts_left_trade"]).sort_index()
#     return df[
#         [
#             "ts_right_deliv",
#             "anticipation",
#             "p_base",
#             "p_peak",
#             "p_offpeak",
#             "basehours",
#             "peakhours",
#             "offpeakhours",
#         ]
#     ]
