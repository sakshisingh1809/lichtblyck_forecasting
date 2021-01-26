# -*- coding: utf-8 -*-
"""
Standardized temperature load profiles for electricity consumption.
"""

import pandas as pd
import datetime
from typing import Union, Callable, Tuple
from .convert import series2function

SOURCEPATH = "lichtblyck/tlp/sourcedata/power/"
SOURCES = [
    {"name": "avacon_hz0", "io": SOURCEPATH + "AVACON_HZ0.xlsx"},
    {"name": "avacon_hzs", "io": SOURCEPATH + "AVACON_HZS.xlsx"},
    {"name": "bayernwerk_nsp", "io": SOURCEPATH + "BAYERNWERK_NSP.xlsx"},
    {"name": "bayernwerk_wp", "io": SOURCEPATH + "BAYERNWERK_WP.xlsx"},
    {"name": "edis_n21", "io": SOURCEPATH + "EDIS_N21.xlsx"},
    {"name": "edis_w21", "io": SOURCEPATH + "EDIS_W21.xlsx"},
    {"name": "enmitte_enm", "io": SOURCEPATH + "ENMITTE_ENM.xlsx"},
    {"name": "netzbw_ez2", "io": SOURCEPATH + "NETZBW_EZ2.xlsx"},
    {"name": "shnetz_e1", "io": SOURCEPATH + "SHNETZ_E1.xlsx"},
    {"name": "shnetz_e2", "io": SOURCEPATH + "SHNETZ_E2.xlsx"},
    {"name": "shnetz_w1", "io": SOURCEPATH + "SHNETZ_W1.xlsx"},
    {"name": "westnetz_wk2", "io": SOURCEPATH + "WESTNETZ_WK2.xlsx"},
    {"name": "wwnetz_nsp", "io": SOURCEPATH + "WWNETZ_NSP.xlsx"},
]


def fromsource(source: Union[str, int], *, spec: float) -> Tuple[Callable, str]:
    """
    Standardized temperature-dependent load profile for a certain DSO.

    Parameters
    ----------
    source : Union[str, int]
        Index position, or value of 'name'-key, for any of the dictionaries
        in the SOURCES list.
    spec : float
        Specific electrical load [kWh/K] with which to scale the profile.
        Describes the heating energy needed by the customer during a single
        day, per degC that the average outdoor temperature of that day is
        below a certain set reference value.

    Returns
    -------
    Callable
        Function that takes a temperature [degC] and timestamp as input and
        returns the consumption [MW] as output, and
        string describing its native frequency.

    Notes
    -----
    To obtain the actual electricity consumption on a certain day or during a
    certain time period, use the function `tmpr2load`.
    """
    s = series_fromsource(source, spec=spec)
    return series2function(s)


def series_fromsource(source: Union[str, int], *, spec: float) -> Tuple[pd.Series, str]:
    """
    Standardized temperature-dependent load profile for a certain DSO.

    Parameters
    ----------
    source : Union[str, int]
        Index position, or value of 'name'-key, for any of the dictionaries
        in the SOURCES list.
    spec : float
        Specific electrical load [kWh/K] with which to scale the profile.
        Describes the heating energy needed by the customer during a single
        day, per degC that the average outdoor temperature of that day is
        below a certain set reference value.

    Returns
    -------
    pd.Series
        Load values (in [MW]), as function of 2-level row index (temperature,
        [degC], time of day with quarter-hourly resolution).

    Notes
    -----
    See also `fromsource`.
    """
    try:
        if isinstance(source, str):
            io = [s for s in SOURCES if s["name"] == source][0]["io"]
        else:
            io = SOURCES[source]["io"]
    except:
        raise ValueError(
            "Value for argument 'source' must be one of {"
            + ", ".join([s["name"] for s in SOURCES])
            + "}, or the index position of one of them."
        )
    # Requirements/assumptions for any tlp excel file:
    # . Have column called 'Uhrzeit' with 15min resolution and right-bound timestamp.
    # . Have column 'Nr.'
    # . Have columns with number representing the temperature.
    df = pd.read_excel(header=0, sheet_name=0, io=io)
    df["ts_right_local"] = pd.to_datetime(df["Uhrzeit"], format="%H:%M:%S")
    df["time_left_local"] = (
        df["ts_right_local"] + datetime.timedelta(hours=-0.25)
    ).dt.time
    df = df.set_index("time_left_local")
    df = df.drop(columns=["Nr.", "Uhrzeit", "ts_right_local"])
    # Put in correct output format (long table).
    s = df.stack().swaplevel().sort_index() * spec * 0.001  # kW to MW
    s.index.rename(["t", "time_left_local"], inplace=True)
    return s
