# -*- coding: utf-8 -*-
"""
Standardized temperature load profiles for electricity consumption.
"""
from typing import Union, Callable, Tuple
from . import convert
import pandas as pd
import datetime as dt

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


def fromsource(
    source: Union[str, int], *, spec: float
) -> Callable[[pd.Series], pd.Series]:
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
    Callable[[pd.Series], pd.Series]
        Function that takes a temperature [degC] timeseries as input and
        returns the consumption [MW] timeseries as output.
    """
    tlp_s = _series_fromsource(source, spec=spec)
    return convert.series2function(tlp_s)


def _series_fromsource(source: Union[str, int], *, spec: float) -> pd.Series:
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
        Load values (in [MW]), as function of 2-level row index ('t' for temperature,
        [degC], 'time_left_local' for time of day).

    Notes
    -----
    See also `fromsource`.
    """

    def get_time(time_or_datetime):
        try:
            return time_or_datetime.time()
        except AttributeError:
            return time_or_datetime

    def subtract_15_min_overflow(time):
        return (
            dt.datetime.combine(dt.date.today(), time) + dt.timedelta(hours=-0.25)
        ).time()

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
    df["time_right_local"] = df["Uhrzeit"].apply(get_time)
    df["time_left_local"] = df["time_right_local"].apply(subtract_15_min_overflow)
    df = df.set_index("time_left_local")
    df = df.drop(columns=["Nr.", "Uhrzeit", "time_right_local"])
    # Put in correct output format (long table).
    tlp_s = df.stack().swaplevel().sort_index() * spec * 0.001  # kW to MW
    tlp_s.index.rename(["t", "time_left_local"], inplace=True)
    return tlp_s
