import pandas as pd


def ts_right(ts_left: pd.Timestamp) -> pd.Timestamp:
    """Right-bound timestamp associated with a (left-bound) timestamp."""
    if ts_left.freq == "D":
        kwargs = {"days": 1}
    elif ts_left.freq == "MS":
        kwargs = {"months": 1}
    elif ts_left.freq == "QS":
        kwargs = {"months": 3}
    elif ts_left.freq == "AS":
        kwargs = {"years": 1}
    else:
        raise ValueError(f"Invalid frequency: {ts_left.freq}.")
    return ts_left + pd.DateOffset(**kwargs)


def duration(ts_left: pd.Timestamp) -> float:
    """Duration [h] associated with a timestamp."""
    return (ts_right(ts_left) - ts_left).total_seconds() / 3600


ts1 = pd.Timestamp("2022-03-01", tz="Europe/Berlin", freq="D")
ts_right(ts1)  # Timestamp('2022-03-02 00:00:00+0100', tz='Europe/Berlin')
duration(ts1)  # 24.0

ts2 = pd.Timestamp("2022-03-27", tz="Europe/Berlin", freq="D")
ts_right(ts2)  # Timestamp('2022-03-28 00:00:00+0200', tz='Europe/Berlin')
duration(ts2)  # 23.0
