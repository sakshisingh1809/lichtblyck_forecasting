"""
Prices at a specific trading day or trading day range
"""

import pandas as pd
from pandas.core.frame import NDFrame


def with_anticipation(
    fr: pd.DataFrame,
    /,
    anticipation: pd.Timedelta = None,
    ts_left_trade: pd.Timestamp = None,
):
    """
    Return a subset of a price based on the moment at which they were valid.

    Parameters
    ----------
    fr : pd.DataFrame
        Prices. Dataframe with multiindex: level 0: start of delivery period, level 1:
        trading moment. And with column "anticipation".
    anticipation : pd.Timedelta, optional
        [description], by default None
    ts_left_trade : pd.Timestamp, optional
        [description]
    """
    pass