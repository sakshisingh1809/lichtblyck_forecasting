"""
Using price time series to estimate volatility.
"""

import lichtblyck as lb
import pandas as pd
import numpy as np
from typing import Union

p = lb.prices.futures("y")

df = p.groupby("ts_left_deliv").apply(
    lambda df: lb.prices.vola(
        df[["p_peak", "p_offpeak"]].droplevel(["deliv_prod", "ts_left_deliv"])
    )
)

df.unstack(0).plot(cmap="jet")
