# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:36:19 2020

@author: ruud.wijtvliet
"""

import lichtblyck_forecasting as lf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Temperature weights
weights = pd.DataFrame(
    {
        "power": [60, 717, 1257, 1548, 1859, 661, 304, 0, 83, 61, 0, 1324, 1131, 0, 21],
        "gas": [
            729,
            3973,
            13116,
            28950,
            13243,
            3613,
            2898,
            0,
            1795,
            400,
            0,
            9390,
            3383,
            9,
            113,
        ],
    },
    index=range(1, 16),
)  # MWh/a in each zone
weights = (
    weights["power"] / weights["power"].sum() + weights["gas"] / weights["gas"].sum()
)

# Temperature to load
t2l = lf.tlp.standardized_tmpr_loadprofile(2)

ispeak = lambda ts: (ts.hour >= 8) & (ts.hour < 20)


# Actual temperature.
act = lf.historic.tmpr()
act = act[act.index >= "1963"]  # after 1963, each day has at most 1 missing station.

# For each missing value, get estimate. Using average difference to other stations' values.
complete = act.dropna()  # only days without missing stations.
filled = act.copy()  # temperature holes filled.
for col in complete.columns:
    d = complete[col] - complete.drop(col, axis=1).mean(axis=1)
    diff = d.mean()  # average difference between this climate zone and all others
    # use diff to fill gaps
    isna = act[col].isna()
    filled.loc[isna, col] = act.drop(col, axis=1).loc[isna].mean(axis=1) + diff

filled["t_germany"] = lf.wavg(filled, weights, axis=1)
filled.t_germany.to_csv("test.csv")
