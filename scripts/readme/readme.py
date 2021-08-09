"""
Create DataFrames in the README.md
"""

#%%

import lichtblyck as lb
import pandas as pd
import numpy as np

# %% UPSAMPLE
years = pd.DataFrame(
    {"q": 1000, "r": 30000, "t": 7.98quart},
    pd.date_range("2020", periods=1, freq="AS", tz="Europe/Berlin"),
)
years["w"] = years.q / years.duration
years["p"] = years.r / years.q
years = lb.set_ts_index(years[["q", "w", "r", "p", "t"]])

quarters = pd.concat(
    [
        lb.changefreq_sum(years[["q", "r"]], "QS"),
        lb.changefreq_avg(years[["w", "t"]], "QS"),
    ],
    axis=1,
)
quarters["p"] = quarters.r / quarters.q
quarters = lb.set_ts_index(quarters[["q", "w", "r", "p", "t"]])

# %% DOWNSAMPLE
quarters = pd.DataFrame(
    {"q": [300, 180, 200, 320], "t": [1.3, 12.3, 15.1, 3.2]},
    pd.date_range("2020", periods=4, freq="QS", tz="Europe/Berlin"),
)
quarters['r'] = quarters.q * [37.767, 25.3, 21.3, 30.8]
quarters["w"] = quarters.q / quarters.duration
quarters["p"] = quarters.r / quarters.q
quarters = lb.set_ts_index(quarters[["q", "w", "r", "p", "t"]])

years = pd.concat(
    [
        lb.changefreq_sum(quarters[["q", "r"]], "AS"),
        lb.changefreq_avg(quarters[["w", "t"]], "AS"),
    ],
    axis=1,
)
years["p"] = years.r / years.q
years = lb.set_ts_index(years[["q", "w", "r", "p", "t"]])
# %%
