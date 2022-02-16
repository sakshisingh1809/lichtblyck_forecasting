# -*- coding: utf-8 -*-
"""
Script to calculate the temperature and PaR-risk in Ludwig PF for last 3
months of 2020

2020_10 RW
"""

import lichtblyck as lb
import pandas as pd
import numpy as np
import importlib

wia = importlib.import_module("scripts.2020_10 Temprisk_Ludwig.WhatIfAnalysis")
wia.GoalSeek

# %% Get offtake curve for expected churn and expected temperatures,
# for heating part of portfolio only.

df = pd.read_excel("scripts/2020_10 Temprisk_Ludwig/Zeitreihenbericht.xlsx", header=1)
df = lb.tools.set_ts_index(df, "ZEITSTEMPEL", "right")
df.columns = pd.MultiIndex.from_tuples(
    (
        ("pfc", "pfc", "p"),
        ("rh", "current", "w"),
        ("hp", "current", "w"),
        ("rest", "current", "w"),
        ("rh", "certain", "w"),
        ("hp", "certain", "w"),
        ("rest", "certain", "w"),
        ("rh", "fwd", "w"),
        ("hp", "fwd", "w"),
        ("rest", "fwd", "w"),
        ("rh", "fwd", "r"),
        ("hp", "fwd", "r"),
        ("rest", "fwd", "r"),
    )
)

# Churnpath as expected
churn_exp = 0.07
factor = -churn_exp / (df.rh.certain.w.q.sum() / df.rh.current.w.q.sum() - 1)
df[("rh", "exp", "w")] = df.rh.current.w + (df.rh.certain.w - df.rh.current.w) * factor
factor = -churn_exp / (df.hp.certain.w.q.sum() / df.hp.current.w.q.sum() - 1)
df[("hp", "exp", "w")] = df.hp.current.w + (df.hp.certain.w - df.hp.current.w) * factor
factor = -churn_exp / (df.rest.certain.w.q.sum() / df.rest.current.w.q.sum() - 1)
df[("rest", "exp", "w")] = (
    df.rest.current.w + (df.rest.certain.w - df.rest.current.w) * factor
)


#%% Find temperature load profiles, that fit to the expected, warm, and cold yearly offtake

# Expected temperatures.
ti = pd.date_range(
    "2021-01-01", "2022-01-01", freq="D", tz="Europe/Berlin", closed="left"
)
tau = (
    (ti - pd.Timestamp("1900-01-01", tz="Europe/Berlin")).total_seconds()
    / 3600
    / 24
    / 365.24
)
t_exp = pd.Series(
    5.83843470203356
    + 0.037894551208033 * tau
    + -9.03387134093431 * np.cos(2 * np.pi * (tau - 19.3661745382612 / 365.24)),
    index=ti,
)

# Expected offtake profiles.
t2l = pd.DataFrame(
    {
        "rh": lb.tlp.standardized_tmpr_loadprofile(2),
        "hp": lb.tlp.standardized_tmpr_loadprofile(3),
    }
)
spec = pd.Series(
    {"rh": 612620, "hp": 53388}
)  # to find spec: spec = (1-churn_exp) * df.hp.current.w.q.sum() / exp.hp.w.q.sum()


def offtake_ludwig_heating(dt):
    w_rh = lb.tlp.tmpr2load(t2l.rh, t_exp + dt, spec=spec.rh)
    w_hp = lb.tlp.tmpr2load(t2l.hp, t_exp + dt, spec=spec.hp)
    return w_rh + w_hp


# Norm, Cold and Warm year
# t2l = lb.tlp.standardized_tmpr_loadprofile(2)
# q = lambda dt: lb.tlp.tmpr2load(t2l, t_exp+dt, spec=1).q.sum()
# q0 = q(0)
# wia.GoalSeek(q, q0*0.9, 1, 0.01)
delta = pd.DataFrame({"warm": 1.06, "cold": -2.15}, index=["t"])
offtake = pd.DataFrame(
    {
        ("norm", "w"): offtake_ludwig_heating(0),
        ("cold", "w"): offtake_ludwig_heating(delta.cold.t),
        ("warm", "w"): offtake_ludwig_heating(delta.warm.t),
    }
)
assert np.isclose(
    offtake.norm.q.sum(),
    -(1 - churn_exp) * (df.rh.current.q.sum() + df.hp.current.q.sum()),
)  # check
delta = pd.concat(
    [
        delta,
        pd.DataFrame(
            {
                "warm": (offtake.warm.q - offtake.norm.q).sum(),
                "cold": (offtake.cold.q - offtake.norm.q).sum(),
            },
            index=["q"],
        ),
    ]
)


#%% See what load and temperatures are, for warm and cold year


import statsmodels.api as sm
import statsmodels.formula.api as smf

quantile = 0.2
warmerdp = daily[daily.dt > 0][["dt", "p_spot_diff"]]
model = smf.quantreg("p_spot_diff ~ dt", warmerdp)
slope = model.fit(quantile).params["dt"]
p_spot_diff = slope * delta.warm.t
temp_cost = p_spot_diff * delta.warm.q
temp_income = offtake.warm.q.sum() * 1.27

quantile = 0.8
colderdp = daily[daily.dt < 0][["dt", "p_spot_diff"]]
model = smf.quantreg("p_spot_diff ~ dt", colderdp)
slope = model.fit(quantile).params["dt"]
p_spot_diff = slope * delta.cold.t
temp_cost = p_spot_diff * delta.cold.q
temp_income = offtake.cold.q.sum() * 1.27


quantile = 0.2  # lowest prices
warmerdp = daily[daily.w_diff < 0][["w_diff", "p_spot_diff"]]
model = smf.quantreg("p_spot_diff ~ w_diff", warmerdp)
slope = model.fit(quantile).params["w_diff"]
p_spot_diff = slope * delta.warm.t
temp_cost = p_spot_diff * delta.warm.q
temp_income = offtake.warm.q.sum() * 1.27

quantile = 0.8
colderdp = daily[daily.dt < 0][["dt", "p_spot_diff"]]
model = smf.quantreg("p_spot_diff ~ dt", colderdp)
slope = model.fit(quantile).params["dt"]
p_spot_diff = slope * delta.cold.t
temp_cost = p_spot_diff * delta.cold.q
temp_income = offtake.cold.q.sum() * 1.27

#%%

pricedelta = monthly.p_spot_diff
model = norm(*norm.fit(pricedelta))
x = np.linspace(-100, 100, 1000)
y = model.pdf(x)
plt.plot(x, y)
plt.axvline(10, color="r")
plt.axvline(-10, color="r")
model.cdf(10)
model.cdf(-10)
