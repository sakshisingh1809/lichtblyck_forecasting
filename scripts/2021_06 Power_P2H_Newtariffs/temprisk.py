"""
Script to calculate temprisk premium for p2h customers that stay for 12 or 24 months.

Based on historic values (backtesting).

Assumptions:
. 3 months between acquisition and start of delivery 
. For 12 month delivery duration --> 3-15 months between acquisition and delivery -->
  use 9 months as average duration between acquisition (= price fixation) and delivery. 
  (Needed to estimate long-term price changes.)
. For 24 month delivery duration --> use 21 months
"""

#%%

import lichtblyck as lb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from scipy.stats import norm

os.chdir(
    "C:\\Users\\ruud.wijtvliet\\Ruud\\OneDrive - LichtBlick SE\\Work_in_RM\\python\\2020_01_lichtblyck"
)

act = pd.DataFrame(columns=[[], []])  # 2-level columns
exp = pd.DataFrame(columns=[[], []])  # 2-level columns
exp1 = pd.DataFrame(columns=[[], []])  # 2-level columns
exp0 = pd.DataFrame(columns=[[], []])  # 2-level columns


# %% TEMPERATURE INFLUENCE

# Temperature weights
weights = pd.DataFrame(
    {
        "power": [0.1, 0.7, 1.3, 1.5, 1.9, 0.7, 0.3, 0, 0.1, 0.1, 0, 1.3, 1.1, 0, 0],
        "gas": [0.7, 4.0, 13, 29.0, 13.2, 3.6, 2.9, 0, 1.8, 0.4, 0, 9.4, 3.4, 0, 0.1],
    },
    index=range(1, 16),
)  # GWh/a in each zone
weights = weights["gas"] / weights["gas"].sum()

# Temperature to load.
p2h_details = {"rh": {"source": 2, "spec": 612620}, "hp": {"source": 3, "spec": 53388}}
tlp = {
    pf: lb.tlp.power.fromsource(details["source"], spec=details["spec"])
    for pf, details in p2h_details.items()
}
tlp["gas"] = lb.tlp.gas.D14(kw=1000000)  # not used, just for comparison
# quick visual check
for pr in tlp.values():
    lb.tlp.plot.vs_time(pr)
    try:
        lb.tlp.plot.vs_t(pr)  # for some reason, fails.
    except:
        pass


# %% ACTUAL

# Actual spot prices.
act[("spot", "p")] = lb.prices.montel.power_spot()
act.spot.p = lb.tools.fill_gaps(act.spot.p, 5)
# (split into (1) month average base and peak, and (2) month-to-hour deviations)
act[("spot_m", "p")] = lb.prices.convert.tseries2tseries(act.spot.p, "MS")
act[("spot_m2h", "p")] = act.spot.p - act.spot_m.p
# visual check
act[["spot", "spot_m"]].plot()
act[["spot", "spot_m", "spot_m2h"]].loc["2020-11":"2020-12", :].plot()

# Actual temperature.
t = lb.historic.tmpr()
t = lb.historic.fill_gaps(t)
t_act = t.wavg(weights.values, axis=1)
act[("envir", "t")] = lb.changefreq_avg(t_act, "H")

# Trim on both sides to discard rows for which temperature and price are missing.
act = act.dropna().resample("H").asfreq()

# Actual offtake.
act[("offtake", "w")] = tlp["rh"](act.envir.t)


# %% EXPECTED, OFFTAKE (General, at timepoint 0 and timepoint 1)

# Expected temperature.
# . Variation 1: expected temperature is monthly average of previous 11 years.
# t_exp = lb.historic.tmpr_monthlymovingavg(11)
# t_exp = t_exp.dropna()
# t_exp['t_germany'] = lb.tools.wavg(t_exp, weights, axis=1)
# t_exp = t_exp.resample('D').ffill()
# exp[('envir', 't1')] = t_exp('t_germany').resample('H').ffill()
# . Variation 2: expected temperature is seasonality and trend (see emrald xlsx file)
ti = pd.date_range(
    "2000-01-01", "2021-01-01", freq="D", tz="Europe/Berlin", closed="left"
)
tau = (ti - pd.Timestamp("1900-01-01", tz="Europe/Berlin")).total_seconds()
tau = tau / 3600 / 24 / 365.24  # zeit in jahresfractionen seit 1900
t_exp = pd.Series(
    5.84 + 0.0378945 * tau + -9.0338 * np.cos(2 * np.pi * (tau - 19.366 / 365.24)), ti
)

# Expected temperature and offtake.
exp[("envir", "t")] = t_exp
exp[("offtake", "w")] = tlp["rh"](exp.envir.t)


# %% EXPECTED, PRICES, 1: ~ 3 weeks before delivery month: no temperature influence in price.


def exp_price(df):
    cols = ["p_offpeak", "p_peak", "ts_left_trade", "anticipation"]  # columns to keep
    # keep suitable
    mask = (df["anticipation"] > datetime.timedelta(days=21)) & (
        df["anticipation"] < datetime.timedelta(days=35)
    )
    df = df[mask].dropna()
    # find one value
    if not df.empty:
        df = df.sort_values("anticipation")  # sort suitable.
        df = df.reset_index()  # to get 'ts_left_trade' in columns.
        return pd.Series(df[cols].iloc[0])  # keep one.
    return pd.Series([], dtype=pd.Float32Dtype)


# Expected spot prices.
# . Use futures prices to calculate the expected average price level.
monthprices = lb.prices.montel.power_futures("m")
# . Prices before temperature influence: at least 21 days before delivery start.
p_exp1_m = monthprices.groupby(level=0).apply(exp_price).dropna().unstack()
print(p_exp1_m[["ts_left_trade", "anticipation"]].describe()) # quick check
p_exp1_m = lb.tools.set_ts_index(
    p_exp1_m.drop(columns=["ts_left_trade", "anticipation"])
)
p_exp1_m["po_spread"] = p_exp1_m.p_peak - p_exp1_m.p_offpeak
# quick visual check
pd.DataFrame({**p_exp1_m, "po_spread": p_exp1_m.p_peak - p_exp1_m.p_offpeak}).plot()

# . Use actual spot prices to calculate expected M2H profile: mean but without extremes
rolling_av = lambda nums: sum(np.sort(nums)[5:-15]) / (len(nums) - 20)
p_act_m2h = act.spot_m2h.p
# .. for each hour and weekday, find average m2h value for past 52 values.
p_exp1_m2h = p_act_m2h.groupby(lambda ts: (ts.weekday(), ts.hour)).apply(
    lambda df: df.rolling(52).apply(rolling_av).shift()
)
# . Make arbitrage free (average = 0)
p_exp1_m2h = p_exp1_m2h - lb.prices.convert.tseries2tseries(p_exp1_m2h, "MS")
# visual check
weeksatsun_function = lambda ts: {5: "Sat", 6: "Sun"}.get(ts.weekday(), "Weekday")
monday_function = lambda ts: ts.floor("D") - pd.Timedelta(ts.weekday(), "D")
cmap = mpl.cm.get_cmap("hsv")  # cyclic map
for daytype, daytypedf in p_exp1_m2h.groupby(weeksatsun_function):  # weekday, sat, sun
    fig, axes = plt.subplots(figsize=(20, 10), ncols=2)
    fig.suptitle(daytype)

    ispeak = lb.is_peak_hour(daytypedf.index)
    for ax, fltr, title in zip(axes, [~ispeak, ispeak], ["offpeak", "peak"]):
        for hour, subdf in daytypedf[fltr].groupby(lambda ts: ts.hour):
            subdf = subdf.groupby(monday_function).mean()
            ax.plot(subdf, color=cmap(hour / 24), label=hour)
            ax.set_title(title)
            ax.legend()

# Expected prices.
exp1[("m", "p")] = lb.prices.convert.bpoframe2tseries(p_exp1_m, "H")
exp1[("m2h", "p")] = p_exp1_m2h
exp1[("pfc", "p")] = exp1.m.p + exp1.m2h.p
exp1 = exp1.dropna()


# %% EXPECTED 0: Price at time of acquisition


def factor_p_a2m(p_m: pd.DataFrame) -> pd.DataFrame:
    """Calculate the factors to go from year to month prices.

    Args:
        p_m: pd.Dataframe with month prices (peak and offpeak columns).
    """
    # . keep only full years and keep only the prices.
    p_m = p_m[["p_peak", "p_offpeak"]]
    first = lb.tools.floor(p_m.index[0], "AS", 1)
    last = lb.tools.floor(p_m.index[-1], "AS", 0)
    mask = (p_m.index >= first) & (p_m.index < last)
    # . each month's price as fraction of yearly average.
    factors = (
        p_m[mask]
        .resample("AS")
        .transform(lambda df: df / lb.changefreq_avg(df, "AS").iloc[0])
    )
    factors.index = pd.MultiIndex.from_tuples(
        [(ts.year, ts.month) for ts in factors.index], names=("year", "month")
    )
    factors = factors.unstack(0)
    # . visual check: curve for each month
    fig, axes = plt.subplots(figsize=(20, 10), ncols=2)
    fig.suptitle("Price factors throughout the year")
    for ax, daytime in zip(axes, ["offpeak", "peak"]):
        factors[f"p_{daytime}"].plot(ax=ax, cmap="jet")
        ax.set_title(daytime)
        ax.legend()
    # . average factor for each month
    return factors.groupby(level=0, axis=1).apply(lambda df: df.mean(axis=1))


def price_at_acquisition(
    index, monthpricefactors, anticipation=pd.Timedelta(days=365 * 1.5)
):
    """Calculate month price at time of acquisition.

    Args:
        index: timestamps of delivery months of interest.
        monthpricefactors (pd.Series): ratio between month and year price for each of 12 months.
        anticipation: how far before start of delivery period, the price is wanted. Defaults to 1.5 years.
    """
    # Find prices.
    # . for which month do we want to know price
    p_exp0_m = pd.DataFrame(index=index)
    # . month and year prices
    fut = {prod: lb.prices.montel.power_futures(prod) for prod in ["a", "m"]}

    def monthprice(ts_left):
        cols = ["p_peak", "p_offpeak", "ts_left_trade"]  # columns to keep
        ts_trade = ts_left - anticipation
        # find price in month futures
        df = fut["m"]
        mask = (df.index.get_level_values("ts_left") == ts_left) & (
            df.index.get_level_values("ts_left_trade") < ts_trade
        )
        df = df[mask].dropna()
        # find one value
        if not df.empty:
            df = df.sort_values("anticipation")  # sort suitable.
            df = df.reset_index()  # to get 'ts_left_trade' in columns.
            return pd.Series(df[cols].iloc[0])  # keep one.
        # find price from year futures
        df = fut["a"]
        mask = (
            df.index.get_level_values("ts_left") == lb.utils.floor(ts_left, "AS")
        ) & (df.index.get_level_values("ts_left_trade") < ts_trade)
        df = df[mask].dropna()
        # find one value
        if not df.empty:
            df = df.sort_values("anticipation")  # sort suitable.
            df = df.reset_index()  # to get 'ts_left_trade' in columns.
            s = pd.Series(df[cols].iloc[0])  # year price
            for key in ["p_peak", "p_offpeak"]:
                s[key] /= monthpricefactors.loc[ts_left.month, key]
            return s  # month price

    p_exp0_m[["p_peak", "p_offpeak", "ts_left_trade"]] = p_exp0_m.index.map(
        lambda ts: monthprice(ts)
    )
    return p_exp0_m


monthpricefactors = factor_p_a2m(p_exp1_m)
prices_at_various_leadtimes = [
    price_at_acquisition(
        p_exp1_m.index, monthpricefactors, pd.Timedelta(days=365 * lead)
    )
    for lead in np.linspace(0.5, 3.5, 5)
]
p0_m = pd.DataFrame([p["p_month"] for p in prices_at_various_leadtimes]).mean().T

# Turn into forward curve.
# (using previously calculated m2d curve. Not entirely correct, because uses data that
# was not available yet. But m2d profile expected to be quite stable)
exp0[("m2d", "p")] = exp1[("m2d", "p")]
exp0[("m", "p")] = p0_m.resample("D").ffill()
exp0[("pfc", "p")] = exp0.m2d.p + exp0.m.p
#%%


print("halo")

# %%

#%%
