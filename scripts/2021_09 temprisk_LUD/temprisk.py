"""
Temperature risk premium in Ludwig PF.

Simulations:
. Starting point (O):
.. Offtake volume with norm temperature.
.. Offtake revenue from (sourced volume+prices) + (open volume+market prices).
.. Full hedge is done.

. Long-term changes (L):
.. Using current volatility and random walk to get change in year prices (peak/offpeak).
.. Using historic factor to get month prices (peak/offpeak).

. Short-term changes (S):
.. Once: historic analysis to find short-term-squared influence for the profile under
   consideration. This is influence of short-term temperature changes on offtake volume,
   combined with short-term changes in prices, leading to shortterm revenue (+/-). For 
   each historic year, we get 12 (volume change, price change, short-term revenue) 
   datapoints.
.. Per simulation: pick a historic year. For each month, combine the short-term volume
   change and the total price change. The total price change is the simulated long-term
   price change and the historic short-term price change.

We get n simulations with each a total influence on the price [Eur/MWh] - per month or
per year. The premium and VaR are calculated from their distribution.



Three time points are considered:
    0: customer is given price based on sourced and unsourced volume
    1: few weeks before start of delivery month
    2: delivery
(0) and (1) share the temperature (expectation), and with it the offtake volume
(expectation). As no more volume is bought between (0) and (1), they also share
the spot volume (expectation).

Process:

(a) valuation
temperature -> (with tlp:) offtake -> (with prices:) value of offtake.

This is done at all 3 points in time.
For (0), the current portfolio mix price for 2022 is used. 
For (1) and (2), first a simulation of monthly prices is needed. This is calculated as the
    sum of simulated year price curve, plus current year-to-month profile. Then, a month-to-
    hour profile need to be added to both, which comes from historic data.
A historic year is picked. For (1), the month-to-hour profile is the average month-to-hour
profile of the preceding period. For (2), the month-to-hour profile is the actual month-to-
hour profile, from the spot prices. So, the difference between (1) and (2) is the same in
the simulation as in the historic year.

For the volume:
For (0) and (1), the offtake is calculated with the temperature expectation of the historic year. 
For (2), the offtake is calculated with the actual historic temperatures.

That way, the historic temperatures influences both volume and price, causing (historic)
covariance costs.

At (0), a hedge is assumed, e.g. at month or year level. Any volume difference between
this hedge and the final volume must be traded at the market.


(b) comparison
. only volume change is between (1) and (2). The volume to be bought is the final volume
minus an assumed hedge at moment (0).
. PaR has 2 components:
.. "lt" = price change (0->1) times open volume --> long term effect due to change in
general price level
.. "st" = price change (1->2) times open volume --> short term effect, including (but
not only) correlation


Variations:
  1. Expected temperature = monthly average, i.e., same temperature for each day of a
  given month.
  2. Expected temperature = 'structured' temperature, i.e., already including some
  temperature path.


Room for improvement:
. Don't pick entire historic year for short-term influence, but instead simulate. Diffi-
  culty is in keeping historic 'cohesion' between months (i.e., if Jan is cold, Feb is
  likely cold too).

Abbreviations:
. rh = resistive heating = nachtspeicher
. hp = heat pump = waermepumpe
. quantities:
.. p = price [Eur/MWh]
.. r = revenue [Eur]
.. w = volume [MW]
.. q = volume [MWh]
. portfolio parts:
.. o = offtake 
.. s = sourced
.. u = unsourced
. combinations:
.. e.g. pu = price of unsourced volume = market price [Eur/MWh]
.. e.g. po = price of offtake volume = portfolio mix price [Eur/MWh]
.. e.g. qo = offtake volume [MWh]
. moments / periods in procurement process:
. O = 0 = when tariff is fixed
. L = 1 = 1 month before start of delivery month
. S = 2 = at spot trade
"""

#%% IMPORTS

from lichtblyck.core2.pfstate import PfState
from typing import Callable
import lichtblyck as lb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from urllib.parse import urlsplit
from io import StringIO
from pathlib import Path
from scipy.stats import norm

# os.chdir(Path(__file__).parent.parent.parent)

stepS = pd.DataFrame(columns=[[], []])  # 2-level columns
exp = pd.DataFrame(columns=[[], []])  # 2-level columns
stepL = pd.DataFrame(columns=[[], []])  # 2-level columns


#%% OUTLINE
#
# PRICES.
#
# Year prices.
# Spot prices.
#
#
# TEMPERATURE TO OFFTAKE.
#
# Temperature weights.
# Expected temperatures.
# Actual temperature.
# Temperature to offtake.
#
#
# STEP O: offtake, market prices, when tariff is fixated.
#
# Temperatures.
# Offtake.
# Market prices.
# . HPFC.
# . Yearly Base.
# . M2H = QHPFC - Base. (M2H assumed constant until after step L.)
# Sourced prices and revenue.
# Offtake prices and revenue.
#
#
# STEP L: long-term market price changes due to volatility.
#
# Market prices: simulation.
# . Base.
# . QHPFC = Base + M2H.
#
#
# STEP S: short-term offtake and market price changes due to temperature.
#
# Historic values.
# . Expected temperatures and offtake volumes.
# . Expected market prices. (=qhpfc)
# . Hedge at expected prices.
# . Actual temperatures and offtake volumes.
# . Actual market prices. (=spot)
# . Change in market prices.
# Short term: simulation.
# . Pick historic year.
# . Get spot price by adding
# .. current market price, and
# .. simulated long-term price change, and
# .. historic short-term price change.
# . Get spot volume = historic offtake volume - hedge
# . Calculate spot revenue from spot volume * spot price
# . Calculate final price per delivery month = total revenue / offtake volume
# . Get simulated temperature risk [Eur/MWh] = final price - current price.
#
#
# SIMULATIONS.
#
# Run n simulations.
# Create distribution.
#
#
# PREMIUMS AND VALUE AT RISK.
#
# Get x-percentile of temperature risk values.

#%% PRICES.

__file__ = "."

# Year prices.
pu_fut = lb.prices.montel.power_futures("a")
# Spot prices.
pu_spot = lb.prices.montel.power_spot()

#%% TEMPERATURE TO OFFTAKE.

# Temperature weights.
weights = pd.DataFrame(
    {
        "power": [0.1, 0.7, 1.3, 1.5, 1.9, 0.7, 0.3, 0, 0.1, 0.1, 0, 1.3, 1.1, 0, 0],
        "gas": [0.7, 4.0, 13, 29.0, 13.2, 3.6, 2.9, 0, 1.8, 0.4, 0, 9.4, 3.4, 0, 0.1],
    },
    index=range(1, 16),
)  # GWh/a in each zone
weights = weights["gas"] / weights["gas"].sum()

# Expected temperatures.
ti = pd.date_range("2000", "2023", freq="D", tz="Europe/Berlin", closed="left")
tau = (ti - pd.Timestamp("1900-01-01", tz="Europe/Berlin")).total_seconds()
tau = tau / 3600 / 24 / 365.24  # zeit in jahresfractionen seit 1900
t_exp = pd.Series(
    5.84 + 0.0379 * tau + -9.034 * np.cos(2 * np.pi * (tau - 19.366 / 365.24)), ti
)

# Actual temperature.
t = lb.tmpr.hist.tmpr()
t = lb.tmpr.hist.fill_gaps(t)
t_act = t.wavg(weights.values, axis=1)

# Temperature to offtake.
p2h_details = {"rh": {"source": 2, "spec": 571570}, "hp": {"source": 3, "spec": 54475}}
tlps = {
    pf: lb.tlp.power.fromsource(details["source"], spec=details["spec"])
    for pf, details in p2h_details.items()
}
tlps["gas"] = lb.tlp.gas.D14(kw=1000000)  # not used, just to comparison with gas prof.
# quick visual check
for tlp in tlps.values():
    lb.tlp.plot.vs_time(tlp)
    lb.tlp.plot.vs_t(tlp)

tlp = lambda ts: tlps["rh"](ts) + tlps["hp"](ts)


#%% STEP O: offtake, market prices, when tariff is fixated. Keep only the offtake price,
# and the current monthly price structure.

# Temperature.
t2022 = lb.changefreq_avg(t_exp.loc["2022"], "15T")

# Offtake.
offtakevolume = -lb.PfLine(tlp(t2022))

# data_url = "https://dev.azure.com/lichtblick/FRM/_git/lichtblyck?path=/scripts/2021_09%20temprisk_LUD/20210922_084614_Zeitreihenbericht.xlsx"
# path = StringIO(urlsplit(data_url).geturl())


# Market prices.
# . QHPFC.
data = pd.read_excel(
    Path(__file__).parent / "20211007_110233_Zeitreihenbericht.xlsx",
    header=1,
    index_col=0,
    names=["ts_right", "qhpfc", "rh_wo", "rh_ws", "rh_rs", "hp_wo", "hp_ws", "hp_rs"],
)

data = lb.set_ts_index(data, bound="right")
pu = data.qhpfc.rename("p")
# Sourced prices and revenue.
sourced = lb.PfLine({"w": data["rh_ws"], "r": data["rh_rs"]}) + lb.PfLine(
    {"w": data["hp_ws"], "r": data["hp_rs"]}
)

# Offtake prices and revenue.
step0 = lb.PfState(offtakevolume, pu, sourced).changefreq("H")
current = pd.DataFrame({("O", "po"): step0.pnl_cost.p})
current[("O", "pu")] = pu


#%% Changes during step L: Simulation. long-term market price changes due to volatility.


def simulate_L_fn(vola: float, marketprice: pd.Series, now: pd.Timestamp) -> Callable:
    """Create function to simulate random walk of base price and return pu_m in base and peak.

    Parameters
    ----------
    vola : float
        Volatility in %/a
    marketprice : pd.Series
        Current marketprices [Eur/MWh]
    now : pd.Timestamp
        Current timestamp
    """
    i = pd.date_range("2022", "2023", freq="MS", closed="left", tz="Europe/Berlin")

    # a2m = factor to turn cal base price into monthly peak and offpeak
    calbase = marketprice.resample("AS").transform("mean")
    monthpeakoffpeak = lb.prices.convert.tseries2tseries(marketprice, "MS")
    a2m = monthpeakoffpeak / calbase

    p_base = marketprice.mean()
    freq = a2m.index.freq

    def fn():
        # Market prices: simulation.
        price = p_base
        ts = now
        prices = pd.Series(index=i, dtype=float)
        # . Base.
        for month in i:
            ts_stepL = lb.floor_ts(month, -1, "MS")  # 1 month before start of delivery
            time = (ts_stepL - ts).total_seconds() / 3600 / 24 / 365.24
            price *= lb.simulate.randomwalkfactor(vola, time)
            prices[month] = price
            ts = ts_stepL
        # . QHPFC = Base * a2m factor
        return lb.changefreq_avg(prices, freq) * a2m

    return fn


sim_L = simulate_L_fn(
    vola=0.01,  # per calendar year
    marketprice=current.O.pu,
    now=pd.Timestamp.today().tz_localize("Europe/Berlin"),
)

#%% Get historic values during various years. Used to sample.

hist = pd.DataFrame(columns=[[], []])  # 2-level columns

# A) Price: after S, after L, change during S.

# Actual spot prices.
hist[("S", "pu")] = pu_spot
hist.S.pu = lb.fill_gaps(hist.S.pu, 5)
# (split into (1) month average peak and offpeak, and (2) month-to-hour deviations)
hist[("S", "pu_m")] = lb.prices.convert.tseries2tseries(hist.S.pu, "MS")
hist[("S", "pu_m2h")] = hist.S.pu - hist.S.pu_m
# visual check
hist.S[["pu", "pu_m"]].plot()
hist.S[["pu", "pu_m", "pu_m2h"]].loc["2020-11":"2020-12", :].plot()


def price_after_stepL(df):
    cols = ["p_offpeak", "p_peak", "ts_left_trade", "anticipation"]  # columns to keep
    # keep suitable
    mask = (df["anticipation"] > datetime.timedelta(days=30)) & (
        df["anticipation"] < datetime.timedelta(days=39)
    )
    df = df[mask].dropna()
    # find one value
    if not df.empty:
        df = df.sort_values("anticipation")  # sort suitable.
        df = df.reset_index()  # to get 'ts_left_trade' in columns.
        return pd.Series(df[cols].iloc[0])  # keep one.
    return pd.Series([], dtype=pd.Float32Dtype)


# Market prices at end of step L.
# . Use futures prices to calculate the expected average price level.
# . Prices before temperature influence: about 30 days before delivery start.
pu_L_m = pu_fut.groupby(level=0).apply(price_after_stepL).dropna().unstack()
print(pu_L_m[["ts_left_trade", "anticipation"]].describe())  # quick check
pu_L_m = lb.set_ts_index(pu_L_m.drop(columns=["ts_left_trade", "anticipation"]))
assert pu_L_m.isna().any().any() == False  # each month has a price
pu_L_m["po_spread"] = pu_L_m.p_peak - pu_L_m.p_offpeak
# quick visual check
pd.DataFrame({**pu_L_m, "po_spread": pu_L_m.p_peak - pu_L_m.p_offpeak}).plot()

# . Use actual spot prices to calculate expected M2H profile: mean but without extremes
rolling_av = lambda nums: sum(np.sort(nums)[5:-15]) / (len(nums) - 20)
# .. for each hour and weekday, find average m2h value for past 52 values.
pu_L_m2h = hist.S.pu_m2h.groupby(lambda ts: (ts.weekday(), ts.hour)).apply(
    lambda df: df.rolling(52).apply(rolling_av).shift()
)
# . Make arbitrage free (average = 0 for each month)
pu_L_m2h = pu_L_m2h - lb.prices.convert.tseries2tseries(pu_L_m2h, "MS")
# visual check
weeksatsun_function = lambda ts: {5: "Sat", 6: "Sun"}.get(ts.weekday(), "Weekday")
monday_function = lambda ts: ts.floor("D") - pd.Timedelta(ts.weekday(), "D")
cmap = mpl.cm.get_cmap("hsv")  # cyclic map
for daytype, daytypedf in pu_L_m2h.groupby(weeksatsun_function):  # weekday, sat, sun
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
hist[("L", "pu_m")] = lb.prices.convert.bpoframe2tseries(pu_L_m, "H")
hist[("L", "pu_m2h")] = pu_L_m2h
hist[("L", "pu")] = hist.L.pu_m + hist.L.pu_m2h
hist = hist.dropna()

# Price change during step S.
hist[("S", "delta_pu")] = hist.S.pu - hist.L.pu

# B) Volume: at O, and after S.

# Historic values.
i = t_exp.index.intersection(t_act.index)
# . Expected temperatures and offtake volumes.
t_exp = t_exp.loc[i]
hist[("O", "t")] = lb.changefreq_avg(t_exp, "H")
hist[("O", "qo")] = tlp(hist.O.t)
# . Actual temperatures and offtake volumes.
t_act = t_act.loc[i]
hist[("S", "t")] = lb.changefreq_avg(t_act, "H")
hist[("S", "qo")] = tlp(hist.S.t)  # q because duration is 1 h


hist = hist.sort_index(1)

# Keep full years, except 2010 (was extremely cold year)
hist = hist.loc["2001":"2020"]
years = [df for y, df in hist.resample("AS") if y.year != 2010]

# %% INFORMATION TO

cols_sum = [
    ("temprisk", "r_Lt"),
    ("temprisk", "r_St"),
    ("S", "qo"),
    ("O", "qo"),
    ("O", "ro"),
    ("delta_q", "qo"),
    ("O", "qhedge"),
]
cols_avg = [
    ("O", "pu"),
    ("L", "pu"),
    ("S", "pu"),
    ("O", "t"),
    ("S", "t"),
    ("delta_p", "chg01"),
    ("delta_p", "chg12"),
]


#%% SIMULATIONS

monthly_sims = []
yearly_sims = []

for n in range(1000):
    # First, pick a historic year
    hourly = np.random.choice(years).copy().iloc[:8760]

    # Simply overwrite the index with the 2022 index. (Not precise: weekends/weekdays different.)
    hourly.index = current.index

    hourly = pd.merge(current, hourly, left_index=True, right_index=True)

    hourly[("L", "pu_m")] = sim_L()  # replace monthly prices with simulation.
    hourly[("L", "pu")] = hourly.L.pu_m + hourly.L.pu_m2h  # recalculate hourly prices L
    hourly[("S", "pu")] = hourly.L.pu + hourly.S.delta_pu  # ..and hourly prices S
    hourly = hourly.drop([("S", "pu_m"), ("S", "pu_m2h")], axis=1)
    hourly[("O", "qhedge")] = lb.hedge(
        hourly.O.qo, hourly.O.pu, "MS", bpo=True
    )  # hedge
    hourly[("O", "ro")] = hourly.O.qo * hourly.O.po
    hourly[("S", "qspot")] = hourly.S.qo - hourly.O.qhedge

    hourly[("delta_q", "qo")] = hourly.S.qo - hourly.O.qo
    hourly[("delta_p", "chg01")] = hourly.L.pu - hourly.O.po
    hourly[("delta_p", "chg12")] = hourly.S.pu - hourly.L.pu

    hourly[("temprisk", "r_Lt")] = hourly.delta_q.qo * hourly.delta_p.chg01
    hourly[("temprisk", "r_St")] = hourly.delta_q.qo * hourly.delta_p.chg12

    hourly = hourly.sort_index(1)

    # keep only relevant information: the temprisk premiums.
    for freq, records in [("MS", monthly_sims), ("AS", yearly_sims)]:
        df = lb.changefreq_sum(hourly[cols_sum], freq)
        for col in cols_avg:
            df[col] = lb.changefreq_avg(hourly[col], freq)
        df[("delta_p", "chg01")] = df.L.pu - df.O.ro / df.O.qo  # correction
        df[("temprisk", "p_Lt")] = df.temprisk.r_Lt / df.S.qo
        df[("temprisk", "p_St")] = df.temprisk.r_St / df.S.qo
        records.append(df)

yearsims = (
    pd.DataFrame([ys.iloc[0] for ys in yearly_sims]).reset_index().drop(["index"], 1)
)


# %% PLOT: short example timeseries (= last simulation)


daily = lb.changefreq_sum(hourly[cols_sum], "D")
monthly = lb.changefreq_sum(hourly[cols_sum], "MS")
yearly = lb.changefreq_sum(hourly[cols_sum], "AS")
for df in [daily, monthly, yearly]:
    for col in cols_avg:
        df[col] = lb.changefreq_avg(hourly[col], df.index.freq)
    df[("O", "po")] = df.O.ro / df.O.qo
    df[("delta_p", "chg01")] = df.L.pu - df.O.po
    df[("L", "po")] = (df.O.ro + df.temprisk.r_Lt) / df.O.qo
    df[("delta_po", "chg01")] = df.L.po - df.O.po
    df[("S", "po")] = (df.O.ro + df.temprisk.r_Lt + +df.temprisk.r_St) / df.S.qo
    df[("delta_po", "chg12")] = df.S.po - df.L.po
    df[("temprisk", "p_Lt")] = df.temprisk.r_Lt / df.S.qo
    df[("temprisk", "p_St")] = df.temprisk.r_St / df.S.qo


# Filter time section
ts_left = pd.Timestamp("2022-01-01", tz="Europe/Berlin")
ts_right = ts_left + pd.offsets.MonthBegin(12)


def get_filter(ts_left, ts_right):
    def wrapped(df):
        return df[(df.index >= ts_left) & (df.index < ts_right)]

    return wrapped


filtr = get_filter(ts_left, ts_right)
df = filtr(daily)

# Values
r_temprisk_lt = df.temprisk.r_Lt.sum()
r_temprisk_st = df.temprisk.r_St.sum()
q = df.S.qo.sum()

# Formatting
plt.style.use("seaborn")

fig = plt.figure(figsize=(16, 10))
axes = []
# Expected and actual
#  degC
axes.append(plt.subplot2grid((3, 3), (0, 0), fig=fig))
#  MW
axes.append(plt.subplot2grid((3, 3), (1, 0), fig=fig, sharex=axes[0]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (2, 0), fig=fig, sharex=axes[0]))
# PaR
#  scatter
axes.append(plt.subplot2grid((3, 3), (0, 1), fig=fig))
#  MW
axes.append(plt.subplot2grid((3, 3), (1, 1), fig=fig, sharex=axes[0], sharey=axes[1]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (2, 1), fig=fig, sharex=axes[0], sharey=axes[2]))
# Covar
#  scatter
axes.append(plt.subplot2grid((3, 3), (0, 2), fig=fig))
#  MW
axes.append(plt.subplot2grid((3, 3), (1, 2), fig=fig, sharex=axes[0], sharey=axes[1]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (2, 2), fig=fig, sharex=axes[0], sharey=axes[2]))

# Expected and actual
#  Temperatures.
ax = axes[0]
ax.title.set_text("Temperature")
ax.yaxis.label.set_text("degC")
ax.plot(df.O.t, "b-", linewidth=0.5, label="expected")
ax.plot(df.S.t, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake")
ax.yaxis.label.set_text("MWh")
ax.plot(df.O.qo, "b-", linewidth=0.5, label="expected")
ax.plot(df.O.qhedge, "b--", alpha=0.4, linewidth=0.5, label="hedge")
ax.plot(df.S.qo, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df.O.po, c="pink", linewidth=0.5, label="portfolio price at offer")
ax.plot(df.O.pu, c="orange", linewidth=0.5, label="market price at offer")
ax.plot(df.L.pu, c="green", linewidth=0.5, label="market price 1M before delivery")
ax.plot(df.S.pu, "r-", linewidth=0.5, label="spot price")
ax.legend()

# Temprisk - Lt
#  Scatter.
ax = axes[3]
ax.title.set_text("Change in offtake vs long-term change in price")
ax.xaxis.label.set_text("MWh")
ax.yaxis.label.set_text("Eur/MWh")
ax.scatter(
    df.delta_q.qo,
    df.delta_p.chg01,
    c="orange",
    s=10,
    alpha=0.5,
    label=f"r_temprisk_lt   = {r_temprisk_lt/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par_lt = {r_temprisk_lt/q:.2f} Eur/MWh",
)
ax.legend()
#  MWh.
ax = axes[4]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MWh")
ax.plot(df.delta_q.qo, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text(
    "Long-term change in price (+ = increase)\n(between acquisition and M-1)"
)
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df.delta_p.chg01, c="orange", linestyle="-", linewidth=0.5, label="diff")
ax.legend()

# Temprisk - St
#  Scatter.
ax = axes[6]
ax.title.set_text("Change in offtake vs short-term change in price")
ax.xaxis.label.set_text("MWh")
ax.yaxis.label.set_text("Eur/MWh")
ax.scatter(
    df.delta_q.qo,
    df.delta_p.chg12,
    c="green",
    s=10,
    alpha=0.5,
    label=f"r_temprisk_st   = {r_temprisk_st/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par_st = {r_temprisk_st/q:.2f} Eur/MWh",
)
ax.legend()
#  MWh.
ax = axes[7]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MWh")
ax.plot(df.delta_q.qo, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("Short-term change in price (+ = increase)\n(between M-1 and spot)")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df.delta_p.chg12, c="green", linestyle="-", linewidth=0.5, label="diff")
ax.legend()

axes[0].set_xlim(ts_left, ts_right)
fig.tight_layout()


# %% Compare values of the simulations.

df = yearsims


# Formatting
plt.style.use("seaborn")

fig = plt.figure(figsize=(12, 13))
axes = []
#  Price change (LT) vs volume change
axes.append(plt.subplot2grid((3, 2), (0, 0), fig=fig))
#  Price change (ST) vs volume change
axes.append(plt.subplot2grid((3, 2), (0, 1), fig=fig, sharey=axes[0]))
#  PNL temprisk LT
axes.append(plt.subplot2grid((3, 2), (1, 0), fig=fig))
#  PNL temprisk ST
axes.append(plt.subplot2grid((3, 2), (1, 1), fig=fig, sharey=axes[2]))
#  Specific temprisk LT
axes.append(plt.subplot2grid((3, 2), (2, 0), fig=fig))
#  Specific temprisk ST
axes.append(plt.subplot2grid((3, 2), (2, 1), fig=fig, sharex=axes[4], sharey=axes[4]))


# Temprisk - Lt
#  Scatter
ax = axes[0]
ax.title.set_text(
    "Change in offtake vs long-term change in price\n(market price vs pf mix price)"
)
ax.xaxis.label.set_text("MWh")
ax.yaxis.label.set_text("Eur/MWh")
ax.scatter(
    df.delta_q.qo, df.delta_p.chg01, c="orange", s=10, alpha=0.5,
)
ax.legend()
#  Eur.
source_vals = yearsims.temprisk.r_Lt
loc, scale = norm.fit(source_vals)
ax = axes[2]
ax.title.set_text("temprisk costs, long-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.xaxis.label.set_text("simulation number")
ax.bar(
    range(len(source_vals)),
    source_vals / 1000,
    width=1,
    color="orange",
    label=f"long-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
ax.legend()
#  Eur/MWh.
source_vals = yearsims.temprisk.p_Lt.sort_values()
loc, scale = norm.fit(source_vals)
x = source_vals.tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[4]
ax.title.set_text("temprisk specific costs, long-term component")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(
    x,
    cumulative=True,
    color="orange",
    density=True,
    bins=x + [x[-1] + 0.1],
    histtype="step",
    linewidth=1,
)
ax.plot(
    x_fit,
    y_fit,
    color="k",
    linewidth=1,
    label=f"long-term component:\nmean: {loc:.2f} Eur/MWh, std: {scale:.2f} Eur/MWh",
)
ax.legend()

# Temprisk - St
#  Scatter
ax = axes[1]
ax.title.set_text("Change in offtake vs short-term change in price")
ax.xaxis.label.set_text("MWh")
ax.yaxis.label.set_text("Eur/MWh")
ax.scatter(
    df.delta_q.qo, df.delta_p.chg12, c="green", s=10, alpha=0.5,
)
ax.legend()
#  Eur.
source_vals = yearsims.temprisk.r_St
loc, scale = norm.fit(source_vals)
ax = axes[3]
ax.title.set_text("temprisk costs, short-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.xaxis.label.set_text("simulation number")
ax.bar(
    range(len(source_vals)),
    source_vals / 1000,
    width=1,
    color="green",
    label=f"short-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
ax.legend()
#  Eur/MWh.
source_vals = yearsims.temprisk.p_St.sort_values()
loc, scale = norm.fit(source_vals)
x = source_vals.tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[5]
ax.title.set_text("temprisk specific costs, short-term component")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(
    x,
    cumulative=True,
    color="green",
    density=True,
    bins=x + [x[-1] + 0.1],
    histtype="step",
    linewidth=1,
)
ax.plot(
    x_fit,
    y_fit,
    color="k",
    linewidth=1,
    label=f"short-term component:\nmean: {loc:.2f} Eur/MWh, std: {scale:.2f} Eur/MWh",
)
ax.legend()

fig.tight_layout()


# %% Just PNL and premiums.

# Formatting
plt.style.use("seaborn")
# positive formatting (costs>0), negative formatting (costs<0)

fig = plt.figure(figsize=(12, 9))
axes = []
#  PNL temprisk LT
axes.append(plt.subplot2grid((2, 2), (0, 0), fig=fig))
#  PNL temprisk ST
axes.append(plt.subplot2grid((2, 2), (0, 1), fig=fig, sharey=axes[0]))
#  Specific temprisk LT
axes.append(plt.subplot2grid((2, 2), (1, 0), fig=fig))
#  Specific temprisk ST
axes.append(plt.subplot2grid((2, 2), (1, 1), fig=fig, sharex=axes[2], sharey=axes[2]))


# Temprisk - Lt
#  Eur.
source_vals = yearsims.temprisk.r_Lt
loc, scale = norm.fit(source_vals)
ax = axes[0]
ax.title.set_text("temprisk costs, long-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.xaxis.label.set_text("simulation number")
ax.bar(
    range(len(source_vals)),
    source_vals / 1000,
    width=1,
    color="orange",
    label=f"long-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
ax.legend()
#  Eur/MWh.
source_vals = yearsims.temprisk.p_Lt.sort_values()
loc, scale = norm.fit(source_vals)
x = source_vals.tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[2]
ax.title.set_text("temprisk specific costs, long-term component")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(
    x,
    cumulative=True,
    color="orange",
    density=True,
    bins=x + [x[-1] + 0.1],
    histtype="step",
    linewidth=1,
)
ax.plot(
    x_fit,
    y_fit,
    color="k",
    linewidth=1,
    label=f"long-term component:\nmean: {loc:.2f} Eur/MWh, std: {scale:.2f} Eur/MWh",
)
ax.legend()

# Temprisk - St
#  Eur.
source_vals = yearsims.temprisk.r_St
loc, scale = norm.fit(source_vals)
ax = axes[1]
ax.title.set_text("temprisk costs, short-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.xaxis.label.set_text("simulation number")
ax.bar(
    range(len(source_vals)),
    source_vals / 1000,
    width=1,
    color="green",
    label=f"short-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
ax.legend()
#  Eur/MWh.
source_vals = yearsims.temprisk.p_St.sort_values()
loc, scale = norm.fit(source_vals)
x = source_vals.tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[3]
ax.title.set_text("temprisk specific costs, short-term component")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(
    x,
    cumulative=True,
    color="green",
    density=True,
    bins=x + [x[-1] + 0.1],
    histtype="step",
    linewidth=1,
)
ax.plot(
    x_fit,
    y_fit,
    color="k",
    linewidth=1,
    label=f"short-term component:\nmean: {loc:.2f} Eur/MWh, std: {scale:.2f} Eur/MWh",
)
ax.legend()

fig.tight_layout()


# %%
