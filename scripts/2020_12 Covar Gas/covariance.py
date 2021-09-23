"""
Module to calculate the historic covariance between spot price and tlp consumption.
(for gas)

Three time points are considered:
    0: customer is acquired and hedged
    1: few weeks before start of delivery month
    2: delivery
(0) and (1) share the temperature (expectation), and with it the offtake volume
(expectation). As no more volume is bought between (0) and (1), they also share
the spot volume (expectation).

Process:

(a) valuation
temperature -> (with tlp:) offtake -> (with prices:) value of offtake.

This is done at all 3 points in time.
For (2), the spot prices can be used with daily frequency.
For (1), month prices are used, plus historic month-to-day profile to get daily prices.
For (0), year prices are used, plus historic year-to-month profile and month-to-day
profile to get daily prices.

(b) comparison
. only volume change is between (1) and (2).
. PaR has 2 components:
.. "lt" = price change (0->1) times volume change --> long term effect due to change in
general price level
.. "st" = price change (1->2) times volume change --> short term effect, including (but
not only) correlation


Variations:
  1. Expected temperature = monthly average, i.e., same temperature for each day of a
  given month.
  2. Expected temperature = 'structured' temperature, i.e., already including some
  temperature path.


Changes 2021-02-17:
. Added long-term price change.
"""

# %% SETUP

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from scipy.stats import norm


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
gas_btb_kw = {
    "contingent": 1_790_000,
    "btb": 440_000,
    "rlm": 90_000,
}  # not offtake in MWh
tlp = lb.tlp.gas.D14(kw=gas_btb_kw["contingent"])
lb.tlp.plot.vs_t(tlp)  # quick visual check


# %% ACTUAL

# Actual spot prices.
act[("spot", "p")] = lb.prices.montel.gas_spot()
act.spot.p = lb.fill_gaps(act.spot.p, 5)
# (split into month average and month-to-day deviations)
act[("spot_m", "p")] = act.spot.p.resample("MS").transform(np.mean)
act[("spot_m2d", "p")] = act.spot.p - act.spot_m.p

# Actual temperature.
t = lb.tmpr.hist.tmpr()
t_act = t.wavg(weights.values, axis=1)
act[("envir", "t")] = t_act  # TODO: use new/own resample function

# Trim on both sides to discard rows for which temperature and price are missing.
act = act.dropna().resample("D").asfreq()

# Actual offtake.
act[("offtake", "w")] = tlp(act.envir.t)


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
exp[("offtake", "w")] = tlp(exp.envir.t)


# %% EXPECTED, PRICES, 1: ~ 3 weeks before delivery month: no temperature influence in price.


def exp_price(df):
    cols = ["p", "ts_left_trade"]  # columns to keep
    # keep suitable
    df = df[df["anticipation"] > datetime.timedelta(30)]
    df = df.dropna()
    # find one value
    if not df.empty:
        df = df.sort_values("anticipation")  # sort suitable.
        df = df.reset_index("ts_left_trade")  # to get 'ts_left_trade' in columns.
        return pd.Series(df[cols].iloc[0])  # keep one.
    return pd.Series([], dtype=pd.Float32Dtype)


# Expected spot prices.
# . Use futures prices to calculate the expected average price level.
frontmonth = lb.prices.montel.gas_futures("m")
# . Prices before temperature influence: at least 21 days before delivery start.
p_exp1_m = frontmonth.groupby("ts_left").apply(exp_price).dropna()

# . Use actual spot prices to calculate expected M2D profile: mean but without extremes
rolling_av = lambda nums: sum(np.sort(nums)[3:-3]) / (len(nums) - 6)
p_act_m2d = act.spot_m2d.p
# .. for each day, find average m2d value for past 52 same values.
p_exp1_m2d = p_act_m2d.groupby(p_act_m2d.index.weekday).apply(
    lambda df: df.rolling(52).apply(rolling_av).shift()
)
# . Make arbitrage free.
p_exp1_m2d = p_exp1_m2d - p_exp1_m2d.groupby(lambda ts: (ts.year, ts.month)).transform(
    np.mean
)
# visual check
p_exp1_m2d.dropna().groupby(p_exp1_m2d.dropna().index.weekday).plot(legend=True)

# Expected prices.
exp1[("m", "p")] = p_exp1_m.p.resample("D").ffill()
exp1[("m2d", "p")] = p_exp1_m2d
exp1[("pfc", "p")] = exp1.m.p + exp1.m2d.p


# %% EXPECTED 0: Price at time of acquisition


def factor_p_a2m(p_m: pd.Series) -> pd.Series:
    """Calculate the factors to go from year to month prices.

    Args:
        p_m: pd.Series with month prices.
    """
    # . keep only full years and keep only the prices.
    first = p_m.index[0] + pd.offsets.YearBegin(0)
    last = p_m.index[-1] + pd.offsets.YearBegin(1) + pd.offsets.YearBegin(-1)
    mask = (p_m.index >= first) & (p_m.index < last)
    # . each month's price as fraction of yearly average.
    factors = p_m[mask].resample("AS").transform(lambda df: df / df.mean())
    factors.index = pd.MultiIndex.from_tuples(
        [(ts.year, ts.month) for ts in factors.index], names=("year", "month")
    )
    factors = factors.unstack(0)
    # . visual check: curve for each month
    factors.plot(cmap="jet")
    # . average factor for each month
    return factors.mean(axis=1)


def price_at_acquisition(index, monthpricefactors, lead_time=1.5):
    """Calculate month price at time of acquisition.

    Args:
        index: timestamps of delivery months of interest.
        monthpricefactors (pd.Series): ratio between month and year price for each of 12 months.
        lead_time (float): how far before delivery price is wanted. Defaults to 1.5.
    """
    # Find year prices.
    # . when do we want to know which price
    p_exp0_m = pd.DataFrame(index=index)
    p_exp0_m["ts_trade"] = p_exp0_m.index + pd.offsets.MonthBegin(-int(lead_time * 12))
    p_exp0_m["period_start"] = p_exp0_m.index.year - p_exp0_m.ts_trade.dt.year
    rang = (p_exp0_m["period_start"].min(), p_exp0_m["period_start"].max() + 1)
    # . find that price
    yearprices = {ps: lb.prices.montel.gas_futures("a", ps) for ps in range(*rang)}

    def yearprice(ts_trade, period_start):
        cols = ["p", "ts_left_trade"]  # columns to keep
        df = yearprices[period_start]
        # keep suitable
        df = df[df.index >= ts_trade]
        df = df[df.index < ts_trade + pd.offsets.MonthBegin(1)]
        df = df.dropna().sort_index()
        # find one value
        if not df.empty:
            df = df.reset_index("ts_left_trade")  # to get 'ts_left_trade' in columns.
            return pd.Series(df[cols].iloc[0])  # keep one.
        return pd.Series([], dtype=pd.Float32Dtype)

    # .. yearprice
    p_exp0_m[["p_year", "ts_left_trade"]] = p_exp0_m.apply(
        lambda row: yearprice(row["ts_trade"], row["period_start"]), axis=1
    )
    # .. monthprice
    p_exp0_m["p_month"] = p_exp0_m.apply(
        lambda row: row["p_year"] * monthpricefactors[row.name.month], axis=1
    )
    return p_exp0_m


monthpricefactors = factor_p_a2m(p_exp1_m.p)
prices_at_various_leadtimes = [
    price_at_acquisition(p_exp1_m.index, monthpricefactors, lead_time)
    for lead_time in np.linspace(0.5, 3.5, 5)
]
p0_m = pd.DataFrame([p["p_month"] for p in prices_at_various_leadtimes]).mean().T

# Turn into forward curve.
# (using previously calculated m2d curve. Not entirely correct, because uses data that
# was not available yet. But m2d profile expected to be quite stable)
exp0[("m2d", "p")] = exp1[("m2d", "p")]
exp0[("m", "p")] = p0_m.resample("D").ffill()
exp0[("pfc", "p")] = exp0.m2d.p + exp0.m.p


# %% Derived quantities

# Combine.
daily = (
    pd.concat([exp, exp0, exp1, act], axis=1, keys=["exp", "exp0", "exp1", "act"])
    .dropna()
    .resample("D")
    .asfreq()  # drop leading/trailing nanvalues but keep frequency.
)

# Hedge. --> Volume hedge.
daily[("pf", "hedge", "w")] = lb.hedge(daily.exp.offtake.w, how="vol")
daily.pf.hedge.w = daily.pf.hedge.w.apply(lambda v: 20 * (v // 20))  # only buy full MW

# Expected spot quantities.
daily[("exp", "spot", "w")] = daily.exp.offtake.w - daily.pf.hedge.w
# check: spot volume should add to 0 for each given month (only if hedge not rounded)
# assert ((daily.exp.spot.w * daily.duration).resample("MS").sum().abs() < 0.1).all()

# Actual spot quantities.
daily[("act", "spot", "w")] = daily.act.offtake.w - daily.pf.hedge.w

# Difference.
daily[("delta", "offtake", "w")] = daily.act.offtake.w - daily.exp.offtake.w
daily[("delta", "chg01", "p")] = daily.exp1.pfc.p - daily.exp0.pfc.p
daily[("delta", "chg12", "p")] = daily.act.spot.p - daily.exp1.pfc.p

daily[("par", "Lt", "r")] = daily.delta.offtake.w * daily.duration * daily.delta.chg01.p
daily[("par", "St", "r")] = daily.delta.offtake.w * daily.duration * daily.delta.chg12.p

daily = daily.sort_index(1)


# %% Aggregations

# only keep full year
start = daily.index[0] + pd.offsets.YearBegin(1)
daily = daily[daily.index >= "2013"]  # TODO: check why missing values before 2013
# aggregate
monthly = daily.resample("MS").mean()  # not exactly accurate
yearly = daily.resample("AS").mean()
for df in [monthly, yearly]:
    # Correction: here sum is needed, not mean
    df[("par", "Lt", "r")] = daily.par.Lt.r.resample(df.index.freq).sum()
    df[("par", "St", "r")] = daily.par.St.r.resample(df.index.freq).sum()
    # New information
    df[("par", "Lt", "p")] = df.par.Lt.r / (df.act.offtake.w * df.duration)
    df[("par", "St", "p")] = df.par.St.r / (df.act.offtake.w * df.duration)


# %% PLOT: short example timeseries

# Filter time section
ts_left = pd.Timestamp("2015-08-01", tz="Europe/Berlin")
ts_right = ts_left + pd.offsets.MonthBegin(6)


def get_filter(ts_left, ts_right):
    def wrapped(df):
        return df[(df.index >= ts_left) & (df.index < ts_right)]

    return wrapped


filtr = get_filter(ts_left, ts_right)
df = filtr(daily)

# Values
r_par_lt = df.par.Lt.r.sum()
r_par_st = df.par.St.r.sum()
q = (df.act.offtake.w * df.duration).sum()

# Formatting
plt.style.use("seaborn")
# positive formatting (costs>0), negative formatting (costs<0)
pos = {"color": "black", "alpha": 0.1}

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
ax.plot(df.exp.envir.t, "b-", linewidth=0.5, label="expected")
ax.plot(df.act.envir.t, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake")
ax.yaxis.label.set_text("MW")
ax.plot(df.exp.offtake.w, "b-", linewidth=0.5, label="expected")
ax.plot(df.pf.hedge.w, "b--", alpha=0.4, linewidth=0.5, label="hedge")
ax.plot(df.act.offtake.w, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df.exp0.pfc.p, c="orange", linewidth=0.5, label="at acquisition")
ax.plot(df.exp1.pfc.p, c="green", linewidth=0.5, label="1M before delivery")
ax.plot(df.act.spot.p, "r-", linewidth=0.5, label="actual")
ax.legend()

# PaR - Lt
#  Times with positive contribution (costs > 0).
times = ((df.par.Lt.r > 0).shift() - (df.par.Lt.r > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[3]
ax.title.set_text("Change in offtake vs long-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(*df.delta.offtake.w, 0), 0), min(*df.delta.chg01.p, 0), 0, **pos)
ax.fill_between((max(*df.delta.offtake.w, 0), 0), max(*df.delta.chg01.p, 0), 0, **pos)
ax.scatter(
    df.delta.offtake.w,
    df.delta.chg01.p,
    c="orange",
    s=10,
    alpha=0.5,
    label=f"r_par_lt   = {r_par_lt/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par_lt = {r_par_lt/q:.2f} Eur/MWh",
)
ax.legend()
#  MW.
ax = axes[4]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.offtake.w, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text(
    "Long-term change in price (+ = increase)\n(between acquisition and M-1)"
)
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.chg01.p, c="orange", linestyle="-", linewidth=0.5, label="diff")
ax.legend()

# PaR - St
#  Times with positive contribution (costs > 0).
times = ((df.par.St.r > 0).shift() - (df.par.St.r > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[6]
ax.title.set_text("Change in offtake vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(*df.delta.offtake.w, 0), 0), min(*df.delta.chg12.p, 0), 0, **pos)
ax.fill_between((max(*df.delta.offtake.w, 0), 0), max(*df.delta.chg12.p, 0), 0, **pos)
ax.scatter(
    df.delta.offtake.w,
    df.delta.chg12.p,
    c="green",
    s=10,
    alpha=0.5,
    label=f"r_par_st   = {r_par_st/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par_st = {r_par_st/q:.2f} Eur/MWh",
)
ax.legend()
#  MW.
ax = axes[7]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.offtake.w, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("Short-term change in price (+ = increase)\n(between M-1 and spot)")
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.chg12.p, c="green", linestyle="-", linewidth=0.5, label="diff")
ax.legend()

axes[0].set_xlim(ts_left, ts_right)
fig.tight_layout()


# %% PLOT: per month or year

# Formatting
plt.style.use("seaborn")
# positive formatting (costs>0), negative formatting (costs<0)
pos = {"color": "black", "alpha": 0.1}

fig = plt.figure(figsize=(16, 10))
axes = []
# Expected and actual.
#  degC.
axes.append(plt.subplot2grid((3, 3), (0, 0), fig=fig))
#  MWh.
axes.append(plt.subplot2grid((3, 3), (1, 0), fig=fig, sharex=axes[0]))
#  Eur/MWh.
axes.append(plt.subplot2grid((3, 3), (2, 0), fig=fig, sharex=axes[0]))
# PaR.
#  distribution.
axes.append(plt.subplot2grid((3, 3), (2, 1), fig=fig))
#  Eur.
axes.append(plt.subplot2grid((3, 3), (0, 1), fig=fig, sharex=axes[0]))
#  Eur/MWh.
axes.append(plt.subplot2grid((3, 3), (1, 1), fig=fig, sharex=axes[0]))
# Covar.
#  distribution.
axes.append(plt.subplot2grid((3, 3), (2, 2), fig=fig, sharex=axes[3]))
#  Eur.
axes.append(plt.subplot2grid((3, 3), (0, 2), fig=fig, sharex=axes[0], sharey=axes[4]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (1, 2), fig=fig, sharex=axes[0], sharey=axes[5]))

# Expected and actual
#  Temperatures.
ax = axes[0]
ax.title.set_text("Temperature, monthly average")
ax.yaxis.label.set_text("degC")
ax.plot(monthly.exp.envir.t, "b-", linewidth=1, label="expected")
ax.plot(monthly.act.envir.t, "r-", linewidth=1, label="actual")
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Change in offtake, per month")
ax.yaxis.label.set_text("MWh")
ax.plot(
    (monthly.act.offtake.w - monthly.exp.offtake.w) * monthly.duration,
    c="purple",
    linewidth=1,
    label="change (+ = more offtake)",
)

ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Month price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(monthly.exp0.pfc.p, c="orange", linewidth=1, label="at acquisition")
ax.plot(monthly.exp1.pfc.p, c="green", linewidth=1, label="1M before delivery")
ax.plot(monthly.act.spot.p, "r-", linewidth=1, label="actual")
ax.legend()

# PaR - Lt
#  Distribution.
source_vals = yearly.par.Lt.p
loc, scale = norm.fit(source_vals)
x = source_vals.sort_values().tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[3]
ax.title.set_text("par premium, long-term component\n(+ = additional costs), per year")
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
    label=f"Fit:\nmean: {loc:.2f}, std: {scale:.2f}",
)
ax.legend()
#  Eur.
ax = axes[4]
ax.title.set_text("par revenue, long-term component\n(+ = additional costs), per month")
ax.yaxis.label.set_text("Eur")
ax.plot(monthly.par.Lt.r, color="orange", linestyle="-", linewidth=1, label="r_par_lt")
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text("par premium, long-term component\n(+ = additional costs), per year")
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(yearly.index, yearly.par.Lt.p, width=365 * 0.9, color="orange", label="p_par_lt")
ax.legend()

# Par - St
#  Distribution.
source_vals = yearly.par.St.p
loc, scale = norm.fit(source_vals)
x = source_vals.sort_values().tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[6]
ax.title.set_text("par premium, short-term component\n(+ = additional costs), per year")
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
    label=f"Fit:\nmean: {loc:.2f}, std: {scale:.2f}",
)
ax.legend()
#  Eur.
ax = axes[7]
ax.title.set_text(
    "par revenue, short-term component\n(+ = additional costs), per month"
)
ax.yaxis.label.set_text("Eur")
ax.plot(monthly.par.St.r, color="green", linestyle="-", linewidth=1, label="r_par_st")
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("par premium, short-term component\n(+ = additional costs), per year")
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(yearly.index, yearly.par.St.p, width=365 * 0.9, color="green", label="p_par_st")
ax.legend()

fig.tight_layout()

# %% Just PNL

# Formatting
plt.style.use("seaborn")
# positive formatting (costs>0), negative formatting (costs<0)

fig = plt.figure(figsize=(8, 4))
axes = []
# Expected and actual.
#  PNL Par LT
axes.append(plt.subplot2grid((1, 2), (0, 0), fig=fig))
#  PNL Par ST
axes.append(plt.subplot2grid((1, 2), (0, 1), fig=fig, sharex=axes[0], sharey=axes[0]))

# PaR - Lt
#  Distribution.
source_vals = yearly.par.Lt.r
loc, scale = norm.fit(source_vals)
#  Eur.
ax = axes[0]
ax.title.set_text("par costs, long-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.bar(
    yearly.index,
    yearly.par.Lt.r / 1000,
    width=365 * 0.9,
    color="orange",
    label=f"long-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
# ax.legend()

# Par - St
#  Distribution.
source_vals = yearly.par.St.r
loc, scale = norm.fit(source_vals)
#  Eur.
ax = axes[1]
ax.title.set_text("par costs, short-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.bar(
    yearly.index,
    yearly.par.St.r / 1000,
    width=365 * 0.9,
    color="green",
    label=f"short-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
# ax.legend()

fig.tight_layout()


# %% Scatter only.

df = monthly
fig = plt.figure(figsize=(10, 5))
axes = []
# Expected and actual
#  degC
axes.append(plt.subplot2grid((1, 1), (0, 0), fig=fig))
#  MW
ax = axes[0]
ax.title.set_text("Change in offtake vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(*df.delta.offtake.w, 0), 0), min(*df.delta.chg12.p, 0), 0, **pos)
ax.fill_between((max(*df.delta.offtake.w, 0), 0), max(*df.delta.chg12.p, 0), 0, **pos)
ax.scatter(
    df.delta.offtake.w,
    df.delta.chg12.p,
    c="green",
    s=10,
    alpha=0.9,
    label=f"r_par_st   = {r_par_st/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par_st = {r_par_st/q:.2f} Eur/MWh",
)
# ax.legend()

# %% CALCULATE: conditional value at risk

p_par_pricedin = 1.12
p_covar_pricedin = 1.56
yearly[("delta", "par", "r")] = yearly.act.par.r - yearly.act.offtake.q * p_par_pricedin
yearly[("delta", "covar", "r")] = (
    yearly.act.covar.r - yearly.act.offtake.q * p_covar_pricedin
)


# %%

u = daily[[("exp", "offtake", "w"), ("act", "offtake", "w"), ("delta", "offtake", "w")]]
ax = u.delta.offtake.w.abs().hist(
    bins=1000, cumulative=True, density=True, linewidth=1, histtype="step"
)
ax.set_xlabel("deviation [MW]")
ax.set_ylabel("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

u.delta.offtake.w.mean()
