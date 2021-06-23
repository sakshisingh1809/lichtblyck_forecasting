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

stepS = pd.DataFrame(columns=[[], []])  # 2-level columns
exp = pd.DataFrame(columns=[[], []])  # 2-level columns
stepL = pd.DataFrame(columns=[[], []])  # 2-level columns
step0 = pd.DataFrame(columns=[[], []])  # 2-level columns


# %% PRICES

# Month and year prices
pu_fut = {prod: lb.prices.montel.power_futures(prod) for prod in ["a", "q", "m"]}


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
tlp["gas"] = lb.tlp.gas.D14(kw=1000000)  # not used, just for comtempriskison
# quick visual check
for pr in tlp.values():
    lb.tlp.plot.vs_time(pr)
    try:
        lb.tlp.plot.vs_t(pr)  # for some reason, fails.
    except:
        pass

tlp_touse = lambda ts: tlp["rh"](ts) + tlp["hp"](ts)

# %% Spot

# Actual spot prices.
stepS[("pu", "p")] = lb.prices.montel.power_spot()
stepS.pu.p = lb.tools.fill_gaps(stepS.pu.p, 5)
# (split into (1) month average base and peak, and (2) month-to-hour deviations)
stepS[("pu_m", "p")] = lb.prices.convert.tseries2tseries(stepS.pu.p, "MS")
stepS[("pu_m2h", "p")] = stepS.pu.p - stepS.pu_m.p
# visual check
stepS[["pu", "pu_m"]].plot()
stepS[["pu", "pu_m", "pu_m2h"]].loc["2020-11":"2020-12", :].plot()

# Actual temperature.
t = lb.historic.tmpr()
t = lb.historic.fill_gaps(t)
t_act = t.wavg(weights.values, axis=1)
stepS[("envir", "t")] = lb.core.functions.changefreq_avg(t_act, "H")

# Trim on both sides to discard rows for which temperature and price are missing.
stepS = stepS.dropna().resample("H").asfreq()

# Actual offtake.
stepS[("qo", "w")] = tlp_touse(stepS.envir.t)


# %% EXPECTED, OFFTAKE (General, at timepoint S and timepoint L)

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
step0[("envir", "t")] = lb.changefreq_avg(t_exp, "H")
step0[("qo", "w")] = tlp_touse(step0.envir.t)


# %% EXPECTED, PRICES, L: ~ 3 weeks before delivery month: no temperature influence in price.


def price_after_stepL(df):
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


# Market prices at end of step L.
# . Use futures prices to calculate the expected average price level.
# . Prices before temperature influence: about 21 days before delivery start.
pu_L_m = pu_fut["m"].groupby(level=0).apply(price_after_stepL).dropna().unstack()
print(pu_L_m[["ts_left_trade", "anticipation"]].describe())  # quick check
pu_L_m = lb.tools.set_ts_index(pu_L_m.drop(columns=["ts_left_trade", "anticipation"]))
assert pu_L_m.isna().any().any() == False  # each month has a price
pu_L_m["po_spread"] = pu_L_m.p_peak - pu_L_m.p_offpeak
# quick visual check
pd.DataFrame({**pu_L_m, "po_spread": pu_L_m.p_peak - pu_L_m.p_offpeak}).plot()

# . Use actual spot prices to calculate expected M2H profile: mean but without extremes
rolling_av = lambda nums: sum(np.sort(nums)[5:-15]) / (len(nums) - 20)
# .. for each hour and weekday, find average m2h value for past 52 values.
pu_L_m2h = stepS.pu_m2h.p.groupby(lambda ts: (ts.weekday(), ts.hour)).apply(
    lambda df: df.rolling(52).apply(rolling_av).shift()
)
# . Make arbitrage free (average = 0)
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
stepL[("pu_m", "p")] = lb.prices.convert.bpoframe2tseries(pu_L_m, "H")
stepL[("pu_m2h", "p")] = pu_L_m2h
stepL[("pu", "p")] = stepL.pu_m.p + stepL.pu_m2h.p
stepL = stepL.dropna()


# %% EXPECTED, 0: Price at time of acquisition


def factor(p_m: pd.DataFrame, freq="QS") -> pd.Series:
    """Calculate the factors to go from quarter or year prices to month prices.

    tempriskameters
    ----------
        p_m : pd.DataFrame
            bpoframe (pd.Dataframe with peak and offpeak columns) with monthly prices.
        freq : str
            frequency at which to calculate the factors; one of {'QS', 'AS'}
    
    Returns
    -------
    pd.Series
        with month (1-12) as index and factor (to multiply quarter or year price with) as values.
    """
    # Keep only full (quarters or years).
    if lb.floor(p_m.index[0], 0, freq) == p_m.index[0]:
        first = p_m.index[0]
    else:
        first = lb.floor(p_m.index[0], 1, freq)
    if lb.floor(p_m.index[-1], 1, freq) == lb.floor(p_m.index[-1], 1):
        last = lb.floor(p_m.index[-1], 1, freq)
    else:
        last = lb.floor(p_m.index[-1], 0, freq)
    mask = (p_m.index >= first) & (p_m.index < last)
    # Each month's price as fraction of quarterly average.
    factors = (
        p_m[mask][["p_peak", "p_offpeak"]]
        .resample(freq)
        .transform(lambda df: df / lb.changefreq_avg(df, freq).iloc[0])
    )
    factors.index = pd.MultiIndex.from_tuples(
        [(ts.year, ts.month) for ts in factors.index], names=("year", "month")
    )
    factors = factors.unstack(0)
    # . visual check: curve for each month
    fig, axes = plt.subplots(figsize=(20, 10), ncols=2)
    fig.suptitle(f"Price factors throughout the year ({freq})")
    for ax, daytime in zip(axes, ["offpeak", "peak"]):
        factors[f"p_{daytime}"].plot(ax=ax, cmap="jet")
        ax.set_title(daytime)
        ax.legend()
    # . average factor for each month
    return factors.groupby(level=0, axis=1).apply(lambda df: df.mean(axis=1))


def price_at_acquisition(
    index,
    year2monthpricefactors,
    quarter2monthpricefactors,
    anticipation=pd.Timedelta(days=365 * 1.5),
):
    """Calculate month price at time of acquisition.

    Args:
        index: timestamps of delivery months of interest.
        year2monthpricefactors (pd.Series): ratio between month and year price for each of 12 months.
        quarter2monthpricefactors (pd.Series): ratio between month and quarter price for each of 12 months.
        anticipation: how far before start of delivery period, the price is wanted. Defaults to 1.5 years.
    """
    # For which month do we want to know price.
    pu_0_m = pd.DataFrame(index=index)

    cols = ["p_peak", "p_offpeak", "ts_left_trade"]  # columns to keep

    def monthprice(ts_left):
        # When do we want to know the price.
        ts_trade = ts_left - anticipation
        ts_trade_min = ts_trade + pd.Timedelta(days=-7)
        ts_trade_max = ts_trade + pd.Timedelta(days=3)
        # Find price directly in month futures.
        df = pu_fut["m"]
        mask = (
            (df.index.get_level_values("ts_left") == ts_left)
            & (df.index.get_level_values("ts_left_trade") >= ts_trade_min)
            & (df.index.get_level_values("ts_left_trade") <= ts_trade_max)
        )  # keep suitable
        df = df[mask].dropna()
        # find one value
        if not df.empty:
            df = df.sort_values("anticipation")  # sort suitable.
            df = df.reset_index()  # to get 'ts_left_trade' in columns.
            s = pd.Series(df[cols].iloc[0])  # keep one.
            return s
        # If not found: find price in quarter or year futures.
        for freq, df, pricefactors in (
            ("QS", pu_fut["q"], quarter2monthpricefactors),
            ("AS", pu_fut["a"], year2monthpricefactors),
        ):
            mask = (
                (df.index.get_level_values("ts_left") == lb.floor(ts_left, 0, freq))
                & (df.index.get_level_values("ts_left_trade") >= ts_trade_min)
                & (df.index.get_level_values("ts_left_trade") <= ts_trade_max)
            )  # keep suitable
            df = df[mask].dropna()
            # find one value
            if not df.empty:
                df = df.sort_values("anticipation")  # sort suitable.
                df = df.reset_index()  # to get 'ts_left_trade' in columns.
                s = pd.Series(df[cols].iloc[0])  # year or quarter price
                for key in ["p_peak", "p_offpeak"]:
                    s[key] *= pricefactors.loc[ts_left.month, key]
                return s  # month price
        # If still not found: return empty series
        s = pd.Series([None] * len(cols), cols)
        return s

    pu_0_m[cols] = pd.DataFrame.from_dict({ts: monthprice(ts) for ts in pu_0_m.index}).T
    return pu_0_m


a2m = factor(pu_L_m, "AS")
q2m = factor(pu_L_m, "QS")
prices_at_various_anticipations = [
    price_at_acquisition(pu_L_m.index, a2m, q2m, pd.Timedelta(days=365 * lead))
    for lead in np.linspace(0.5, 3.5, 5)
]

# If acquisition 3 months before delivery start, and 12 months delivery duration:
# --> 9 months average anticipation
pu_0_m = price_at_acquisition(pu_L_m.index, a2m, q2m, pd.Timedelta(days=270 + 365))
pu_0_m = pu_0_m.drop(columns="ts_left_trade")
# Quick visual check.
pu_0_m.plot()

# Turn into forward curve.
# (using previously calculated m2h curve. Not entirely correct, because uses data that
# was not available yet. But m2h profile expected to be quite stable)
step0[("pu_m2h", "p")] = stepL[("pu_m2h", "p")]
step0[("pu_m", "p")] = lb.prices.convert.bpoframe2tseries(pu_0_m, "H")
step0[("pu", "p")] = step0.pu_m.p + step0.pu_m2h.p

# %% Derived quantities

# Combine.
dfs = {
    key: df.drop(columns=["pu_m", "pu_m2h"])
    for key, df in [("0", step0), ("L", stepL), ("S", stepS)]
}
hourly = (
    pd.concat(dfs, axis=1)
    .dropna()
    .resample("H")
    .asfreq()  # drop leading/trailing nanvalues but keep frequency.
)

# Hedge. --> Volume hedge.
hourly[("0", "qs", "w")] = lb.hedge(hourly["0"].qo.w, freq="AS", how="vol")
hourly[("0", "qs", "w")] = hourly["0"].qs.w.apply(lambda v: v // 1)  # only buy full MW

# Expected spot quantities.
hourly[("0", "spot", "w")] = hourly["0"].qo.w - hourly["0"].qs.w
# check: spot volume should add to 0 for each given month (only if hedge not rounded)
# assert ((daily.exp.spot.w * daily.duration).resample("MS").sum().abs() < 0.1).all()

# Actual spot quantities.
hourly[("S", "spot", "w")] = hourly.S.qo.w - hourly["0"].qs.w

# Difference.
hourly[("S", "delta_qo", "w")] = hourly.S.qo.w - hourly["0"].qo.w
hourly[("L", "delta_pu", "p")] = hourly.L.pu.p - hourly["0"].pu.p
hourly[("S", "delta_pu", "p")] = hourly.S.pu.p - hourly.L.pu.p

hourly[("temprisk", "Lt", "r")] = (
    hourly.S.delta_qo.w * hourly.duration * hourly.L.delta_pu.p
)
hourly[("temprisk", "St", "r")] = (
    hourly.S.delta_qo.w * hourly.duration * hourly.S.delta_pu.p
)

hourly = hourly.sort_index(1)

# %% Aggregations

# only keep full years
hourly = hourly[(hourly.index >= "2004") & (hourly.index < "2021")]
# aggregate
daily = lb.changefreq_avg(hourly, "D")
monthly = lb.changefreq_avg(hourly, "MS")
yearly = lb.changefreq_avg(hourly, "AS")
for df in [daily, monthly, yearly]:
    for period in ("Lt", "St"):
        column = ("temprisk", period, "r")
        # Correction: here sum is needed, not mean
        df[column] = lb.changefreq_sum(hourly[column], df.index.freq)
        # New information
        df[("temprisk", period, "p")] = df[column] / (df.S.qo.w * df.duration)

# %% Temprisk distribution.

temprisk_dist = pd.DataFrame([], columns=["loc", "scale"])
# Shortterm.
temprisk_dist.loc["St"] = norm.fit(yearly.temprisk.St.p)
# Longterm.
temprisk_dist.loc["Lt"] = norm.fit(yearly.temprisk.Lt.p)

# %% PLOT: short example timeseries

# Filter time section
ts_left = pd.Timestamp("2015-02-01", tz="Europe/Berlin")
ts_right = lb.floor(ts_left, 1, "MS")


def get_filter(ts_left, ts_right):
    def wrapped(df):
        return df[(df.index >= ts_left) & (df.index < ts_right)]

    return wrapped


filtr = get_filter(ts_left, ts_right)
df = filtr(hourly)

# Values
r_temprisk_lt = df.temprisk.Lt.r.sum()
r_temprisk_st = df.temprisk.St.r.sum()
q = (df.S.qo.w * df.duration).sum()

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
# Temprisk
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
ax.plot(df["0"].envir.t, "b-", linewidth=0.5, label="expected")
ax.plot(df.S.envir.t, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake")
ax.yaxis.label.set_text("MW")
ax.plot(df["0"].qo.w, "b-", linewidth=0.5, label="expected")
ax.plot(df["0"].qs.w, "b--", alpha=0.4, linewidth=0.5, label="hedge")
ax.plot(df.S.qo.w, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df["0"].pu.p, c="orange", linewidth=0.5, label="at acquisition")
ax.plot(df.L.pu.p, c="green", linewidth=0.5, label="1M before delivery")
ax.plot(df.S.pu.p, "r-", linewidth=0.5, label="actual")
ax.legend()

# Temprisk - Lt
#  Times with positive contribution (costs > 0).
times = ((df.temprisk.Lt.r > 0).shift() - (df.temprisk.Lt.r > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[3]
ax.title.set_text("Change in offtake vs long-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(*df.S.delta_qo.w, 0), 0), min(*df.L.delta_pu.p, 0), 0, **pos)
ax.fill_between((max(*df.S.delta_qo.w, 0), 0), max(*df.L.delta_pu.p, 0), 0, **pos)
ax.scatter(
    df.S.delta_qo.w,
    df.L.delta_pu.p,
    c="orange",
    s=10,
    alpha=0.5,
    label=f"r_temprisk_lt   = {r_temprisk_lt/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_temprisk_lt = {r_temprisk_lt/q:.2f} Eur/MWh",
)
ax.legend()
#  MW.
ax = axes[4]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.S.delta_qo.w, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text(
    "Long-term change in price (+ = increase)\n(between acquisition and M-1)"
)
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.L.delta_pu.p, c="orange", linestyle="-", linewidth=0.5, label="diff")
ax.legend()

# Temprisk - St
#  Times with positive contribution (costs > 0).
times = ((df.temprisk.St.r > 0).shift() - (df.temprisk.St.r > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[6]
ax.title.set_text("Change in offtake vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(*df.S.delta_qo.w, 0), 0), min(*df.S.delta_pu.p, 0), 0, **pos)
ax.fill_between((max(*df.S.delta_qo.w, 0), 0), max(*df.S.delta_pu.p, 0), 0, **pos)
ax.scatter(
    df.S.delta_qo.w,
    df.S.delta_pu.p,
    c="green",
    s=10,
    alpha=0.5,
    label=f"r_temprisk_st   = {r_temprisk_st/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_temprisk_st = {r_temprisk_st/q:.2f} Eur/MWh",
)
ax.legend()
#  MW.
ax = axes[7]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.S.delta_qo.w, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("Short-term change in price (+ = increase)\n(between M-1 and spot)")
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.S.delta_pu.p, c="green", linestyle="-", linewidth=0.5, label="diff")
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
# Temprisk.
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
ax.plot(monthly["0"].envir.t, "b-", linewidth=1, label="expected")
ax.plot(monthly.S.envir.t, "r-", linewidth=1, label="actual")
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Change in offtake, per month")
ax.yaxis.label.set_text("MWh")
ax.plot(
    (monthly.S.qo.w - monthly["0"].qo.w) * monthly.duration,
    c="purple",
    linewidth=1,
    label="change (+ = more offtake)",
)

ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Month price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(monthly["0"].pu.p, c="orange", linewidth=1, label="at acquisition")
ax.plot(monthly.L.pu.p, c="green", linewidth=1, label="1M before delivery")
ax.plot(monthly.S.pu.p, "r-", linewidth=1, label="actual")
ax.legend()

# Temprisk - Lt
#  Distribution.
x = yearly.temprisk.Lt.p.sort_values().tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(**temprisk_dist.loc["Lt"]).cdf(x_fit)
ax = axes[3]
ax.title.set_text(
    "temprisk premium, long-term component\n(+ = additional costs), per year"
)
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
    label="Fit:\nmean: {loc:.2f}, std: {scale:.2f}".format(**temprisk_dist.loc["St"]),
)
ax.legend()
#  Eur.
ax = axes[4]
ax.title.set_text(
    "temprisk revenue, long-term component\n(+ = additional costs), per month"
)
ax.yaxis.label.set_text("Eur")
ax.plot(
    monthly.temprisk.Lt.r,
    color="orange",
    linestyle="-",
    linewidth=1,
    label="r_temprisk_lt",
)
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text(
    "temprisk premium, long-term component\n(+ = additional costs), per year"
)
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(
    yearly.index,
    yearly.temprisk.Lt.p,
    width=365 * 0.9,
    color="orange",
    label="p_temprisk_lt",
)
ax.legend()

# Temprisk - St
#  Distribution.
x = yearly.temprisk.St.p.sort_values().tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(**temprisk_dist.loc["St"]).cdf(x_fit)
ax = axes[6]
ax.title.set_text(
    "temprisk premium, short-term component\n(+ = additional costs), per year"
)
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
    label="Fit:\nmean: {loc:.2f}, std: {scale:.2f}".format(**temprisk_dist.loc["St"]),
)
ax.legend()
#  Eur.
ax = axes[7]
ax.title.set_text(
    "temprisk revenue, short-term component\n(+ = additional costs), per month"
)
ax.yaxis.label.set_text("Eur")
ax.plot(
    monthly.temprisk.St.r,
    color="green",
    linestyle="-",
    linewidth=1,
    label="r_temprisk_st",
)
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text(
    "temprisk premium, short-term component\n(+ = additional costs), per year"
)
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(
    yearly.index,
    yearly.temprisk.St.p,
    width=365 * 0.9,
    color="green",
    label="p_temprisk_st",
)
ax.legend()

fig.tight_layout()

# %% Just Cost

# Formatting
plt.style.use("seaborn")
# positive formatting (costs>0), negative formatting (costs<0)

fig = plt.figure(figsize=(8, 8))
axes = []
#  Cost temprisk LT
axes.append(plt.subplot2grid((2, 2), (0, 0), fig=fig))
#  Cost temprisk ST
axes.append(plt.subplot2grid((2, 2), (0, 1), fig=fig, sharex=axes[0], sharey=axes[0]))
#  Cost [Eur/MWh] temprisk LT
axes.append(plt.subplot2grid((2, 2), (1, 0), fig=fig))
#  Cost [Eur/MWh] temprisk ST
axes.append(plt.subplot2grid((2, 2), (1, 1), fig=fig, sharex=axes[2], sharey=axes[2]))

# Temprisk - Lt
#  Distribution.
source_vals = yearly.temprisk.Lt.r
loc, scale = norm.fit(source_vals)
#  Eur.
ax = axes[0]
ax.title.set_text("temprisk costs, long-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.bar(
    yearly.index,
    yearly.temprisk.Lt.r / 1000,
    width=365 * 0.9,
    color="orange",
    label=f"long-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
#  Eur/MWh.
ax = axes[2]
x = yearly.temprisk.Lt.p.sort_values().tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(**temprisk_dist.loc["Lt"]).cdf(x_fit)
ax.title.set_text(
    "temprisk premium, long-term component\n(+ = additional costs), per year"
)
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
    label="Fit:\nmean: {loc:.2f}, std: {scale:.2f}".format(**temprisk_dist.loc["Lt"]),
)
ax.legend()

# temprisk - St
#  Distribution.
source_vals = yearly.temprisk.St.r
loc, scale = norm.fit(source_vals)
#  Eur.
ax = axes[1]
ax.title.set_text("temprisk costs, short-term component\nper year")
ax.yaxis.label.set_text("kEur")
ax.bar(
    yearly.index,
    yearly.temprisk.St.r / 1000,
    width=365 * 0.9,
    color="green",
    label=f"short-term component\nmean: {loc/1e3:.0f} kEur\n std: {scale/1e3:.0f} kEur",
)
#  Eur/MWh.
ax = axes[3]
x = yearly.temprisk.St.p.sort_values().tolist()
x_fit = np.linspace(1.5 * min(x) - 0.5 * max(x), -0.5 * min(x) + 1.5 * max(x), 300)
y_fit = norm(**temprisk_dist.loc["St"]).cdf(x_fit)
ax.title.set_text(
    "temprisk premium, short-term component\n(+ = additional costs), per year"
)
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
    label="Fit:\nmean: {loc:.2f}, std: {scale:.2f}".format(**temprisk_dist.loc["St"]),
)
ax.legend()

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
ax.fill_between((min(*df.S.delta_qo.w, 0), 0), min(*df.S.delta_pu.p, 0), 0, **pos)
ax.fill_between((max(*df.S.delta_qo.w, 0), 0), max(*df.S.delta_pu.p, 0), 0, **pos)
ax.scatter(
    df.S.delta_qo.w,
    df.S.delta_pu.p,
    c="green",
    s=10,
    alpha=0.9,
    label=f"r_temprisk_st   = {r_temprisk_st/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_temprisk_st = {r_temprisk_st/q:.2f} Eur/MWh",
)
# ax.legend()

# %% CALCULATE: conditional value at risk


# %% Premium

quantile = 0.8

ppf_Lt = norm(loc=0, scale=temprisk_dist.loc["Lt", "scale"]).ppf
ppf_St = norm(**temprisk_dist.loc["St"]).ppf

premium_cost = np.sqrt(ppf_Lt(0.5) ** 2 + ppf_St(0.5) ** 2)
premium_risk = np.sqrt(ppf_Lt(quantile) ** 2 + ppf_St(quantile) ** 2) - premium_cost
premium_total = premium_cost + premium_risk
premium = pd.Series(
    {"cost": premium_cost, "risk": premium_risk, "total": premium_total}
)
print(
    "Premium: Cost: {cost:.2f} Eur/MWh; Risk: {risk:.2f} Eur/MWh; total: {total:.2f} Eur/MWh.".format(
        **premium
    )
)
premium.round(2).to_clipboard()
# %%
