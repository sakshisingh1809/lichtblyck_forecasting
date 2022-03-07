"""Calculate the short-term temprisk from historic values."""

#%%


from matplotlib import pyplot as plt
import matplotlib as mpl
import lichtblyck as lb
import pandas as pd
import numpy as np
import datetime


# %% BIG DF TO STORE ALL.
hourly = pd.DataFrame(columns=[[], []])
daily = pd.DataFrame(columns=[[], []])

# %% TEMPERATURES AND OFFTAKES.

# Temperature weights.
weights = pd.DataFrame(
    {
        "power": [0.1, 0.7, 1.3, 1.5, 1.9, 0.7, 0.3, 0, 0.1, 0.1, 0, 1.3, 1.1, 0, 0],
        "gas": [0.7, 4.0, 13, 29.0, 13.2, 3.6, 2.9, 0, 1.8, 0.4, 0, 9.4, 3.4, 0, 0.1],
    },
    index=range(1, 16),
)  # GWh/a in each zone
weights = weights["gas"] / weights["gas"].sum()  # expect same distribution as in gas pf

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

hourly[("L", "wo")] = tlp(t_exp).resample("H").mean()
hourly[("S", "wo")] = tlp(t_act).resample("H").mean()
hourly[("L", "qo")] = hourly.L.wo * 1.0
hourly[("S", "qo")] = hourly.S.wo * 1.0
daily[("S", "t")] = t_act
daily[("L", "t")] = t_exp
daily = daily.dropna()

# %% PRICES.


# Historic prices
# Month.
pu_fut = lb.prices.montel.power_futures("m", 1)
# Spot.
pu_spot = lb.prices.montel.power_spot()
hourly[("S", "pu")] = lb.fill_gaps(pu_spot, 5)

# %% TRY TO calculate the quarterhourly prices before start of the month.


hist = pd.DataFrame(columns=[[], []])


# (a) Split historic spot prices into (1) month average peak and offpeak, and (2) month-to-hour deviations)


hist[("S", "pu")] = pu_spot.pint.m  # Won't work when units are included.
hist[("S", "pu_m")] = lb.prices.convert.tseries2tseries(hist.S.pu, "MS")
hist[("S", "pu_m2h")] = hist.S.pu - hist.S.pu_m
# visual check
hist.S[["pu", "pu_m"]].plot()
hist.S[["pu", "pu_m", "pu_m2h"]].loc["2020-11":"2020-12", :].plot()


# (b) For each delivery month, find historic peak-price and historic offpeak-price
# (for a trading day representing moment L).


def pick_pu_fut(df):
    cols = ["p_offpeak", "p_peak", "ts_left_trade", "anticipation"]  # columns to keep
    # keep suitable
    mask = (df["anticipation"] > datetime.timedelta(days=25)) & (
        df["anticipation"] < datetime.timedelta(days=39)
    )
    df = df[mask].dropna()
    # find one value
    if not df.empty:
        df = df.sort_values("anticipation")  # sort suitable.
        df = df.reset_index()  # to get 'ts_left_trade' in columns.
        return pd.Series(df[cols].iloc[0])  # keep one.
    return pd.Series([], dtype=float)


pu_L_m = pu_fut.groupby(level=0).apply(pick_pu_fut).dropna().unstack()
for col in ("p_peak", "p_offpeak", "p_base"):
    if col in pu_L_m.columns:
        pu_L_m[col] = pd.Series(
            [v.magnitude for v in pu_L_m[col].values],
            index=pu_L_m.index,
            dtype="pint[Eur/MWh]",
        )
# Quick check to see from which day the relevant price comes.
print(pu_L_m[["ts_left_trade", "anticipation"]].describe())  # quick check
# Drop irrelevant data.
pu_L_m = lb.set_ts_index(pu_L_m.drop(columns=["ts_left_trade", "anticipation"]))
pu_L_m["po_spread"] = pu_L_m.p_peak - pu_L_m.p_offpeak
# Quick visual check.
pu_L_m.plot()


# (c) For each month, turn the uniform peak- and offpeak-price into a (quarter)hourly profile.


# Use actual spot prices to calculate expected M2H profile: mean but without extremes
rolling_av = lambda nums: sum(np.sort(nums)[5:-15]) / (len(nums) - 20)
# .. for each hour and weekday, find average m2h value for past 52 values.
groupfn = lambda ts: (ts.weekday(), ts.hour)
apply_rolling_av = lambda df: df.rolling(52).apply(rolling_av).shift()
pu_L_m2h = hist.S.pu_m2h.groupby(groupfn).apply(apply_rolling_av)
# . Make arbitrage free (average = 0 for each month)
pu_L_m2h = pu_L_m2h - lb.prices.convert.tseries2tseries(pu_L_m2h, "MS")
# visual check
weeksatsun_function = lambda ts: {5: "Sat", 6: "Sun"}.get(ts.weekday(), "Weekday")
monday_function = lambda ts: ts.floor("D") - pd.Timedelta(ts.weekday(), "D")
cmap = mpl.cm.get_cmap("hsv")  # cyclic map
for daytype, daytypedf in pu_L_m2h.groupby(weeksatsun_function):  # weekday, sat, sun
    fig, axes = plt.subplots(figsize=(20, 10), ncols=2)
    fig.suptitle(f"{daytype}: Historic M2H values; rolling average.")

    ispeak = lb.is_peak_hour(daytypedf.index)
    for ax, fltr, title in zip(axes, [~ispeak, ispeak], ["offpeak", "peak"]):
        for hour, subdf in daytypedf[fltr].groupby(lambda ts: ts.hour):
            subdf = subdf.groupby(monday_function).mean()
            ax.plot(subdf, color=cmap(hour / 24), label=hour)
            ax.set_title(title)
            ax.legend()

# (d) Add monthly values and month-2-hour values to get expected hourly prices.

hist[("L", "pu_m")] = lb.prices.convert.bpoframe2tseries(pu_L_m, "H", "p_")
hist[("L", "pu_m2h")] = pu_L_m2h.astype("pint[Eur/MWh]")
hist[("L", "pu")] = hist.L.pu_m + hist.L.pu_m2h
hist = hist.dropna()

hourly[("L", "pu")] = hist.L.pu

# %% SHORT-TERM COSTS:

# sttr = 'short-term temperature risk'
hourly = hourly.dropna()
hourly[("sttr", "delta_pu")] = hourly.S.pu.astype("pint[Eur/MWh]") - hourly.L.pu
hourly[("sttr", "delta_qo")] = hourly.S.qo - hourly.L.qo
hourly[("sttr", "delta_ro")] = hourly.sttr.delta_pu * hourly.sttr.delta_qo
hourly = hourly.sort_index(axis=1)


cols_sum = [
    ("sttr", "delta_ro"),
    ("sttr", "delta_qo"),
    ("S", "qo"),
    ("L", "qo"),
]
cols_avg = [("L", "pu"), ("S", "pu")]


def changefreq(freq):
    dfs = [
        hourly[cols_sum].resample(freq).sum(),
        hourly[cols_avg].resample(freq).mean(),
        daily.resample(freq).mean(),
    ]
    df = pd.concat(dfs, axis=1)
    df[("sttr", "p_premium")] = df.sttr.delta_ro / df.S.qo
    df[("S", "delta_pu")] = df.S.pu.astype("pint[Eur/MWh]") - df.L.pu
    return df.sort_index(1)


monthly = changefreq("MS")
quarterly = changefreq("QS")
yearly = changefreq("AS")


# %% VISUALIZE DISTRIBUTIONS.

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm


def visualze_fn(monthly_premiums_per_simulation, quartely_premiums_per_simulation):
    def fn(show: str = "M"):
        if show == "M":
            shape = (3, 4)
            df_premiums = monthly_premiums_per_simulation.groupby(lambda ts: ts.month)
        elif show == "Q":
            shape = (2, 2)
            df_premiums = quartely_premiums_per_simulation.groupby(
                lambda ts: ts.quarter
            )

        fig, axes = plt.subplots(*shape, sharey=True, figsize=(16, 10))

        for (ts, s), ax in zip(df_premiums, axes.flatten()):
            #  Distribution.
            source_vals = s.pint.m
            loc, scale = norm.fit(source_vals)
            x = source_vals.sort_values().tolist()
            x_fit = np.linspace(
                1.1 * min(x) - 0.1 * max(x), -0.1 * min(x) + 1.1 * max(x), 300
            )
            y_fit = norm(loc, scale).cdf(x_fit)
            ax.title.set_text(f"{ts}.")
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

    return fn


viz = visualze_fn(
    monthly.loc[monthly.index > "2002-08"].sttr.p_premium,
    quarterly.loc[quarterly.index > "2002-08"].sttr.p_premium,
)

viz("M")
viz("Q")
# %% PLOT AND FIND QUANTILE.

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


df = monthly

# Expected and actual
#  Temperatures.
ax = axes[0]
ax.title.set_text("Temperature")
ax.yaxis.label.set_text("degC")
ax.plot(df.L.t, "b-", linewidth=0.5, label="expected")
ax.plot(df.S.t, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake")
ax.yaxis.label.set_text("MWh")
ax.plot(df.L.qo, "b-", linewidth=0.5, label="expected")
ax.plot(df.S.qo, "r-", linewidth=0.5, label="actual")
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(
    df.L.pu.pint.m, c="green", linewidth=0.5, label="market price 1M before delivery"
)
ax.plot(df.S.pu.pint.m, "r-", linewidth=0.5, label="spot price")
ax.legend()


# Temprisk - St
#  Scatter.
ax = axes[3]
ax.title.set_text("Change in offtake vs short-term change in price")
ax.xaxis.label.set_text("MWh")
ax.yaxis.label.set_text("Eur/MWh")
ax.scatter(df.sttr.delta_qo, df.S.delta_pu.pint.m, c="green", s=10, alpha=0.5)
ax.legend()
#  MWh.
ax = axes[4]
ax.title.set_text("Change in offtake (+ = increase)\n(= change in spot volume)")
ax.yaxis.label.set_text("MWh")
ax.plot(df.sttr.delta_qo, c="purple", linewidth=0.5, label="change in offtake")
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text("Short-term change in price (+ = increase)\n(between M-1 and spot)")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df.S.delta_pu.pint.m, c="green", linestyle="-", linewidth=0.5, label="diff")
ax.legend()


ax = axes[6]
ax.plot(df.sttr.p_premium.pint.m, linestyle="-", c="green")
ax.title.set_text("Actual premium as needed.")
fig.tight_layout()

# %% Monthly premium.

from scipy.stats import norm


def fit(series):
    loc, scale = norm.fit(series.pint.m.values)
    ppf = norm(loc, scale).ppf
    return pd.Series(
        {
            "loc": loc,
            "scale": scale,
            "58%": ppf(0.58),
            "80%": ppf(0.8),
            "90%": ppf(0.9),
            "95%": ppf(0.95),
        }
    )


premiums_m = (
    monthly.dropna().sttr.p_premium.groupby(lambda ts: ts.month).apply(fit).unstack()
)
premiums_q = (
    quarterly.dropna()
    .sttr.p_premium.groupby(lambda ts: ts.quarter)
    .apply(fit)
    .unstack()
)

premiums_m.to_excel("sttr_monthly_premiumsbasedondeltap.xlsx")
premiums_q.to_excel("sttr_quarterly_premiumsbasedondeltap.xlsx")


# %% INPUT INTO LONG-TERM PREMIUM: VOLUME DEVIATIONS.


def describe_volumechanges(series):
    loc, scale = norm.fit(series.values)
    return pd.Series({"loc": loc, "scale": scale})


volumefactor = (monthly.S.qo / monthly.L.qo).dropna()
factors_per_month = (
    volumefactor.groupby(lambda ts: {"m": ts.month})
    .apply(describe_volumechanges)
    .unstack()
)


# %%
