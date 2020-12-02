# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:39:48 2020

@author: ruud.wijtvliet
"""


import pandas as pd

"""
Use case: resample aggregate (i.e., mean or time-integrated) data. 
e.g., a dataframe with mean speed and total distance driven by a car on 3 days in March.




NB1: Can't (shouldn't) use DateTimeIndex, as these denote *moments* in time, not periods. 
Therefore, a DateTimeIndex should only be used for quantities that have a meaning at any
moment in time. In this example: using a DateTimeIndex can be used to specify the velocity 
*at a given moment* (and not the mean velocity on a day) and the *cumulative distance at 
(=until and including) that moment* (and not the distance driven on a day.)
"""

# Without timezones:
pi = pd.period_range("2020-03-28", periods=3, freq="D")
hours = ((pi + 1).start_time - pi.start_time).total_seconds() / 3600
#   1: average speed in period --> distance traveled in period
avspeed = pd.Series([2, 3, 4], pi)  # km/h
distance = avspeed * hours  # km
#   2: upsample a mean quantity (=speed)
avspeed_h = avspeed.resample("H").ffill()
#   3: upsample an integrated quantity (=distance)
distance_h = (
    distance.resample("15T")
    .asfreq()
    .fillna(0)
    .groupby(pd.Grouper(freq=distance.index.freq))
    .transform(np.mean)
)
#   4: downsample a mean quantity
avspeed2 = avspeed_h.resample("M").mean()


distance = pd.Series([366], pd.PeriodIndex(["2020"], freq="Y"))
dist_m = (
    distance.resample("M")
    .asfreq()
    .fillna(0)
    .groupby(pd.Grouper(freq=distance.index.freq))
    .transform(np.mean)
)


#%% Setup.
COL1, COL2 = "r", "b"
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("seaborn")


def speedometer(t: float) -> float:
    """Return speedometer value [km/h] as function of time [h] since start."""
    return 1 - 0.2 * np.cos(t / 15) - 0.8 * np.sin(t / 200)


def odometer(t: float) -> float:
    """Return odometer value [km] as function of time [h] since start; start value is 0 km."""
    return t - 0.2 * 15 * np.sin(t / 15) + 0.8 * 200 * np.cos(t / 200) - 160


# Plot continuous functions.
hours = np.linspace(0, 1000, 10000)
data = pd.DataFrame({"speedo": speedometer(hours), "odo": odometer(hours)}, index=hours)
# Plot.
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
#   continuous
axes[0].plot(data["speedo"], COL1)
axes[1].plot(data["odo"], COL2)
#   titles, labels and limits
axes[0].title.set_text("Speedometer")
axes[1].title.set_text("Odometer")
axes[0].yaxis.label.set_text("velocity [km/h]")
axes[1].yaxis.label.set_text("distance [km]")
axes[0].xaxis.label.set_text("time since start [h]")
axes[1].xaxis.label.set_text("time since start [h]")
axes[0].set_ylim(0, 1.5)
axes[1].set_ylim(0, 500)
axes[0].set_xlim(0, 700)
fig.tight_layout()

#%% Daily timestamps
hour = lambda ts: (ts - pd.Timestamp("2020-01-01")).total_seconds() / 3600
# Minute values. (just for graph)
idx_ts_minute = pd.date_range("2020-01-01", "2020-04-10", freq="T", closed="left")
df_ts_minute = pd.DataFrame(
    {"speedo": speedometer(hour(idx_ts_minute)), "odo": odometer(hour(idx_ts_minute))},
    index=idx_ts_minute,
)
# Daily values.
idx_ts = pd.date_range("2020-01-01 00:00", "2020-04-10", freq="D", closed="left")
df_ts = pd.DataFrame(
    {"speedo": speedometer(hour(idx_ts)), "odo": odometer(hour(idx_ts))}, index=idx_ts
)
# Plot.
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
#   continuous
axes[0].plot(df_ts_minute["speedo"], f"{COL1}--", alpha=0.35, lw=1)
axes[1].plot(df_ts_minute["odo"], f"{COL2}--", alpha=0.35, lw=1)
#   discrete
axes[0].plot(df_ts["speedo"], f"{COL1}o")
axes[1].plot(df_ts["odo"], f"{COL2}o")
#   titles, labels and limits
axes[0].title.set_text("Speedometer")
axes[1].title.set_text("Odometer")
axes[0].yaxis.label.set_text("velocity [km/h]")
axes[1].yaxis.label.set_text("distance [km]")
axes[0].set_ylim(0, 1.5)
axes[1].set_ylim(0, 175)
axes[0].set_xlim([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-12")])
axes[0].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-\n%m-%d"))
fig.tight_layout()
#   highlights
i = 2
ts, speedo, odo = df_ts.index[i], df_ts["speedo"][i], df_ts["odo"][i]
axes[0].axvline(ts, color="grey", linewidth=2, alpha=0.2)
axes[1].axvline(ts, color="grey", linewidth=2, alpha=0.2)
axes[0].annotate(
    f"Data at row index {i}:\nAt {ts},\nspeedometer shows {speedo:.2f} km/h",
    xy=(ts, speedo),
    xycoords="data",
    xytext=(0.4, 0.95),
    textcoords="axes fraction",
    arrowprops={"fc": "black", "alpha": 0.6, "shrink": 0.15},
    horizontalalignment="left",
    verticalalignment="top",
)
axes[1].annotate(
    f"Data at row index {i}:\nAt {ts},\nodometer shows {odo:.1f} km",
    xy=(ts, odo),
    xycoords="data",
    xytext=(0.45, 0.25),
    textcoords="axes fraction",
    arrowprops={"fc": "black", "alpha": 0.6, "shrink": 0.15},
    horizontalalignment="left",
    verticalalignment="top",
)

# Usability


#%% Daily periods
duration = lambda pe: ((pe + 1).start_time - pe.start_time).total_seconds() / 3600
distance = lambda pe: odometer(hour((pe + 1).start_time)) - odometer(
    hour(pe.start_time)
)
avspeed = lambda pe: distance(pe) / duration(pe)
fracTs = lambda pe, f: pe.start_time + f * ((pe + 1).start_time - pe.start_time)
midTs = lambda pe: fracTs(pe, 0.5)  # timestamp mid-period
idx_pe = pd.period_range("2020-01-01", periods=100, freq="D")
df_pe = pd.DataFrame(
    {"avspeed": avspeed(idx_pe), "dist": distance(idx_pe)}, index=idx_pe
)
# Helper df for plot.
df = df_pe.copy()
df["odo_start"] = df_ts.loc[idx_pe.start_time, "odo"].values
df["odo_end"] = df["odo_start"] + df["dist"]
df["midTs"] = midTs(idx_pe)
# Plot.
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
#   continuous
axes[0].plot(df_ts_minute["speedo"], f"{COL1}--", alpha=0.35, lw=1)
axes[1].plot(df_ts_minute["odo"], f"{COL2}--", alpha=0.35, lw=1)
#   discrete
axes[0].hlines(
    df_pe["avspeed"], df_pe.index.start_time, (df_pe.index + 1).start_time, COL1
)
axes[1].scatter(df.index.start_time, df["odo_start"], c=COL2, s=4)
axes[1].hlines(
    df["odo_start"],
    fracTs(idx_pe - 1, 0.05),
    fracTs(idx_pe, 0.95),
    COL2,
    linewidth=1,
    alpha=0.125,
)
#   titles, labels and limits
axes[0].title.set_text("Speedometer")
axes[1].title.set_text("Odometer")
axes[0].yaxis.label.set_text("velocity [km/h]")
axes[1].yaxis.label.set_text("distance [km]")
axes[0].set_ylim(0, 1.5)
axes[1].set_ylim(0, 175)
axes[0].set_xlim([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-12")])
axes[0].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-\n%m-%d"))
fig.tight_layout()
for x, y, dy in zip(df["midTs"], df["odo_start"], df["dist"]):
    axes[1].annotate(
        "",
        xytext=(x, y),
        xy=(x, y + dy),
        arrowprops={"color": COL2, "lw": 1, "arrowstyle": "-|>"},
    )
#   highlights
i = 2
pe, ts, avspeed, odo_start, dist = (
    df.index[i],
    df["midTs"][i],
    df["avspeed"][i],
    df["odo_start"][i],
    df["dist"][i],
)
axes[0].axvspan(pe.start_time, (pe + 1).start_time, alpha=0.2, color="grey")
axes[1].axvspan(pe.start_time, (pe + 1).start_time, alpha=0.2, color="grey")
axes[0].annotate(
    f"Data at row index {i}:\n\nDuring {pe}\n(from {pe.start_time}\nuntil {(pe+1).start_time})\n\nAverage velocity was {avspeed:.2f} km/h",
    xy=((pe + 1).start_time, avspeed),
    xycoords="data",
    xytext=(0.4, 0.95),
    textcoords="axes fraction",
    arrowprops={"fc": "black", "alpha": 0.6, "shrink": 0.15},
    horizontalalignment="left",
    verticalalignment="top",
)
axes[1].annotate(
    f"Data at row index {i}:\n\nDuring {pe}\n(from {pe.start_time}\nuntil {(pe+1).start_time})\n\nDistance (i.e., increase in\nodometer value) was {dist:.2f} km",
    xy=(ts, odo_start + dist / 2),
    xycoords="data",
    xytext=(0.45, 0.35),
    textcoords="axes fraction",
    arrowprops={"fc": "black", "alpha": 0.6, "shrink": 0.15},
    horizontalalignment="left",
    verticalalignment="top",
)

#%% Calculating additional values
df_pe["dur"] = (
    (df_pe.index + 1).start_time - df_pe.index.start_time
).total_seconds() / 3600
df_pe["dist2"] = df_pe["avspeed"] * df_pe["dur"]
df_pe[["dist", "dist2"]]
df_pe["avspeed2"] = df_pe["dist"] / df_pe["dur"]
df_pe[["avspeed", "avspeed2"]]

# Plot.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#   continuous
ax.plot(df_ts_minute["speedo"], f"{COL1}--", alpha=0.35, lw=1)
#   discrete
ax.hlines(df_pe["avspeed"], df_pe.index.start_time, (df_pe.index + 1).start_time, COL1)
#   titles, labels and limits
ax.title.set_text("Speedometer")
ax.yaxis.label.set_text("velocity [km/h]")
ax.set_ylim(0, 1.5)
ax.set_xlim([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-12")])
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-\n%m-%d"))
fig.tight_layout()
#   highlights
i = 2
pe, ts, avspeed, odo_start, dist = (
    df.index[i],
    df["midTs"][i],
    df["avspeed"][i],
    df["odo_start"][i],
    df["dist"][i],
)
ax.axvspan(
    pe.start_time,
    (pe + 1).start_time,
    ymax=1,
    ymin=avspeed / 1.5,
    alpha=0.2,
    color="grey",
)
ax.axvspan(
    pe.start_time,
    (pe + 1).start_time,
    ymin=0,
    ymax=avspeed / 1.5,
    alpha=0.4,
    color=COL2,
)
ax.annotate(
    f"Data at row index {i}:\n\nBy multiplying\n· average velocity = {avspeed:.2f} km/h\nwith\n· duration = 24 h,\nwe obtain\n· distance = {dist:.2f} km\nas the area under curve.",
    xy=((pe + 1).start_time, avspeed * 0.85),
    xycoords="data",
    xytext=(0.5, 0.95),
    textcoords="axes fraction",
    arrowprops={"fc": "black", "alpha": 0.6, "shrink": 0.15},
    horizontalalignment="left",
    verticalalignment="top",
)


# Plot.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#   continuous
ax.plot(df_ts_minute["odo"], f"{COL2}--", alpha=0.35, lw=1)
#   discrete
ax.scatter(df.index.start_time, df["odo_start"], c=COL2, s=4)
ax.hlines(
    df["odo_start"],
    fracTs(idx_pe - 1, 0.05),
    fracTs(idx_pe, 0.95),
    COL2,
    linewidth=1,
    alpha=0.125,
)
#   titles, labels and limits
ax.title.set_text("Odometer")
ax.yaxis.label.set_text("distance [km]")
ax.set_ylim(0, 175)
ax.set_xlim([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-12")])
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-\n%m-%d"))
fig.tight_layout()
for x, y, dy in zip(df["midTs"], df["odo_start"], df["dist"]):
    ax.annotate(
        "",
        xytext=(x, y),
        xy=(x, y + dy),
        arrowprops={"color": COL2, "lw": 1, "arrowstyle": "-|>"},
    )
#   highlights
i = 2
pe, ts, avspeed, odo_start, dist = (
    df.index[i],
    df["midTs"][i],
    df["avspeed"][i],
    df["odo_start"][i],
    df["dist"][i],
)
ax.axvspan(pe.start_time, (pe + 1).start_time, alpha=0.2, color="grey")
ax.plot(
    (pe.start_time, (pe + 1).start_time),
    (odo_start, odo_start + dist),
    alpha=1,
    color=COL1,
    lw=2,
)
ax.annotate(
    f"Data at row index {i}:\n\nBy dividing\n· distance = {dist:.2f} km\nby\n· duration = 24 h,\nwe obtain\n· average velocity = {avspeed:.2f} km/h\nas the slope of the curve.",
    xy=((pe + 1).start_time, odo_start + dist / 2),
    xycoords="data",
    xytext=(0.5, 0.35),
    textcoords="axes fraction",
    arrowprops={"fc": "black", "alpha": 0.6, "shrink": 0.15},
    horizontalalignment="left",
    verticalalignment="top",
)

#%% Conversion


df_ts.index


df_pe["odo_end"] = df_pe["dist"].cumsum()
df_pe["odo_start"] = df_pe["odo_end"] - df_pe["dist"]
df_pe[["odo_start", "dist", "odo_end"]]
df_tmp = pd.DataFrame({"odo": df_pe["odo_start"]}, df_pe.index.start_time)
df_tmp.index
(df_tmp["odo"] == df_ts["odo"]).all()
