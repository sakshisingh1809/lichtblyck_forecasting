"""
Analyze and plot historic temperatures, yearly as well as monthly for better 
understanding of dataset. 
"""

import lichtblyck_forecasting as lf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

ANALYSISDATAFOLDER = Path(__file__).parent / "Data Analysis"


def yearly_plot(tmpr: pd.DataFrame, title: str):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(title, fontsize=18, y=0.95)
    climatezones = [
        "t_1",
        "t_2",
        "t_3",
        "t_4",
        "t_5",
        "t_6",
        "t_7",
        "t_8",
        "t_9",
        "t_10",
        "t_11",
        "t_12",
        "t_13",
        "t_14",
        "t_15",
    ]

    df = tmpr.resample(rule="AS").mean()  # resampling the data into "Yearly" format

    fig.text(0.5, 0.09, "year", ha="center")
    fig.text(0.09, 0.5, "temp", va="center", rotation="vertical")

    for cz, ax in zip(climatezones, axes.ravel()):  # loop through climatezones and axes
        df[cz].plot(
            ax=ax, color="blue", linewidth=1
        )  # filter df for cz and plot on specified axes
        ax.set_title(cz)  # chart formatting
        ax.set_xlabel("")

    axes[3, 3].axis(
        "off"
    )  # since we have only 15 climatezones, so we ignore the 16th graph

    for i in range(
        4
    ):  # set all labels of the 1st row at the top and make bottom labels invisible
        axes[0, i].xaxis.set_tick_params(labeltop=True)
        axes[0, i].xaxis.set_tick_params(labelbottom=False)
        # axes[i, 0].set_ylabel("temp")

    for i in range(1, 4):
        for j in range(4):
            axes[i, j].axes.get_xaxis().set_visible(False)
    plt.show()
    fig.savefig(os.path.join(ANALYSISDATAFOLDER, f"{title}.png"))


def monthly_plots(t: pd.DataFrame, title: str):
    t = t.dropna()
    tavg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
    tavg.index = pd.MultiIndex.from_tuples(tavg.index, names=("year", "month"))

    fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
    fig.suptitle(title)
    for (m, df), ax in zip(tavg.groupby("month"), axes.flatten()):
        ax.set_title(f"Month: {m}")
        for name, s in df.droplevel(1).items():
            ax.plot(s, label=name)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSISDATAFOLDER, f"{title}.png"))


def single_monthly_plot(t: pd.DataFrame, title: str):
    t = t.dropna()
    tavg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
    tavg.index = pd.MultiIndex.from_tuples(tavg.index, names=("year", "month"))

    fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
    fig.suptitle(title)
    for (m, df), ax in zip(tavg.groupby("month"), axes.flatten()):
        ax.set_title(f"Month: {m}")
        ax.plot(df.values)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSISDATAFOLDER, f"{title}.png"))


def simple_curve(t: pd.DataFrame):
    fig, axs = plt.subplots(5, 3, figsize=(20, 20), sharex=True)
    axx = axs.ravel()
    for i in range(0, 15):
        t[t.columns[i]].loc[
            "1945-01-01 00:00:00+01:00":"2020-12-31 00:00:00+01:00"
        ].plot(
            ax=axx[i]
        )  # plot from 1945-2020
        axx[i].set_xlabel("year")
        axx[i].set_ylabel("tmp")
        axx[i].grid(which="minor", axis="x")


if __name__ == "__main__":

    t = lf.temperatures.historic.tmpr()
    complete_tmpr = lf.temperatures.historic.fill_gaps(t)

    # plot yearly data with missing values
    yearly_plot(t, "Yearly temperatures with missing gaps")

    # plot yearly data after filling the missing values with multiple Linear regression
    yearly_plot(complete_tmpr, "Yearly temperatures after filling missing gaps")

    # plot montly data with complete data
    monthly_plots(complete_tmpr, "Monthly temperature averages for all climatezones")

    # plot montly data for t_3 (Hamburg) climate zone
    single_monthly_plot(
        complete_tmpr["t_3"],
        "t_3 - Monthly temperature averages",
    )
