import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def explolatoryDataAnalysis():

    t = lb.temperatures.historic.tmpr()
    complete_tmpr = lb.temperatures.historic.fill_gaps(t)

    # plot yearly data with missing values
    yearly_plot(t, "Yearly temperatures with missing gaps")

    # plot yearly data after filling the missing values with multiple Linear regression
    yearly_plot(complete_tmpr, "Yearly temperatures after filling missing gaps")

    # plot montly data with complete data
    monthly_plot(complete_tmpr, "Monthly temperature averages for all climatezones")

    return


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
    fig.savefig(f"{title}.png")


def monthly_plot(t: pd.DataFrame, title: str):

    colors = {
        "t_1": "gray",
        "t_2": "orange",
        "t_3": "green",
        "t_4": "purple",
        "t_5": "red",
        "t_6": "blue",
        "t_7": "black",
        "t_8": "violet",
        "t_9": "olive",
        "t_10": "lavender",
        "t_11": "pink",
        "t_12": "yellow",
        "t_13": "white",
        "t_14": "red",
        "t_15": "purple",
    }

    t = t.dropna()
    tavg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
    tavg.index = pd.MultiIndex.from_tuples(tavg.index, names=("year", "month"))

    fig, axes = plt.subplots(3, 4, sharey=False, figsize=(15, 10))
    fig.suptitle(title)
    for (m, df), ax in zip(tavg.groupby("month"), axes.flatten()):
        ax.set_title(f"Month: {m}")
        for name, s in df.droplevel(1).items():
            ax.plot(s, c=colors.get(name, "gray"), label=name)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{title}.png")


def regression():

    t = lb.temperatures.historic.fill_gaps(lb.temperatures.historic.tmpr())
    t = t.resample(rule="MS")
    xtrain, xtest = train_test_split(t, test_size=0.30, random_state=42)

    ytrain = xtrain["t_3"]
    ytest = xtest["t_3"]
    xtrain = xtrain.index
    xtest = xtest.index
    print(xtrain.shape)
    print(xtest.shape)

    lin_reg = LinearRegression()
    lin_reg.fit(xtrain, ytrain)
