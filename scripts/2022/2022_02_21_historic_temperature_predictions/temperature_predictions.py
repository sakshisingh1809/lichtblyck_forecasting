import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf


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


# funtion to create data for univariate forecasting
def univariate_data(dataset, start_idx, end_idx, history_size, target_size):
    data = []
    labels = []
    start_idx = start_idx + history_size
    if end_idx is None:
        end_idx = len(dataset) - target_size
    for i in range(start_idx, end_idx):
        idxs = range(i - history_size, i)
        data.append(np.reshape(dataset[idxs], (history_size, 1)))  # reshape data
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def data_preprocessing():
    climate_zone = "t_3"
    t = lb.temperatures.historic.fill_gaps(lb.temperatures.historic.tmpr())

    data = t[climate_zone]  # using univariate feature(Only temperature for given time)
    data = data.values

    train_split = 20000  # train test split for simple time series moving window average
    tf.random.set_seed(13)

    # standardize data
    data_mean = data[:train_split].mean()
    data_std = data[:train_split].std()
    data = (data - data_mean) / data_std

    data_history = 200  # last 200 values
    data_future = 0  # future data

    xtrain, ytrain = univariate_data(data, 0, train_split, data_history, data_future)

    xval, yval = univariate_data(data, train_split, None, data_history, data_future)
    """
    print(xtrain.shape, ytrain.shape)
    print(xval.shape, yval.shape)
    print("Single window of history data", xtrain[0])
    print("Target Temperature to predict ", ytrain[0])
    """

    return xtrain, xval, ytrain, yval


# fucntion to create time steps
def create_time_steps(length):
    return list(range(-length, 0))


# Moving window average
def MWA(history):
    return np.mean(history)


# function to prepare tensorflow dataset
def tensorflow_preprocessing(xtrain, xval, ytrain, yval):
    batch_size = 256
    buffer_size = 10000

    trainset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    trainset = trainset.cache().shuffle(buffer_size).batch(batch_size).repeat()

    validationset = tf.data.Dataset.from_tensor_slices((xval, yval))
    validationset = (
        validationset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    )

    return trainset, validationset


# function to plot time series data
def plot_time_series(plot_data, y, title):
    labels = ["History", "True Future", "Model Predcited"]
    marker = [".-", "ro", "go"]
    time_steps = create_time_steps(plot_data[0].shape[0])

    if y:
        future = y
    else:
        future = 0
    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])

    plt.xlabel("Time_Step")
    plt.savefig(f"{title}.png")
    return plt


# function to plot train test loss
def plot_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Train Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f"{title}.png")


"""
# multi step plotting function
def multi_step_plot(history, true_future, prediction):
    STEP = 6
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.grid()
    plt.plot(num_in, np.array(history[:, 1]), label="History")
    plt.plot(
        np.arange(num_out) / STEP, np.array(true_future), "bo", label="True Future"
    )
    if prediction.any():
        plt.plot(
            np.arange(num_out) / STEP,
            np.array(prediction),
            "ro",
            label="Predicted Future",
        )
    plt.legend(loc="upper left")
    plt.show()
"""


def lstm_model():

    i = 58  # select a random sample to plot
    steps = 400
    EPOCHS = 50

    (
        xtrain,
        xval,
        ytrain,
        yval,
    ) = data_preprocessing()  # prepare original timeseries dataset

    plot_time_series([xtrain[i], ytrain[i]], 0, "Sample plot")  # random sample plots
    plot_time_series([xtrain[i], ytrain[i], MWA(xtrain[i])], 0, "MWA predicted")

    trainset, validationset = tensorflow_preprocessing(
        xtrain, xval, ytrain, yval
    )  # prepare tensorflow dataset

    # Define LSTM model
    lstm_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(16, input_shape=xtrain.shape[-2:]),
            tf.keras.layers.Dense(1),
        ]
    )

    lstm_model.compile(optimizer="adam", loss="mae")

    model_history = lstm_model.fit(
        trainset,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=validationset,
        validation_steps=50,
    )

    for i, j in validationset.take(5):
        plot = plot_time_series(
            [i[0].numpy(), j[0].numpy(), lstm_model.predict(i)[0]], 0, "LSTM UNIVARIATE"
        )
        plot.show()
        plot.savefig("LSTM UNIVARIATE.png")

    # plot train and validation loss
    plot_loss(model_history, "Training vs validation loss")


"""
def offsetDataset(t: pd.DataFrame) -> pd.DataFrame:
    for i in range(1, 15):  # iterate through all timezones
        j = 2
        # one specific timezone

        t[i] = 1
    return
"""

if __name__ == "__main__":
    # explolatoryDataAnalysis()
    lstm_model()
