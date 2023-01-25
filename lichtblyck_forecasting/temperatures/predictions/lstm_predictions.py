"""
Predict future temperatures for a given climate zone using LSTM machine learning model, given historic temperatures. 
"""

import lichtblyck as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import os


# https://github.com/sunjoshi1991/Time-Series-Forecasting-using-LSTM/blob/master/Time_Series_Forecasting_(Predicting_Temperature)_using_LSTM_.ipynb

LSTMPREDICTIONSDATAFOLDER = Path(__file__).parent / "LSTM Predictions"
climate_zone = "t_3"  # t_3 = hamburg, t_4 = potsdamm, t_5 = essen, t_12 = mannheim


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

    t = lb.temperatures.historic.fill_gaps(lb.temperatures.historic.tmpr())

    data = t[climate_zone]  # using univariate feature(Only temperature for given time)
    data.plot()
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
    plt.savefig(os.path.join(LSTMPREDICTIONSDATAFOLDER, f"{title}.png"))


# function to plot train test loss
def plot_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Train Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title(title)
    plt.savefig(os.path.join(LSTMPREDICTIONSDATAFOLDER, f"{title}.png"))
    plt.legend()
    plt.grid()
    plt.show()


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
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def lstm_model():

    i = 20  # select a random sample to plot
    steps = 200
    EPOCHS = 20

    (
        xtrain,
        xval,
        ytrain,
        yval,
    ) = data_preprocessing()  # prepare original timeseries dataset

    plot_time_series(
        [xtrain[i], ytrain[i]], 0, f"{climate_zone} - Sample plot"
    )  # random sample plots
    plot_time_series(
        [xtrain[i], ytrain[i], MWA(xtrain[i])],
        0,
        f"{climate_zone} - Moving Window Average predicted",
    )

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
    # lstm_model.compile(
    #    optimizer="rmsprop", loss=root_mean_squared_error, metrics=["accuracy"])

    model_history = lstm_model.fit(
        trainset,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=validationset,
        validation_steps=50,
    )

    for i, j in validationset.take(5):
        plot = plot_time_series(
            [i[0].numpy(), j[0].numpy(), lstm_model.predict(i)[0]],
            0,
            f"{climate_zone} - LSTM UNIVARIATE",
        )
        plt.savefig(
            os.path.join(
                LSTMPREDICTIONSDATAFOLDER,
                f"{climate_zone} - LSTM single prediction.png",
            )
        )
        # plt.savefig(f"{climate_zone} - LSTM single prediction.png")

    # plot train and validation loss
    plot_loss(model_history, f"{climate_zone} -Training vs validation loss")


"""
# function to calculate the offset in each climate zone.
# offset = Actual temperature in database - Predicted temperature by LSTM model 
def offsetDataset(t: pd.DataFrame) -> pd.DataFrame:
    
    for i in range(1, 15):  # iterate through all timezones
        j = 2
        # one specific timezone

        t[i] = 1
    return
"""

if __name__ == "__main__":
    lstm_model()
