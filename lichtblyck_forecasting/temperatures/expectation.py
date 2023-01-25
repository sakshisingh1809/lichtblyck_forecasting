"""
Module used for finding the Expectation value of temperatures at each climatezones.
"""
from sourcedata.climate_zones import forallzones
from . import historic
from ..tools import stamps

# import numpy as np
import pandas as pd
import datetime
from typing import Union
import datetime as dt
import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

set_random_seed(0)


def avg_tmpr(df, cz):
    y = df[cz]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(y, marker=".", linestyle="-", linewidth=0.5, label="tmp")
    ax.plot(
        y.resample("M").mean(), markersize=8, linestyle="-", label="Mnthly Mean tmp"
    )
    ax.set_ylabel("tmp")
    ax.legend()


def plot_forecast(n, forecast):
    return n.plot(forecast)  # , plotting_backend="plotly")


def plot_forecast_components(n, forecast):
    return n.plot_components(forecast)


def plot_forecast_parameters(n, forecast):
    return n.plot_parameters(forecast)


def plot_loss(model):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(model["RMSE"], "-b", label="Training Loss")
    ax.plot(model["RMSE_val"], "-r", label="Validation Loss")


def forecast_tmpr(t, climate_zone, p=730):

    # if f < t.index[-1]:
    #    return "The third parameter should be in the future"

    df = t.copy()
    df["date"] = df.index.date
    df = df[["date", climate_zone]]
    df.dropna(inplace=True)
    df.columns = ["ds", "y"]
    df.reset_index(drop=True, inplace=True)

    # p = f - df["ds"].iloc[-1]
    n = NeuralProphet(
        growth="linear",
        n_changepoints=0,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=1000,
        global_normalization=True,
        # learning_rate=0.001,
    )
    """ n_lags=3 * 24,
    num_hidden_layers=4,
    d_hidden=32,
    learning_rate=0.003,
    # impute_missing=True,)
    ) """
    # n.set_plotting_backend("plotly")

    df_train, df_test = n.split_df(df, freq="D", valid_p=0.2)
    model = n.fit(df_train, validation_df=df_test, progress="plot-all")
    future = n.make_future_dataframe(df, periods=p, n_historic_predictions=len(df))
    forecast = n.predict(future)
    plot_forecast(n, forecast)
    plot_forecast_parameters(n, forecast)

    # plot_forecast_components(n, forecast)
    # forecast.yhat1.plot()
    # plot_loss(model)
    return model, forecast


def _tmpr(climate_zone: int, ts_left, ts_right) -> pd.Series:

    t = historic.fill_gaps(historic.tmpr(ts_left, ts_right))
    forecast = forecast_tmpr(t, climate_zone, 365)
    # avg_tmpr(t, climate_zone)
    # ARIMA_model(t, climate_zone)
    # return pd.Series(values, i, name="t")


def tmpr(
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Return the expected daily temperature for each climate zone.

    Parameters
    ----------
    ts_left, ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of time period (left-closed).

    Returns
    -------
    Dataframe
        With daily temperature values. Index: timestamp (daily). Columns: climate zones
        (1..15). Values: norm temperature for corresponding day and climate zone in
        degC.
    """
    # Fix timestamps (if necessary).
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    return forallzones(lambda cz: _tmpr(cz, ts_left, ts_right))
