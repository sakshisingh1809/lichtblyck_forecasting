"""Create timeseries with norm temperatures for past and future dates."""


from . import historic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def create_fourier_series(x, y, Nh):
    f = np.array(
        [
            (1 if i == 0 else 2)
            * (
                (y * np.exp(-1j * 2 * i * np.pi * x)).sum()
                / (y * np.exp(-1j * 2 * i * np.pi * x)).size
            )
            * np.exp(1j * 2 * i * np.pi * x)
            for i in range(0, Nh)
        ]
    )
    return f.sum()


""" def create_fourier_series2(x, fourier_coeff, n_coeff, omega=np.pi * 2):
    fourier_series = np.array([0.0 for _ in range(len(x))])
    for n, coeff in enumerate(fourier_coeff[:n_coeff]):
        fourier_series += [(coeff * np.exp(-xx * 1j * omega * n)).real for xx in x]
    return fourier_series
 """


def tmpr(ts_left, ts_right, n_coeff: int = 2):
    """
    Return the expected (i.e., norm) daily temperature for each climate zone.

    Parameters
    ----------
    ts_left, ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of time period (left-closed).
    ncoeff: int, the number of coefficients for evaluating the Fourier transformation.
        Default is 2, is it is the best number of coefficients with least error rate.

    Returns
    -------
    Dataframe
        With daily temperature values. Index: timestamp (daily). Columns: climate zones
        (1..15). Values: norm temperature for corresponding day and climate zone in
        degC.
    """
    t = historic.fill_gaps(historic.tmpr(ts_left, ts_right))
    climate_zone = "t_3"  # Hamburg
    y = signal.detrend(t[climate_zone].values, type="linear")
    x = np.array(
        (t.index - pd.Timestamp("2000-01-01", tz=t.index.tz))
        / pd.Timedelta(days=365.25)
    )

    """fourier_coeff = []
    input = pd.Series(y, x)
    fast_fourier_transform = np.fft.fft(input)
    freq = np.fft.fftfreq(len(input))
    fs = 365.25  # sample rate: pattern is repeating every year
    n_bins = (len(freq) / fs) * 100
    for n in range(0, n_coeff):
        a = 0.5 * fast_fourier_transform[n].real / n_bins
        b = -0.5 * fast_fourier_transform[n].imag / n_bins
        fourier_coeff.append([a, b]) """

    yfit = np.array([create_fourier_series(t, y, n_coeff).real for t in x])
    print("Mean squared error is:", (np.round(((y - yfit) ** 2).mean(axis=0), 4)))
    # pd.DataFrame({"data": y, "fit": yfit}, x).plot()
    plot_fourier_coeff(x, y, yfit, n_coeff, climate_zone)
    return yfit


def plot_fourier_coeff(x, y, fourier_series, n_coeff, climate_zone):
    plt.title("{cz}".format(cz=climate_zone))
    plt.plot(x, y, color="grey", lw=0.5)
    plt.plot(
        x,
        fourier_series,
        color="b",
        lw=0.5,
        label="fourier comp: {c}".format(c=n_coeff),
    )
    plt.legend()
