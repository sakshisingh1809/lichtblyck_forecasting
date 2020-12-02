# -*- coding: utf-8 -*-
"""
Created 2020-09

Investigating various ways of smoothing a piecewise contant function, while
retaining the piecewise average.

(Goal: smoothing the monthly prices for price-forward-curve).
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Iterable


# index: x-values. values: function value starting at that time.
num = 100  # number of intervals
s_original = pd.Series(
    np.random.normal(0, 1.1 + np.sin(np.linspace(0, 6 * np.pi, num))).cumsum(0),
    np.arange(num) + 0.1 * np.random.random(num),
)
ppi = 1000  # points-per-interval for approximated continuous function.

#%% Using fourier series

# Calculate the coefficients of the fitted function.
def fouriercoefficients(yvalues: Iterable, times: Iterable) -> np.ndarray:
    """Calculate the coefficients for a fourier approximation of a function that
    is piecewise constant, such that, in each piece, the average value of fourier
    approximation equals the original value.
    'yvalues' is an iterable of n values, 'times' is an iterable of n+1 values.
    yvalues[i] is understood to be the value for the original function between
    times[i] and times[i+1].
    """
    n = len(yvalues)
    coeffmtx = []
    rightside = []
    # Each piece adds one linear equation between the fourier coefficients.
    for val, tl, tr in zip(yvalues, times[:-1], times[1:]):

        def fouriereq(i, tr, tl):
            fr = np.fft.fftfreq(n)[i]
            if i == 0:
                return tr - tl
            return (
                1
                / (2 * np.pi * 1j * fr)
                * (np.exp(2 * np.pi * 1j * fr * tr) - np.exp(2 * np.pi * 1j * fr * tl))
            )

        coeffline = [fouriereq(i, tr, tl) for i in range(n)]
        coeffmtx.append(coeffline)
        rightsidevalue = val * (tr - tl)
        rightside.append(rightsidevalue)
    # Solving the linear equation.
    solution = np.linalg.solve(coeffmtx, rightside)
    return solution


# Calculate the value of the fitted function.
def fouriereval(t, coeff: np.array):
    """Evaluate the fourier approximation that is given by the coefficients in
    'coeff', at time t."""
    n = len(coeff)
    freqs = np.fft.fftfreq(n)
    return sum([c * np.exp(2 * np.pi * 1j * t * fr) for c, fr in zip(coeff, freqs)])


# Test
y = np.array([2, 3, -1, 4, 5])
x = np.array([0, 1.3, 2, 3, 4, 5.6])  # one value more to mark end time

coeff = fouriercoefficients(y, x)
x2 = np.linspace(x[0], x[-1], 10000, endpoint=False)
y2 = fouriereval(x2, coeff).real

fig, axes = plt.subplots()
axes.hlines(y, x[1:], x[:-1], "k")
axes.plot(x2, y2, "r")

# check if averages are equal
dx = x2[1] - x2[0]
avgs = [
    (
        val,
        val2 := sum(val2 * dx for val2, xx2 in zip(y2, x2) if xl <= xx2 < xr)
        / (xr - xl),
        val2 - val,
    )
    for val, xl, xr in zip(y, x[:-1], x[1:])
]

# %% Smoothing at month-boundary


def smooth_at_monthboundary(s_in: pd.Series, x_boundaries: Iterable):
    """Add smoothing so that curve s_in is continuously differentiable.
    x_boundaries hold index values of s_in at which gap needs to be closed/smoothed."""

    def smoothing_coefficients(y0, dydx0, y1, dydx1):
        """Calculate the coefficients c0, c1, ..., c4 for the polynomial that has
        value y0 and first derivative dydx0 at x=0, value y1 and first derivative
        dydx1 at x=1, and integral between x=0 and x=1 of 0."""
        coeffmtx = np.array(
            [
                [1, 0, 0, 0, 0],  # y0
                [0, 1, 0, 0, 0],  # dydx0
                [1, 1, 1, 1, 1],  # y1
                [0, 1, 2, 3, 4],  # dydx1
                [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5],
            ]
        )  # integral
        rightside = np.array([y0, dydx0, y1, dydx1, 0])
        return np.linalg.solve(coeffmtx, rightside)

    records = []
    for s in s_in.rolling(4):
        if len(s) != 4:
            continue
        if s.index[2] not in x_boundaries:
            continue
        x = s.index
        x_boundary = x[2]
        y = s.values
        dydx_l = (y[1] - y[0]) / (x[1] - x[0])
        dydx_r = (y[3] - y[2]) / (x[3] - x[2])
        Δy = y[2] - (
            y[1] + dydx_l * (x[2] - x[1])
        )  # extrapolate leftside to meet rightside
        Δdydx = dydx_r - dydx_l
        records.append((x_boundary, Δy, Δdydx))
    boundaries = (
        pd.DataFrame.from_records(records)
        .set_index(0)
        .rename({1: "Δy", 2: "Δdydx"}, axis=1)
    )
    change_intrvl = pd.DataFrame(index=boundaries.index)
    change_intrvl["y_l"] = -0.5 * boundaries["Δy"]
    change_intrvl["dydx_l"] = -0.5 * boundaries["Δdydx"]
    change_intrvl["y_r"] = 0.5 * boundaries["Δy"].shift(-1)
    change_intrvl["dydx_r"] = 0.5 * boundaries["Δdydx"].shift(-1)
    change_intrvl = change_intrvl.fillna(0)

    s_out = pd.Series([], [], dtype=np.float32)
    for left, right in zip(x_boundaries[:-1], x_boundaries[1:]):
        s = s_in[(s_in.index >= left) & (s_in.index < right)]
        t = (s.index - left) / (right - left)
        try:
            wanted = change_intrvl.loc[left, :]
        except KeyError:
            continue
        coeff = smoothing_coefficients(
            wanted["y_l"], wanted["dydx_l"], wanted["y_r"], wanted["dydx_r"]
        )
        change = sum(c * t ** i for i, c in enumerate(coeff))
        s_out = s_out.append(pd.Series(s.values + change, s.index))
    return s_out


#%% Calculate various interpolations, and plot.

# All interpolation methods have the condition, that they maintain the same
# piecewise average as the original function.

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

# Fit with polynomial with k coefficients (i.e., of degree k-1) -> k degrees of freedom.
# Conditions:
#   . Correct average value in month i
#   . Correct average values in surrounding (k-3) months
#   . Connect at midpoint between months i-1 and i, and i and i+1
polybound = pd.Series([], [], dtype=np.float32)
k = 4  # number of coefficients in y. (minimum: 4)
w = k - 2  # number of x-intervals needed for estimate.
val = lambda t: np.array([t ** i for i in range(k)]).T  # value
itg = lambda t: np.array([t ** (i + 1) / (i + 1) for i in range(k)]).T  # integration
for s in s_original.rolling(w + 1):
    if len(s) != w + 1:
        continue

    i = (w - 1) // 2  # index of middle interval, which we keep
    # function values
    y = s.values[:-1]  # num of values: w
    # make x-values dimensionless (in case they are e.g. as datetime) for fitting
    x = s.index  # num of values: w+1
    t = (x - x[0]) / (
        x[-1] - x[0]
    )  # turn x-values into dimensionless t-values, scaled onto [0, 1]
    t_left, t_right = t[:-1], t[1:]  # num of values: w

    # find coefficients
    coeffmtx = np.concatenate(
        [
            itg(t_right) - itg(t_left),  # the integrals
            val(t_left[i : i + 1]),  # value at start of middle interval
            val(t_right[i : i + 1]),  # value at end of middle interval
        ]
    )
    rightside = np.array(
        [*(y * (t_right - t_left)), (y[i - 1] + y[i]) / 2, (y[i] + y[i + 1]) / 2]
    )
    solution = np.linalg.solve(coeffmtx, rightside)
    calc_y2 = lambda t_val: solution.dot(t_val ** range(k))

    # Use fit function to estimate middle interval
    t2 = np.linspace(t[i], t[i + 1], ppi, endpoint=False)
    x2 = (
        t2 * (x[-1] - x[0]) + x[0]
    )  # turn dimensionless t-value back into correct x-values
    y2 = np.array([calc_y2(tt) for tt in t2])

    polybound = polybound.append(pd.Series(y2, x2))

axes[0, 0].plot(polybound, "#f00")
axes[0, 0].set_title(
    f"Per month: polynomial with {k} coefficients (i.e., degree {k-1}).\n(used rolling window of {w} months to estimate coefficients;\ncorrect integral in each month within window.)\nBoundary values fixed to midway points, so function is continuous."
)
axes[1, 0].plot(smooth_at_monthboundary(polybound, s_original.index), "#f06")
axes[1, 0].set_title(
    f"Polynomial as above, but smoothed\n(i.e., made continuously differentiable by adding\n5-coefficient-polynomial to each month)."
)


# Fit with polynomial with k coefficients (i.e., of degree k-1) -> k degrees of freedom.
# Conditions:
#   . Correct average value in month i
#   . Correct average values in surrounding (k-1) months
polynobound = pd.Series([], [], dtype=np.float32)
k = 5  # number of coefficients in y.
w = k  # number of x-intervals needed for estimate.
itg = lambda t: np.array([t ** (i + 1) / (i + 1) for i in range(k)]).T  # integration
for s in s_original.rolling(w + 1):
    if len(s) != w + 1:
        continue

    i = (w - 1) // 2  # index of middle interval, which we keep
    # function values
    y = s.values[:-1]  # num of values: w
    # make x-values dimensionless (in case they are e.g. as datetime) for fitting
    x = s.index  # num of values: w+1
    t = (x - x[0]) / (
        x[-1] - x[0]
    )  # turn x-values into dimensionless t-values, scaled onto [0, 1]
    t_left, t_right = t[:-1], t[1:]  # num of values: w

    # find coefficients
    coeffmtx = itg(t_right) - itg(t_left)
    rightside = y * (t_right - t_left)
    solution = np.linalg.solve(coeffmtx, rightside)
    calc_y2 = lambda t_val: solution.dot(t_val ** range(k))

    # Use fit function to estimate middle interval
    t2 = np.linspace(t[i], t[i + 1], ppi, endpoint=False)
    x2 = (
        t2 * (x[-1] - x[0]) + x[0]
    )  # turn dimensionless t-value back into correct x-values
    y2 = np.array([calc_y2(tt) for tt in t2])

    polynobound = polynobound.append(pd.Series(y2, x2))

axes[0, 1].plot(polynobound, "#0c0")
axes[0, 1].set_title(
    f"Per month: polynomial with {k} coefficients (i.e., degree {k-1}).\n(used rolling window of {w} months to estimate coefficients;\ncorrect integral in each month within window.)\nNo condition on boundary values, so function is not continuous."
)
axes[1, 1].plot(smooth_at_monthboundary(polynobound, s_original.index), "#6c0")
axes[1, 1].set_title(
    f"Polynomial as above, but smoothed\n(i.e., made continuously differentiable by adding\n5-coefficient-polynomial to each month)."
)


# Fit with fourier approximation with k coefficients -> k degrees of freedom
# Conditions:
#   . Correct average value in month i
#   . Correct average values in surrounding (k-1) months
fourier = pd.Series([], [], dtype=np.float32)
k = 5  # number of coefficients in y.
w = k  # number of x-intervals needed for estimate.
for s in s_original.rolling(w + 1):
    if len(s) != w + 1:
        continue

    i = (w - 1) // 2  # index of middle interval, which we keep
    # function values
    y = s.values[:-1]  # num of values: w
    # make x-values dimensionless (in case they are e.g. as datetime) for fitting
    x = s.index  # num of values: w+1
    t = (x - x[0]) / (
        x[-1] - x[0]
    )  # turn x-values into dimensionless t-values, scaled onto [0, 1]
    t_left, t_right = t[:-1], t[1:]  # num of values: w

    # find coefficients
    coeff = fouriercoefficients(y, t)

    # Use fit function to estimate middle interval
    t2 = np.linspace(t[i], t[i + 1], ppi, endpoint=False)
    x2 = (
        t2 * (x[-1] - x[0]) + x[0]
    )  # turn dimensionless t-value back into correct x-values
    y2 = np.array([fouriereval(tt, coeff).real for tt in t2])

    fourier = fourier.append(pd.Series(y2, x2))

axes[0, 2].plot(fourier, "#00f")
axes[0, 2].set_title(
    f"Per month: Fourier series with {k} coefficients.\n(used rolling window of {k} months to estimate coefficients;\ncorrect integral in each month within window.)\nNo condition on boundary values, so function is not continuous."
)
axes[1, 2].plot(smooth_at_monthboundary(fourier, s_original.index), "#06f")
axes[1, 2].set_title(
    f"Fourier series as above, but smoothed\n(i.e., made continuously differentiable by adding\n5-coefficient-polynomial to each month)."
)


# Fit with fourier approximation for entire timeperiod at once.
k = 20
s = s_original[0 : 0 + k]
y = s.values[:-1]
x = s.index
t = (x - x[0]) / (x[-1] - x[0])

# find coefficients
coeff = fouriercoefficients(y, t)

t2 = np.linspace(t[0], t[-1], k * ppi, endpoint=False)
x2 = t2 * (x[-1] - x[0]) + x[0]  # turn dimensionless t-value back into correct x-values
y2 = np.array([fouriereval(tt, coeff).real for tt in t2])

fouriersingle = pd.Series(y2, x2)

axes[2, 2].plot(fouriersingle, "#60f")
axes[2, 2].set_title(
    f"One fourier series for {k} values.\nContinuous but possibly badly behaved."
)


# Idea for future: by minimizing energy of bent beam (i.e., minimizing the cumulative angle change)
# Function consists of 2 terms:
#   polynomiel y(t) makes sure, each section has correct average.
#   fourier series x(t) is added to minimise energy.

for ax in axes.flatten():
    if ax.lines:
        ax.hlines(
            s_original.values[:-1],
            s_original.index[:-1],
            s_original.index[1:],
            "k",
            zorder=10,
        )
