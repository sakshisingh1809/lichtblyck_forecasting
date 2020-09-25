# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:14:10 2020

Investigating various ways of smoothing a piecewise contant function, while
retaining the piecewise average.

(Goal: smoothing the monthly prices for PFC).

@author: ruud.wijtvliet
"""

from matplotlib import pyplot as plt
import numpy as np
import lichtblyck as lb
from lichtblyck import tools
import datetime
import pandas as pd
from typing import Iterable

futures = lb.prices.frontmonth()
def exp_and_act(df):
    """Returns expected (pre-tmpr-influence) and actual (including-tmpr-influence)
    base, peak, offpeak prices, as well as respective trading days."""
    # Columns to keep.
    cols = df.columns[tools.is_price(df.columns)]
    cols = np.append(cols, 'ts_left_trade')
    df = df.reset_index('ts_left_trade') #to get 'ts_left_trade' in columns.
    df = df.sort_values('trade_before_deliv')
    data = []
    exp = df[df['trade_before_deliv'] > datetime.timedelta(15)]
    if not exp.empty:
        data.append(exp[cols].iloc[0].rename('exp'))
    act = df[df['trade_before_deliv'] < datetime.timedelta(-20)]
    if not act.empty:
        data.append(act[cols].iloc[0].rename('act'))
    return pd.DataFrame(data)
p_fwd = futures.groupby('ts_left_deliv').apply(exp_and_act).unstack().dropna()
p_fwd = p_fwd.swaplevel(axis=1).sort_index(axis=1)['exp']
p_fwd = p_fwd.resample('MS').asfreq()


#%% Using fourier series

#Calculate the coefficients of the fitted function.
def fouriercoefficients(values:Iterable, times:Iterable) -> np.ndarray:
    """Calculate the coefficients for a fourier approximation of a function that
    is piecewise constant, such that, in each piece, the average value of fourier 
    approximation equals the original value.
    'values' is an iterable of n values, 'times' is an iterable of n+1 values. 
    values[i] is understood to be the value for the original function between 
    times[i] and times[i+1].
    """
    n = len(values)
    coeffmtx = []
    rightside = []
    #Each piece adds one linear equation between the fourier coefficients.
    for val, tl, tr in zip(values, times[:-1], times[1:]):
        def fouriereq(i, tr, tl):
            fr = np.fft.fftfreq(n)[i]
            if i == 0:
                return tr-tl
            return 1/(2*np.pi*1j*fr)*(np.exp(2*np.pi*1j*fr*tr)-np.exp(2*np.pi*1j*fr*tl))
        coeffline = [fouriereq(i, tr, tl) for i in range(n)]
        coeffmtx.append(coeffline)
        rightsidevalue = val*(tr-tl)
        rightside.append(rightsidevalue)
    #Solving the linear equation.
    solution = np.linalg.solve(coeffmtx, rightside)
    return solution

#Calculate the value of the fitted function.
def fouriereval(t, coeff:np.array):
    """Evaluate the fourier approximation that is given by the coefficients in
    'coeff', at time t."""
    n = len(coeff)
    freqs = np.fft.fftfreq(n)
    return sum([c * np.exp(2*np.pi*1j*t*fr) for c, fr in zip(coeff, freqs)])


# Test
y = np.array([2, 3, -1, 4, 5])
t = np.array([0, 1.3, 2, 3, 4, 5.6]) #one value more to mark end time

coeff = fouriercoefficients(y, t)
t2 = np.linspace(t[0], t[-1], 10000, endpoint=False)
y2 = fouriereval(t2, coeff).real

fig, axes = plt.subplots()
axes.hlines(y, t[1:], t[:-1], 'k')
axes.plot(t2, y2, 'r')

#check if averages are equal
dt = t2[1] - t2[0]
avgs = [(val, 
         val2:=sum(val2 * dt for val2, tt2 in zip(y2, t2) if tl <= tt2 < tr) / (tr-tl),
         val2-val)
        for val, tl, tr in zip(y, t[:-1], t[1:])]

# %% Smoothing at month-boundary

def smooth_at_monthboundary(s_in):
    """Add smoothing so that curve s_in is continuously differentiable"""

    def smoothing_coefficients(f0, dfdx0, f1, dfdx1):
        """Calculate the coefficients c0, c1, ..., c4 for the polynomial that has
        value f0 and first derivative dfdx0 at x=0, value f1 and first derivative
        dfdx1 at x=1, and integral between x=0 and x=1 of 0."""
        coeffmtx = np.array([[1, 0, 0, 0, 0], #f0
                             [0, 1, 0, 0, 0], #dfdx0
                             [1, 1, 1, 1, 1], #f1
                             [0, 1, 2, 3, 4], #dfdx1
                             [1, 1/2, 1/3, 1/4, 1/5]]) #integral
        rightside = np.array([f0, dfdx0, f1, dfdx1, 0])
        return np.linalg.solve(coeffmtx, rightside)

    records = []
    for s in s_in.rolling(4):
        if len(s) != 4:
            continue
        if s.index[1].month == s.index[2].month:
            continue
        ts = s.index[2]
        dfl = s.values[1] - s.values[0]
        dfr = s.values[3] - s.values[2]
        Δf = s.values[2] - (s.values[1] + dfl) #extrapolate leftside to meet rightside
        Δdf = dfr - dfl
        records.append((ts, Δf, Δdf))
    boundaries = pd.DataFrame.from_records(records).set_index(0)\
        .rename({1:'Δf', 2:'Δdf'}, axis=1).rename_axis('ts', axis=0)
    wanted_changes = pd.DataFrame(index=boundaries.index)
    wanted_changes['f_l']  = -0.5 * boundaries['Δf']
    wanted_changes['df_l'] = -0.5 * boundaries['Δdf']
    wanted_changes['f_r']  =  0.5 * boundaries['Δf'].shift(-1)
    wanted_changes['df_r'] =  0.5 * boundaries['Δdf'].shift(-1)
    wanted_changes = wanted_changes.fillna(0)
    s_out = pd.Series([], [], dtype=np.float32)
    for ts, s in s_in.groupby(pd.Grouper(freq='MS')):
        t = (s.index - s.index[0])/((s.index[-1] - s.index[0]) * (len(s) + 1)/len(s)) #[0..1)
        try:
            wanted = wanted_changes.loc[ts]
        except:
            continue
        coeff = smoothing_coefficients(wanted['f_l'], wanted['df_l']*30,
                                       wanted['f_r'], wanted['df_r']*30)
        change = sum(c * t**i for i, c in enumerate(coeff))
        s_out = s_out.append(pd.Series(s.values + change, s.index))
    return s_out
    
    
    


#%% Calculate various interpolations, and plot.

# All interpolation methods have the condition, that they maintain the same
# piecewise average as the original function.

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

# # Fit with quadratic -> 3 degrees of freedom.
# # Conditions: 
# #   . Correct average value in month i
# #   . Connect at midpoint between months i-1 and i, and i and i+1
# p_fwd_interp1 = pd.Series([], [], dtype=np.float32)
# for df in p_fwd['p_base'].rolling(3):
#     if len(df) != 3:
#         continue
#     #Timestamps mapped to t1 -> 0 and t2 -> 1
#     t0, t1, t2 = (df.index - df.index[1]) / (df.index[2] - df.index[1])
#     a0, a1, a2 = df.values
#     coeffmtx = np.array([[1, t1, t1**2], #average on left boundary
#                          [1, t2, t2**2], #average on right boundary
#                          [1, (t2+t1)/2,(t2**2 + t2*t1 + t1**2)/3]]) #integral
#     rightside = np.array([(a0 + a1)/2, (a1 + a2)/2, a1])
#     solution = np.linalg.solve(coeffmtx, rightside)
    
#     idx = pd.date_range(df.index[1], df.index[2], freq='H', closed='left')
#     t = (idx - df.index[1]) / (df.index[2] - df.index[1])
#     values = solution.dot(np.array([[1]*len(t), t, t**2]))
    
#     p_fwd_interp1 = p_fwd_interp1.append(pd.Series(values, idx))
    
# axes[0,0].plot(p_fwd_interp1, '#f00')
# axes[0,0].set_title('Per month: polynomial with 3 coefficients (i.e., quadratic).\nAt boundary: continuous, not differentiable')
   

# Fit with polynomial with k coefficients (i.e., of degree k-1) -> k degrees of freedom.
# Conditions: 
#   . Correct average value in month i
#   . Correct average values in surrounding (k-3) months
#   . Connect at midpoint between months i-1 and i, and i and i+1
polybound = pd.Series([], [], dtype=np.float32)
polybound_avgd = pd.Series([], [], dtype=np.float32)
k = 5 # number of coefficients in y. 
w = k-2 # number of time-intervals needed for estimate.
val = lambda t: np.array([t**i for i in range(k)]).T
itg = lambda t: np.array([t**(i+1)/(i+1) for i in range(k)]).T
for df in p_fwd['p_base'].rolling(w+1):
    if len(df) != k-1:
        continue

    av_values = df.values[:-1] #num of values: w
    t = (df.index - df.index[0]) / (df.index[1] - df.index[0]) #num of values: w+1
    t_left, t_right = t[:-1], t[1:] #num of values: w
    i = (w-1) //2 #index of middle interval, which we keep
       
    coeffmtx = np.concatenate([
        itg(t_right) - itg(t_left), #the integrals
        val(t_left[i:i+1]), #value at start of middle interval
        val(t_right[i:i+1]) #value at end of middle interval
    ]) 
    rightside = np.array([
        *(av_values * (t_right - t_left)), 
        (av_values[0] + av_values[1])/2, 
        (av_values[1] + av_values[2])/2
    ])
    solution = np.linalg.solve(coeffmtx, rightside)
    y = lambda t: solution.dot(t ** range(k))
    
    #Use fit function to estimate middle time interval
    idx = pd.date_range(df.index[i], df.index[i+i], freq='D', closed='left')
    t2 = np.linspace(t[i], t[i+1], len(idx), endpoint=False)
    values = [y(tt) for tt in t2]
    s = pd.Series(values, idx)

    polybound = polybound.append(s)


axes[0,0].plot(polybound, '#f00')
axes[0,0].set_title(f'Per month: polynomial with {k} coefficients (i.e., degree {k-1}).\n(used rolling window of {w} months to estimate coefficients;\ncorrect integral in each month within window.)\nBoundary values fixed to midway points, so function is continuous.')
axes[1,0].plot(smooth_at_monthboundary(polybound), '#f06')
axes[1,0].set_title(f'Polynomial as above, but smoothed\n(i.e., made continuously differentiable by adding\n5-coefficient-polynomial to each month).')


# Fit with polynomial with k coefficients (i.e., of degree k-1) -> k degrees of freedom.
# Conditions: 
#   . Correct average value in month i
#   . Correct average values in surrounding (k-1) months
polynobound = pd.Series([], [], dtype=np.float32)
polynobound_avgd = pd.Series([], [], dtype=np.float32)
k = 5 # number of coefficients in y.
w = k # number of time-intervals needed for estimate.
avg = 3 # if avg == 1, only include estimate from 1 window.
itg = lambda t: np.array([t**(i+1)/(i+1) for i in range(k)]).T
for df in p_fwd['p_base'].rolling(w+1):
    if len(df) != k+1:
        continue

    av_values = df.values[:-1] #num of values: w
    t = (df.index - df.index[0]) / (df.index[1] - df.index[0]) #num of values: w+1
    t_left, t_right = t[:-1], t[1:] #num of values: w
    i = (w-1) //2 #index of middle interval, which we keep

    #coeffmtx = np.array([(t_right**(kk+1) - t_left**(kk+1))/(kk+1) for kk in range(k)]).T
    coeffmtx = itg(t_right) - itg(t_left)
    rightside = av_values * (t_right - t_left)
    solution = np.linalg.solve(coeffmtx, rightside)
    y = lambda t: solution.dot(t ** range(k))
    
    #Use fit function to estimate middle time interval
    i_left = i - avg//2
    i_right = i + (avg+1)//2
    idx = pd.date_range(df.index[i_left], df.index[i_right], freq='D', closed='left')
    t2 = np.linspace(t[i_left], t[i_right], len(idx), endpoint=False)
    values = [y(tt) for tt in t2]
    s = pd.Series(values, idx)
    
    polynobound = polynobound.append(s[(s.index >= df.index[i]) & (s.index < df.index[i+1])])
    polynobound_avgd = polynobound_avgd.append(s)

polynobound_avgd = polynobound_avgd.groupby(polynobound_avgd.index).mean()

axes[0,1].plot(polynobound, '#0c0')
axes[0,1].set_title(f'Per month: polynomial with {k} coefficients (i.e., degree {k-1}).\n(used rolling window of {w} months to estimate coefficients;\ncorrect integral in each month within window.)\nNo condition on boundary values, so function is not continuous.')
axes[1,1].plot(smooth_at_monthboundary(polynobound), '#6c0')
axes[1,1].set_title(f'Polynomial as above, but smoothed\n(i.e., made continuously differentiable by adding\n5-coefficient-polynomial to each month).')


# Fit with fourier approximation with k coefficients -> k degrees of freedom
# Conditions: 
#   . Correct average value in month i
#   . Correct average values in surrounding (k-1) months
fourier = pd.Series([], [], dtype=np.float32)
fourier_avgd = pd.Series([], [], dtype=np.float32)
k = 5 # number of coefficients in y. Same as number of time-intervals needed for estimate.
avg = 3 # if avg == 1, only include estimate from 1 window.
for df in p_fwd['p_base'].rolling(k+1):
    if len(df) != k+1:
        continue
    
    av_values = df.values[:-1] #num of values: k
    #times, normalised to 0 for t[0] and 1 for t[1]
    t_vec = (df.index - df.index[0]) / (df.index[1] - df.index[0]) #num of values: k+1
    t_vec_left, t_vec_right = t_vec[:-1], t_vec[1:] #num of values: k

    coeff = fouriercoefficients(av_values, t_vec)
    
    #Use fit function to estimate middle time interval
    #  Find index of middle interval
    i = (k-1) //2
    #  Find index of left and right timestamp
    i_left = i - avg//2
    i_right = i + (avg+1)//2
    idx = pd.date_range(df.index[i_left], df.index[i_right], freq='D', closed='left')
    t = np.linspace(t_vec[i_left], t_vec[i_right], len(idx), endpoint=False)
    values = [fouriereval(tt, coeff).real for tt in t]
    s = pd.Series(values, idx)
    
    fourier = fourier.append(s[(s.index >= df.index[i]) & (s.index < df.index[i+1])])
    fourier_avgd = fourier_avgd.append(s)
    
fourier_avgd = fourier_avgd.groupby(fourier_avgd.index).mean()

axes[0,2].plot(fourier, '#00f')
axes[0,2].set_title(f'Per month: Fourier series with {k} coefficients.\n(used rolling window of {k} months to estimate coefficients;\ncorrect integral in each month within window.)\nNo condition on boundary values, so function is not continuous.')
axes[1,2].plot(smooth_at_monthboundary(fourier), '#06f')
axes[1,2].set_title(f'Fourier series as above, but smoothed\n(i.e., made continuously differentiable by adding\n5-coefficient-polynomial to each month).')


# Fit with fourier approximation for entire timeperiod at once.
k = 100
df = p_fwd['p_base'][0:0+k]
values = df.values[:-1]
t = (df.index - df.index[0]) / (df.index[1] - df.index[0])
coeff = fouriercoefficients(values, t)

idx = pd.date_range(df.index[0], df.index[-1], freq='D', closed='left')
t2 = (idx - df.index[0]) / (df.index[1] - df.index[0])
values_interp = [fouriereval(tt, coeff).real for tt in t2]
fouriersingle = pd.Series(values_interp, idx)

axes[2,2].plot(fouriersingle, '#60f')
axes[2,2].set_title(f'One fourier series for {k} values.\nContinuous but possibly badly behaved.')


# Idea for future: by minimizing energy of bent beam (i.e., minimizing the cumulative angle change)
# Function consists of 2 terms: 
#   polynomiel y(t) makes sure, each section has correct average.
#   fourier series x(t) is added to minimise energy.

for ax in axes.flatten():
    if ax.lines:
        ax.hlines(p_fwd['p_base'], p_fwd.index, p_fwd.index + p_fwd.index.freq, 'k', zorder=10)





#%%


def resample(df, freq, func):
    if type(df.index) == pd.DatetimeIndex and df.index.name.startswith('ts_left'):
        #add one row
        idx = [df.index[-1] + df.index.freq]
        if type(df) == pd.DataFrame:
            df = df.append(pd.DataFrame([[None] * len(df.columns)], idx))
        elif type(df) == pd.Series:
            df = df.append(pd.Series([None], idx))
        df = df.resample(freq).apply(func)
        return df.iloc[:-1]
    return df.resample(freq).apply(func)


u = df.resample('H')
u.ffill