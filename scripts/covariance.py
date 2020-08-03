"""
Module to calculate the historic covariance between spot price and tlp consumption.

Process:

expected temperature -> (with tlp:) expected offtake -> (with pfc:) hedge of expected offtake
  -> expected spot volume -> (with pfc:) expected spot costs
actual temperatures -> (with tlp:) actual offtake 
  -> (with hedge:) actual spot volume -> (with spot prices:) actual spot costs
delta offtake volume, caused by temperature: actual - expected
delta spot costs, caused by many things incl. temperature: actual - expected
Interested in: average spot costs in given year --> Expected covariance costs
And interested in: distribution of spot costs --> Temperature risk

Variations:
  1. Expected temperature = monthly average, i.e., same temperature for each day of a given month.
  2. Expected temperature = 'structured' temperature, i.e., already including some temperature path.

"""

# %% SETUP

import lichtblyck as lb
from lichtblyck import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Temperature weights
weights = pd.DataFrame({'power': [60,717,1257,1548,1859,661,304,0,83,61,0,1324,1131,0,21],
                        'gas': [729,3973,13116,28950,13243,3613,2898,0,1795,400,0,9390,3383,9,113]},
                       index=range(1,16)) #MWh/a in each zone
weights = weights['power'] / weights['power'].sum() + weights['gas'] / weights['gas'].sum()

# Temperature to load
t2l = lb.tlp.standardized_tmpr_loadprofile(3) + lb.tlp.standardized_tmpr_loadprofile(2)

ispeak = lb.prices.is_peak_hour

# %% EXPECTED: ~ 2 weeks before delivery month, without temperature influence in price.

# Expected temperature.
# . Variation 1: expected temperature is monthly average of previous 11 years.
t_exp = lb.historic.tmpr_monthlymovingavg(11)
t_exp = t_exp.dropna()
t_exp['t_germany'] = tools.wavg(t_exp, weights, axis=1)
t_exp = t_exp.resample('D').ffill() #TODO: use upsampling tool
df_daily = pd.DataFrame(t_exp['t_germany'].rename('t_exp'))
# . Variation 2: expected temperature is seasonality and trend (see emrald xlsx file)
ti = pd.date_range('2000-01-01', '2020-01-01', freq='D', tz='Europe/Berlin')
tau = (ti - pd.Timestamp('1900-01-01', tz='Europe/Berlin')).total_seconds()/3600/24/365.24
t_exp = pd.Series(5.83843470203356 + 0.037894551208033 * tau + -9.03387134093431 * 
                  np.cos(2*np.pi*(tau - 19.3661745382612/365.24)), index=ti)
df_daily = pd.DataFrame(t_exp.rename('t_exp'))

# Expected offtake.
w_exp = lb.tlp.tmpr2load(t2l, df_daily['t_exp'], spec=1000)
w_exp = w_exp.resample('H').mean()  #TODO: use downsampling tool
df_hourly = pd.DataFrame(w_exp.rename('w_exp'))

# Expected prices. Not necessary for calculations, but necessary for plots.
futures = lb.prices.futures('m')
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
p_fwd = futures.groupby('ts_left_deliv').apply(exp_and_act).unstack().dropna() #expected and actual
p_fwd = p_fwd.swaplevel(axis=1).sort_index(axis=1)
p_exp_po = p_fwd['exp'][['p_peak', 'p_offpeak']].resample('H').ffill()
p_exp = pd.Series(np.where(ispeak(p_exp_po.index), p_exp_po['p_peak'], p_exp_po['p_offpeak']),
                  p_exp_po.index, name='p_spot_exp')
df_hourly = df_hourly.join(p_exp, how='inner').dropna()

# Hedge.
# is_peak = (w_e.index.hour >= 8) & (w_e.index.hour < 20)
# w_hedge = pd.Series(index=w_e.index, dtype=np.float64)
# w_hedge[is_peak] = w_e[is_peak].mean()
# w_hedge[~is_peak] = w_e[~is_peak].mean()
w_hedge = w_exp.groupby(lambda ts: (ts.year, ts.month, ispeak(ts))).transform(np.mean)
df_hourly = df_hourly.join(w_hedge.rename('w_hedge'))
df_hourly['w_spot_exp'] = df_hourly['w_exp'] - df_hourly['w_hedge']

# Expected spot revenue (verification: should add to 0 for each given month)
duration = (df_hourly.index[1:] - df_hourly.index[:-1]).total_seconds()/3600 #get duration in h for each datapoint
duration = np.append(duration, np.median(duration)) #add duration of final datapoint (guessed)
df_hourly['duration'] = duration
df_hourly['r_spot_exp'] = df_hourly['w_spot_exp'] * df_hourly['duration'] * df_hourly['p_spot_exp']

# %% ACTUAL: after delivery month.

# Actual temperature.
t_act = lb.historic.tmpr()
t_act = lb.historic.fill_gaps(t_act)
t_act['t_germany'] = tools.wavg(t_act, weights, axis=1)
df_daily = df_daily.join(t_act['t_germany'].rename('t_act'), how='inner')

# Actual offtake.
w_act = lb.tlp.tmpr2load(t2l, t_act['t_germany'], spec=1000)
w_act = w_act.resample('H').mean() #TODO: use downsampling tool
df_hourly = df_hourly.join(w_act.rename('w_act'), how='inner')

# Actual spot quantities.
df_hourly['w_spot'] = df_hourly['w_act'] - df_hourly['w_hedge']

# Actual prices.
p_spot = lb.prices.spot()
df_hourly = df_hourly.join(p_spot.rename('p_spot'), how='inner')

# Actual spot revenue.
df_hourly['r_spot'] = df_hourly['w_spot'] * df_hourly['duration'] * df_hourly['p_spot']


# %% ADDITIONAL data and aggregations

#add
df_hourly['w_spot_diff'] = df_hourly['w_spot'] - df_hourly['w_spot_exp']
df_hourly['p_spot_diff'] = df_hourly['p_spot'] - df_hourly['p_spot_exp']
df_daily = df_daily.join(df_hourly.resample('D').mean(), how='inner')
#aggregate
df_monthly = df_daily.resample('MS').mean()
df_yearly = df_daily.resample('AS').mean()
for df in [df_daily, df_monthly, df_yearly]:
    # Correction: here sum is needed, not mean
    df['r_spot'] = df_hourly['r_spot'].resample(df.index.freq).sum()
    df['r_spot_exp'] = df_hourly['r_spot_exp'].resample(df.index.freq).sum()
    df['duration'] = df_hourly['duration'].resample(df.index.freq).sum()
    # New information
    df['p_par'] = df['r_spot'] / (df['w_act'] * df['duration'])
    df['p_par_exp'] = df['r_spot_exp'] / (df['w_exp'] * df['duration'])



# %% VERIFICATION OF DATA INTEGRITY

pop1 = p_spot.groupby(lambda ts: (ts.year, ts.month, ispeak(ts))).mean()
pop1.index = pd.MultiIndex.from_tuples(pop1.index, names=['YY','MM',''])
pop1 = pop1.unstack().rename({True: 'p_peak', False: 'p_offpeak'}, axis=1)

pop2 = p_fwd.set_index(pd.MultiIndex.from_arrays([p_fwd.index.year, 
    p_fwd.index.month], names=['YY', 'MM']))['act'][['p_peak', 'p_offpeak']]

# should be equal
pop1.join(pop2, how='inner', lsuffix='_fromspot', rsuffix='_fromfwd').plot()


# %% PLOT

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax

# plot_multi(df_hourly[['p_spot', 'w_spot']])





# %% PLOT: short example timeseries

# Filter time section
ts_left = pd.Timestamp('2019-04-01', tz='Europe/Berlin')
ts_right = ts_left + pd.offsets.MonthBegin(10)
def get_filter(ts_left, ts_right):
    def wrapped(df):
        return df[(df.index >= ts_left) & (df.index < ts_right)]
    return wrapped
filtr = get_filter(ts_left, ts_right)

df_display = filtr(df_hourly) #pick hourly or daily


plt.style.use('seaborn')
fig = plt.figure(figsize=(16,10))
axes = []
#degC
axes.append(plt.subplot2grid((4, 2), (0, 0), fig=fig))
#MW
axes.append(plt.subplot2grid((4, 2), (1, 0), fig=fig, sharex=axes[0], xticklabels=[]))
axes.append(plt.subplot2grid((4, 2), (2, 0), fig=fig, sharex=axes[0], xticklabels=[]))
axes.append(plt.subplot2grid((4, 2), (3, 0), fig=fig, sharex=axes[0], sharey=axes[1]))
#Eur/MWh
axes.append(plt.subplot2grid((4, 2), (2, 1), fig=fig, sharex=axes[0]))
axes.append(plt.subplot2grid((4, 2), (3, 1), fig=fig, sharex=axes[0], sharey=axes[4]))
#both
axes.append(plt.subplot2grid((4, 2), (0, 1), rowspan=2, fig=fig))


# Temperatures.
ax = axes[0]
ax.title.set_text("Temperature")
ax.yaxis.label.set_text("degC")
ax.plot(df_daily['t_exp'], 'b-', linewidth=0.5, label='expected')
ax.plot(df_daily['t_act'], 'r-', linewidth=0.5, label='actual')
ax.legend()

# Load.
ax = axes[1]
ax.title.set_text("Load")
ax.yaxis.label.set_text("MW")
ax.plot(df_display['w_exp'], 'b-', linewidth=0.5, label='expected')
ax.plot(df_display['w_hedge'], 'b--', alpha=0.4, linewidth=0.5, label='hedge')
ax.plot(df_display['w_act'], 'r-', linewidth=0.5, label='actual')
ax.legend()

# Spot quantity.
ax = axes[2]
ax.title.set_text("Spot quantity (+ = buy)")
ax.yaxis.label.set_text("MW")
ax.plot(df_display['w_spot_exp'], 'b-', linewidth=0.5, label='spot expected')
ax.plot(df_display['w_spot'], 'r-', linewidth=0.5, label='spot actual')
ax.legend()

# Spot quantity difference (spot actual vs expected).
ax = axes[3]
ax.title.set_text("Spot quantity: actual vs expected (+ = buy more (or sell less) than expected)")
ax.yaxis.label.set_text("MW")
ax.plot(df_display['w_spot_diff'], 'g-', linewidth=0.5, label='diff')
ax.legend()

# Spot prices.
ax = axes[4]
ax.title.set_text("Price of spot quantity")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df_display['p_spot_exp'], 'b-', linewidth=0.5, label='expected')
ax.plot(df_display['p_spot'], 'r-', linewidth=0.5, label='actual')
ax.legend()

# Spot price difference.
ax = axes[5]
ax.title.set_text("Spot price: actual vs expected (+ = higher than expected)")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df_display['p_spot_diff'], 'g-', linewidth=0.5, label='diff')
ax.legend()


# ax = axes[5]
# ax_new = ax.twinx()
# # ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))

# Spot price difference vs spot quantity difference.
ax = axes[6]
ax.title.set_text("Spot quantity deviation vs Spot price deviation")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.scatter(df_display['w_spot_diff'], df_display['p_spot_diff'], c='g', s=10, alpha=0.1)

axes[0].set_xlim(ts_left, ts_right)


# %% PLOT: per month or year


def plot_per_month_or_year(agg='M'):
    if agg.upper() == 'M':
        df = df_monthly
        w = 30
    else:
        df = df_yearly
        w = 350
    
    fig = plt.figure(figsize=(16,10))
    axes = []
    #degC
    axes.append(plt.subplot2grid((4, 2), (0, 0), fig=fig))
    #MWh
    axes.append(plt.subplot2grid((4, 2), (1, 0), fig=fig, sharex=axes[0], xticklabels=[]))
    axes.append(plt.subplot2grid((4, 2), (2, 0), fig=fig, sharex=axes[0], xticklabels=[]))
    axes.append(plt.subplot2grid((4, 2), (3, 0), fig=fig, sharex=axes[0]))
    #Eur/MWh
    axes.append(plt.subplot2grid((4, 2), (2, 1), fig=fig, sharex=axes[0]))
    axes.append(plt.subplot2grid((4, 2), (3, 1), fig=fig, sharex=axes[0]))
    #both
    axes.append(plt.subplot2grid((4, 2), (0, 1), rowspan=2, fig=fig))
    
    # Temperatures.
    ax = axes[0]
    ax.title.set_text("Temperature")
    ax.yaxis.label.set_text("degC")
    ax.plot(df['t_exp'], 'b-', linewidth=0.5, label='expected')
    ax.plot(df['t_act'], 'r-', linewidth=0.5, label='actual')
    ax.legend()
    
    # Load.
    ax = axes[1]
    ax.title.set_text("Load")
    ax.yaxis.label.set_text("MWh")
    ax.bar(df.index-pd.Timedelta(days=w/4), df['w_exp'] * df['duration'], width=w/2, color='b', label='expected')
    ax.hlines(df['w_hedge'] * df['duration'], df.index-pd.Timedelta(days=w/2), df.index+pd.Timedelta(days=w/2), color='b', alpha=0.4, label='hedge')
    ax.bar(df.index+pd.Timedelta(days=w/4), df['w_act'] * df['duration'], width=w/2, color='r', label='actual')
    ax.legend()
    
    # Spot quantity.
    ax = axes[2]
    ax.title.set_text("Spot quantity (+ = buy)")
    ax.yaxis.label.set_text("MWh")
    ax.bar(df.index-pd.Timedelta(days=w/4), df['w_spot_exp'] * df['duration'], width=w/2, color='b', label='spot expected')
    ax.bar(df.index+pd.Timedelta(days=w/4), df['w_spot'] * df['duration'], width=w/2, color='r', label='spot actual')
    ax.legend()
    
    # Spot quantity difference (spot actual vs expected).
    ax = axes[3]
    ax.title.set_text("Spot quantity: actual vs expected (+ = buy more (or sell less) than expected)")
    ax.yaxis.label.set_text("MWh")
    ax.bar(df.index, df['w_spot_diff'] * df['duration'], width=w/2, color='g', label='diff')
    ax.legend()
    
    # Spot prices.
    ax = axes[4]
    ax.title.set_text("Revenue of spot quantity")
    ax.yaxis.label.set_text("Eur")
    ax.bar(df.index-pd.Timedelta(days=w/4), df['r_spot_exp'], width=w/2, color='b', label='expected')
    ax.bar(df.index+pd.Timedelta(days=w/4), df['r_spot'], width=w/2, color='r', label='actual')
    ax.legend()
    
    # Spot price difference.
    ax = axes[5]
    ax.title.set_text("Temperature premium: actual vs expected (+ = higher than expected)")
    ax.yaxis.label.set_text("Eur/MWh")
    ax.plot(df['p_par_exp'], 'b-', linewidth=0.5, label='expected (verify: 0 on monthly basis)')
    ax.plot(df['p_par'], 'r-', linewidth=0.5, label='actual')
    ax.legend()
      
    # Spot price difference vs spot quantity difference.
    ax = axes[6]
    ax.title.set_text("Spot quantity deviation vs Spot price deviation")
    ax.xaxis.label.set_text("MWh")
    ax.yaxis.label.set_text("Eur/MWh")
    ax.scatter(df['w_spot_diff'] * df['duration'], df['p_par'], c='g', s=10, alpha=0.75)

plot_per_month_or_year('A')