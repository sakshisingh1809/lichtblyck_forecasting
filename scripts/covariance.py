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
import matplotlib as mpl
import datetime
from scipy.stats import norm


# Temperature weights
weights = pd.DataFrame({'power': [60,717,1257,1548,1859,661,304,0,83,61,0,1324,1131,0,21],
                        'gas': [729,3973,13116,28950,13243,3613,2898,0,1795,400,0,9390,3383,9,113]},
                       index=range(1,16)) #MWh/a in each zone
weights = weights['power'] / weights['power'].sum() + weights['gas'] / weights['gas'].sum()

# Temperature to load
t2l = 0.9*lb.tlp.standardized_tmpr_loadprofile(2) \
    + 0.1*lb.tlp.standardized_tmpr_loadprofile(3)   #2=nsp, 3=wp
t2l.unstack().plot(cmap='coolwarm')

ispeak = lb.prices.is_peak_hour

specfc_el_load = 0.79e6 


# %% ACTUAL: after delivery month.

# Actual temperature.
t = lb.historic.tmpr()
t = lb.historic.fill_gaps(t)
t['t_germany'] = tools.wavg(t, weights, axis=1)
df_daily = pd.DataFrame(t['t_germany'].rename('t'))

# Actual offtake.
w = lb.tlp.tmpr2load(t2l, t['t_germany'], spec=specfc_el_load)
w = w.resample('H').mean()
df_hourly = pd.DataFrame(w.rename('w'))

# Actual spot prices.
p_spot = lb.prices.spot()
df_hourly = df_hourly.join(p_spot.rename('p_spot'), how='inner')


# %% EXPECTED: ~ 2 weeks before delivery month, without temperature influence in price.

# Expected temperature.
# . Variation 1: expected temperature is monthly average of previous 11 years.
# t_exp = lb.historic.tmpr_monthlymovingavg(11)
# t_exp = t_exp.dropna()
# t_exp['t_germany'] = tools.wavg(t_exp, weights, axis=1)
# t_exp = t_exp.resample('D').ffill()
# t = pd.DataFrame(t_exp['t_germany'].rename('t_exp'))
# . Variation 2: expected temperature is seasonality and trend (see emrald xlsx file)
ti = pd.date_range('2000-01-01', '2020-01-01', freq='D', tz='Europe/Berlin')
tau = (ti - pd.Timestamp('1900-01-01', tz='Europe/Berlin')).total_seconds()/3600/24/365.24
t_exp = pd.Series(5.83843470203356 + 0.037894551208033 * tau + -9.03387134093431 * 
                  np.cos(2*np.pi*(tau - 19.3661745382612/365.24)), index=ti)
df_daily = df_daily.join(t_exp.rename('t_exp'), how='inner')

# Expected offtake.
w_exp = lb.tlp.tmpr2load(t2l, df_daily['t_exp'], spec=specfc_el_load)
w_exp = w_exp.resample('H').mean()
df_hourly = df_hourly.join(w_exp.rename('w_exp'), how='inner')

# Expected spot prices.
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
p_fwd = futures.groupby('ts_left_deliv').apply(exp_and_act).unstack().dropna() #expected and actual
p_fwd = p_fwd.swaplevel(axis=1).sort_index(axis=1)
p_exp_po = p_fwd['exp'][['p_peak', 'p_offpeak']].resample('H').ffill()
p_exp_m = pd.Series(np.where(ispeak(p_exp_po.index), p_exp_po['p_peak'], p_exp_po['p_offpeak']),
                    p_exp_po.index, name='p_spot_m_exp')
# Use actual spot prices to calculate expected M2H profile.
p = pd.DataFrame(p_spot)
p['p_spot_m'] = lb.prices.p_bpo_long(p_spot, 'MS')
p['p_spot_m2h'] = p['p_spot'] - p['p_spot_m']
rolling_av = lambda nums: sum(np.sort(nums)[3:-3]) / (len(nums) - 6) #mean but without the extremes
p_exp_m2h = p.groupby(p.index.map(lambda ts: (ts.weekday(), ts.hour)))\
    .apply(lambda df: df['p_spot_m2h'].rolling(75).apply(rolling_av).shift())\
    .droplevel(0).resample('H').asfreq()
p_exp_m2h -= p_exp_m2h.groupby(lambda ts: (ts.year, ts.month, ispeak(ts))).transform(np.mean) #arbitrage free
df_hourly = df_hourly.join((p_exp_m + p_exp_m2h).rename('p_spot_exp'), how='inner').dropna()



b = p_fwd['exp'][['p_peak', 'p_offpeak']]


### TODO: ADD BETTER PFC (df_hourly)






# %% Derived quantities

# Duration.
duration = (df_hourly.index[1:] - df_hourly.index[:-1]).total_seconds()/3600 #get duration in h for each datapoint
duration = np.append(duration, np.median(duration)) #add duration of final datapoint (guessed)
df_hourly['duration'] = duration

# Hedge.
df_hourly['w_hedge'] = lb.prices.w_hedge_long(df_hourly['w_exp'], df_hourly['p_spot_exp'], 'MS')

# Expected spot quantities.
df_hourly['w_spot_exp'] = df_hourly['w_exp'] - df_hourly['w_hedge']
# Expected spot revenue (verification: should add to 0 for each given month)
df_hourly['r_spot_exp'] = df_hourly['w_spot_exp'] * df_hourly['duration'] * df_hourly['p_spot_exp']

# Actual spot quantities.
df_hourly['w_spot'] = df_hourly['w'] - df_hourly['w_hedge']
# Actual spot revenue.
df_hourly['r_spot'] = df_hourly['w_spot'] * df_hourly['duration'] * df_hourly['p_spot']

# Difference.
df_hourly['w_diff'] = df_hourly['w'] - df_hourly['w_exp']
df_hourly['p_spot_diff'] = df_hourly['p_spot'] - df_hourly['p_spot_exp']
df_hourly['r_par'] = df_hourly['w_spot_exp'] * df_hourly['duration'] * df_hourly['p_spot_diff']
df_hourly['r_covar'] = df_hourly['w_diff'] * df_hourly['duration'] * df_hourly['p_spot_diff']


# %% Aggregations

#only keep full year
start = df_hourly.index[0] + pd.offsets.YearBegin(0)
df_hourly = df_hourly[df_hourly.index >= start]
#aggregate
hourly = df_hourly
daily = df_hourly.resample('D').mean().join(df_daily, how='inner')
monthly = df_hourly.resample('MS').mean().join(df_daily.resample('MS').mean(), how='inner')
yearly = df_hourly.resample('AS').mean().join(df_daily.resample('AS').mean(), how='inner')
for df in [daily, monthly, yearly]:
    # Correction: here sum is needed, not mean
    df['r_spot'] = df_hourly['r_spot'].resample(df.index.freq).sum()
    df['r_spot_exp'] = df_hourly['r_spot_exp'].resample(df.index.freq).sum()
    df['r_covar'] = df_hourly['r_covar'].resample(df.index.freq).sum()
    df['r_par'] = df_hourly['r_par'].resample(df.index.freq).sum()
    df['duration'] = df_hourly['duration'].resample(df.index.freq).sum()
    # New information
    df['p_covar'] = df['r_covar'] / (df['w'] * df['duration'])
    df['p_par'] = df['r_par'] / (df['w'] * df['duration'])


# %% VERIFICATION OF DATA INTEGRITY

# pop1 = p_spot.groupby(lambda ts: (ts.year, ts.month, ispeak(ts))).mean()
# pop1.index = pd.MultiIndex.from_tuples(pop1.index, names=['YY','MM',''])
# pop1 = pop1.unstack().rename({True: 'p_peak', False: 'p_offpeak'}, axis=1)

# pop2 = p_fwd.set_index(pd.MultiIndex.from_arrays([p_fwd.index.year, 
#     p_fwd.index.month], names=['YY', 'MM']))['act'][['p_peak', 'p_offpeak']]

# # should be equal
# pop1.join(pop2, how='inner', lsuffix='_fromspot', rsuffix='_fromfwd').plot()


# %% PLOT: short example timeseries

# Filter time section
ts_left = pd.Timestamp('2007-01-01', tz='Europe/Berlin')
ts_right = ts_left + pd.offsets.MonthBegin(1)
def get_filter(ts_left, ts_right):
    def wrapped(df):
        return df[(df.index >= ts_left) & (df.index < ts_right)]
    return wrapped
filtr = get_filter(ts_left, ts_right)
df = filtr(hourly)

# Values
r_covar = df['r_covar'].sum() 
r_par = df['r_par'].sum()
q = (df['w'] * df['duration']).sum() 

# Formatting
plt.style.use('seaborn')
#positive formatting (costs>0), negative formatting (costs<0)
pos = {'color':'black', 'alpha':0.1}

fig = plt.figure(figsize=(16,10))
axes = []
#Expected and actual
#  degC
axes.append(plt.subplot2grid((3, 3), (0, 0), fig=fig))
#  MW
axes.append(plt.subplot2grid((3, 3), (1, 0), fig=fig, sharex=axes[0]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (2, 0), fig=fig, sharex=axes[0]))
#PaR
#  scatter
axes.append(plt.subplot2grid((3, 3), (0, 1), fig=fig))
#  MW
axes.append(plt.subplot2grid((3, 3), (1, 1), fig=fig, sharex=axes[0], sharey=axes[1]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (2, 1), fig=fig, sharex=axes[0], sharey=axes[2]))
#Covar
#  scatter
axes.append(plt.subplot2grid((3, 3), (0, 2), fig=fig))
#  MW
axes.append(plt.subplot2grid((3, 3), (1, 2), fig=fig, sharex=axes[0], sharey=axes[1]))
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (2, 2), fig=fig, sharex=axes[0], sharey=axes[2]))

#Expected and actual
#  Temperatures.
ax = axes[0]
ax.title.set_text("Temperature")
ax.yaxis.label.set_text("degC")
ax.plot(daily['t_exp'], 'b-', linewidth=0.5, label='expected')
ax.plot(daily['t'], 'r-', linewidth=0.5, label='actual')
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake")
ax.yaxis.label.set_text("MW")
ax.plot(df['w_exp'], 'b-', linewidth=0.5, label='expected')
ax.plot(df['w_hedge'], 'b--', alpha=0.4, linewidth=0.5, label='hedge')
ax.plot(df['w'], 'r-', linewidth=0.5, label='actual')
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df['p_spot_exp'], 'b-', linewidth=0.5, label='expected')
ax.plot(df['p_spot'], 'r-', linewidth=0.5, label='actual')
ax.legend()

#PaR
#  Times with positive contribution (costs > 0).
times = ((df['r_par'] > 0).shift() - (df['r_par'] > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[3]
ax.title.set_text("Expected spot quantity vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(df['w_spot_exp'].min(),0), 0), min(df['p_spot_diff'].min(),0), 0, **pos)
ax.fill_between((max(df['w_spot_exp'].max(),0), 0), max(df['p_spot_diff'].max(),0), 0, **pos)
ax.scatter(df['w_spot_exp'], df['p_spot_diff'], c='orange', s=10, alpha=0.5,
           label=f'r_par   = {r_par/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par = {r_par/q:.2f} Eur/MWh')
ax.legend()
#  MW.
ax = axes[4]
ax.title.set_text("Expected spot quantity (+ = buy)\n(unhedged volume before month)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df['w_spot_exp'], 'b-', linewidth=0.5, label='spot expected')
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text("Short-term change in price (+ = higher than expected)\n(spot vs pfc)")
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df['p_spot_diff'], c='purple', linestyle='-', linewidth=0.5, label='diff')
ax.legend()

#Covar
#  Times with positive contribution (costs > 0).
times = ((df['r_covar'] > 0).shift() - (df['r_covar'] > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[6]
ax.title.set_text("Short-term change in offtake quantity vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(df['w_diff'].min(),0), 0), min(df['p_spot_diff'].min(),0), 0, **pos)
ax.fill_between((max(df['w_diff'].max(),0), 0), max(df['p_spot_diff'].max(),0), 0, **pos)
ax.scatter(df['w_diff'], df['p_spot_diff'], c='green', s=10, alpha=0.5,
           label=f'r_covar = {r_covar/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_covar = {r_covar/q:.2f} Eur/MWh')
ax.legend()
#  MW.
ax = axes[7]
ax.title.set_text("Short-term change in offtake quantity (+ = more than expected)\n(offake due to actual temperatures vs due to expected temperatures)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df['w_diff'], c='purple', linestyle='-', linewidth=0.5, label='diff')
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("Short-term change in price (+ = higher than expected)\n(spot vs pfc)")
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df['p_spot_diff'], c='purple', linestyle='-', linewidth=0.5, label='diff')
ax.legend()

axes[0].set_xlim(ts_left, ts_right)
fig.tight_layout()


# %% PLOT: per month or year

# Formatting
plt.style.use('seaborn')
#positive formatting (costs>0), negative formatting (costs<0)
pos = {'color':'black', 'alpha':0.1}

fig = plt.figure(figsize=(16,10))
axes = []
#Expected and actual.
#  degC.
axes.append(plt.subplot2grid((3, 3), (0, 0), fig=fig))
#  MWh.
axes.append(plt.subplot2grid((3, 3), (1, 0), fig=fig, sharex=axes[0]))
#  Eur/MWh.
axes.append(plt.subplot2grid((3, 3), (2, 0), fig=fig, sharex=axes[0]))
#PaR.
#  distribution.
axes.append(plt.subplot2grid((3, 3), (2, 1), fig=fig))
#  Eur.
axes.append(plt.subplot2grid((3, 3), (0, 1), fig=fig, sharex=axes[0]))
#  Eur/MWh.
axes.append(plt.subplot2grid((3, 3), (1, 1), fig=fig, sharex=axes[0]))
#Covar.
#  distribution.
axes.append(plt.subplot2grid((3, 3), (2, 2), fig=fig, sharex=axes[3]))
#  Eur.
axes.append(plt.subplot2grid((3, 3), (0, 2), fig=fig, sharex=axes[0], sharey=axes[4]))
#  Eur/MWh.
axes.append(plt.subplot2grid((3, 3), (1, 2), fig=fig, sharex=axes[0], sharey=axes[5]))

#Expected and actual
#  Temperatures.
ax = axes[0]
ax.title.set_text("Temperature, monthly average")
ax.yaxis.label.set_text("degC")
ax.plot(monthly['t_exp'], 'b-', linewidth=1, label='expected')
ax.plot(monthly['t'], 'r-', linewidth=1, label='actual')
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake, per month")
ax.yaxis.label.set_text("MWh")
ax.plot(monthly['w_exp'] * monthly['duration'], 'b-', linewidth=1, label='expected')
ax.plot(monthly['w_hedge'] * monthly['duration'], 'b--', alpha=0.4, linewidth=1, label='hedge')
ax.plot(monthly['w'] * monthly['duration'], 'r-', linewidth=1, label='actual')
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Spot base price, per month")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(monthly['p_spot_exp'], 'b-', linewidth=1, label='expected') 
ax.plot(monthly['p_spot'], 'r-', linewidth=1, label='actual')
ax.legend()

#PaR
#  Distribution.
source_vals = yearly['p_par']
loc, scale = norm.fit(source_vals)
x = source_vals.sort_values().tolist()
x_fit = np.linspace(1.5*min(x)-0.5*max(x), -0.5*min(x)+1.5*max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[3]
ax.title.set_text("par premium (+ = additional costs), per year")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(x, cumulative=True, color='orange', density=True, bins=x+[x[-1]+0.1], histtype='step', linewidth=1)
ax.plot(x_fit, y_fit, color='k', linewidth=1, label=f"Fit:\nmean: {loc:.2f}, std: {scale:.2f}")
ax.legend()
#  Eur.
ax = axes[4]
ax.title.set_text("par revenue (+ = additional costs), per month")
ax.yaxis.label.set_text("Eur")
ax.plot(monthly['r_par'], color='orange', linestyle='-', linewidth=1, label='r_par')
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text("par premium (+ = additional costs), per year")
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(yearly.index, yearly['p_par'], width=365*0.9, color='orange', label='p_par')
ax.legend()

#Covar
#  Distribution.
source_vals = yearly['p_covar']
loc, scale = norm.fit(source_vals)
x = source_vals.sort_values().tolist()
x_fit = np.linspace(1.5*min(x)-0.5*max(x), -0.5*min(x)+1.5*max(x), 300)
y_fit = norm(loc, scale).cdf(x_fit)
ax = axes[6]
ax.title.set_text("covar premium (+ = additional costs), per year")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.hist(x, cumulative=True, color='green', density=True, bins=x+[x[-1]+0.1], histtype='step', linewidth=1)
ax.plot(x_fit, y_fit, color='k', linewidth=1, label=f"Fit:\nmean: {loc:.2f}, std: {scale:.2f}")
ax.legend()
#  Eur.
ax = axes[7]
ax.title.set_text("covar revenue (+ = additional costs), per month")
ax.yaxis.label.set_text("Eur")
ax.plot(monthly['r_covar'], color='green', linestyle='-', linewidth=1, label='r_covar')
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("covar premium (+ = additional costs), per year")
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(yearly.index, yearly['p_covar'], width=365*0.9, color='green', label='p_covar')
ax.legend()


# %% CALCULATE: conditional value at risk

p_par_pricedin = 1.12
p_covar_pricedin = 1.56
yearly['r_par_loss'] = yearly['r_par'] - yearly['w'] * yearly['duration'] * p_par_pricedin
yearly['r_covar_loss'] = yearly['r_covar'] - yearly['w'] * yearly['duration'] * p_covar_pricedin


# %% PLOT: delta q vs delta p_spot

agg = 'A'
if agg.upper().startswith('M'):
    agg = 'MS'
elif agg.upper().startswith('A'):
    agg = 'AS'
else:
    raise ValueError('"agg" must start with M (monthly) or A (yearly).')
data = hourly[['w_diff', 'p_spot_diff']].resample(agg)

limits = hourly[['w_diff', 'p_spot_diff']].apply(lambda x: (x.quantile(0.05), x.quantile(0.95)))
limits = limits.abs().mean()
limits = pd.DataFrame([-limits, limits])

datapoint_count = np.median([len(df) for _, df in data])
bins = int((np.sqrt(datapoint_count/10) // 2) * 4)
axCount = len(data)
colCount = int(np.ceil(np.sqrt(axCount)))
rowCount = int(np.ceil(axCount/colCount))
fig, axes = plt.subplots(rowCount, colCount, sharex=True, sharey=True, figsize=(16, 10))
for (ts_left, df), ax in zip(data, axes.flatten()):
    ax.title.set_text(ts_left.date())
    ax.hist2d(df['w_diff'], df['p_spot_diff'], bins=bins, range=limits.values.T, cmap='jet')
    #axes
    ax.plot(limits['w_diff'], [0, 0], 'k')
    ax.plot([0, 0], limits['p_spot_diff'], 'k')
axes[0, 0].set_xlim(*limits['w_diff'])
axes[0, 0].set_ylim(*limits['p_spot_diff'])

# %% Analysis: spot prices, actual vs expectation

fig = plt.figure(figsize=(16,10))
ax = plt.subplot()
for msk, color, label in zip((hourly['w_exp'] < 28, 
                             (hourly['w_exp'] >= 28) & (hourly['w_exp'] < 130), 
                             (hourly['w_exp'] >= 130)),
                             ('yellow', 'orange', 'red'),
                             ('wenig', 'mittel', 'viel')):
    x = hourly['p_spot_diff'][msk].sort_values().tolist()
    loc, scale = norm.fit(x)
    ax.hist(x, cumulative=True, color=color, density=True, bins=x, histtype='step', linewidth=1, label=f'{label} Leistung: mittelwert: {loc:.2f} Eur/MWh')
ax.title.set_text("Spot price vs expectation (+ = Spot price is higher)")
ax.yaxis.label.set_text("cumulative fraction")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.xaxis.label.set_text("Eur/MWh")
ax.legend()
ax.set_xlim(-30, 30)
