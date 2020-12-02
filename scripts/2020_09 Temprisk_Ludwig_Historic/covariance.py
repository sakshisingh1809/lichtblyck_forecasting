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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from scipy.stats import norm



exp = pd.DataFrame(columns=[[],[]]) #2-level columns
act = pd.DataFrame(columns=[[],[]]) #2-level columns


# %% TEMPERATURE INFLUENCE

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
t_act = lb.tools.wavg(t, weights, axis=1)
act[('envir', 't')] = t_act.resample('H').ffill() #TODO: use new/own resample function

# Actual offtake.
w_act = lb.tlp.tmpr2load(t2l, t_act, spec=specfc_el_load)
act[('offtake', 'w')] = w_act.resample('H').mean()

# Actual spot prices.
act[('spot', 'p')] = lb.prices.spot()

act = act.dropna() #drop rows for which temperature is available but price is not (or vice versa)


# %% EXPECTED: ~ 2 weeks before delivery month, without temperature influence in price.

# Expected temperature.
# . Variation 1: expected temperature is monthly average of previous 11 years.
# t_exp = lb.historic.tmpr_monthlymovingavg(11)
# t_exp = t_exp.dropna()
# t_exp['t_germany'] = lb.tools.wavg(t_exp, weights, axis=1)
# t_exp = t_exp.resample('D').ffill()
# exp[('envir', 't1')] = t_exp('t_germany').resample('H').ffill()
# . Variation 2: expected temperature is seasonality and trend (see emrald xlsx file)
ti = pd.date_range('2000-01-01', '2020-01-01', freq='D', tz='Europe/Berlin', closed='left')
tau = (ti - pd.Timestamp('1900-01-01', tz='Europe/Berlin')).total_seconds()/3600/24/365.24
t_exp = pd.Series(5.83843470203356 + 0.037894551208033 * tau + -9.03387134093431 * 
                  np.cos(2*np.pi*(tau - 19.3661745382612/365.24)), index=ti)
exp[('envir', 't')] = t_exp.resample('H').ffill() #TODO: use new/own resample function

# Expected offtake.
w_exp = lb.tlp.tmpr2load(t2l, t_exp, spec=specfc_el_load)
exp[('offtake', 'w')] = w_exp.resample('H').mean()

# Expected spot prices.
# . Use futures prices to calculate the expected average price level.
futures = lb.prices.frontmonth()
def exp_and_act(df):
    """Returns expected (pre-tmpr-influence) and actual (including-tmpr-influence)
    base, peak, offpeak prices, as well as respective trading days."""
    # Columns to keep.
    cols = df.columns[lb.tools.is_price(df.columns)]
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
# . Use actual spot prices to calculate expected M2H profile.
p = pd.DataFrame({'p_spot': lb.prices.spot()})
p['p_spot_m'] = lb.prices.p_bpo_long(p.p_spot, 'MS')
p['p_spot_m2h'] = p['p_spot'] - p['p_spot_m']
rolling_av = lambda nums: sum(np.sort(nums)[3:-3]) / (len(nums) - 6) #mean but without the extremes
p_exp_m2h = p.groupby(p.index.map(lambda ts: (ts.weekday(), ts.hour)))\
    .apply(lambda df: df['p_spot_m2h'].rolling(75).apply(rolling_av).shift())\
    .droplevel(0).resample('H').asfreq()
p_exp_m2h -= p_exp_m2h.groupby(lambda ts: (ts.year, ts.month, ispeak(ts))).transform(np.mean) #arbitrage free
# . Add together to get expected prices.
exp[('spot', 'p')] = p_exp_m + p_exp_m2h



### TODO: ADD BETTER PFC (df_hourly)
b = p_fwd['exp'][['p_peak', 'p_offpeak']]








# %% Derived quantities

# Combine.
hourly = pd.concat([exp, act], axis=1, keys=['exp', 'act']).dropna()

# Hedge.
hourly[('pf', 'hedge', 'w')] = lb.prices.w_hedge_long(
    hourly.exp.offtake.w, hourly.exp.spot.p, 'MS')

# Expected spot quantities.
hourly[('exp', 'spot', 'w')] = hourly.exp.offtake.w - hourly.pf.hedge.w
# check: spot revenue should add to 0 for each given month
# hourly.exp.spot.r.resample('MS').sum()

# Actual spot quantities.
hourly[('act', 'spot', 'w')] = hourly.act.offtake.w - hourly.pf.hedge.w

# Difference.
hourly[('delta', 'offtake', 'w')] = hourly.act.offtake.w - hourly.exp.offtake.w
hourly[('delta', 'spot', 'p')] = hourly.act.spot.p - hourly.exp.spot.p
hourly[('act', 'par', 'r')] = hourly.exp.spot.q * hourly.delta.spot.p
hourly[('act', 'covar', 'r')] = hourly.delta.offtake.q * hourly.delta.spot.p


# %% Aggregations

#only keep full year
start = hourly.index[0] + pd.offsets.YearBegin(0)
hourly = hourly[hourly.index >= start]
#aggregate
daily = hourly.resample('D').mean()
monthly = hourly.resample('MS').mean()
yearly = hourly.resample('AS').mean()
for df in [daily, monthly, yearly]:
    # Correction: here sum is needed, not mean
    df[('act', 'covar', 'r')] = hourly.act.covar.r.resample(df.index.freq).sum()
    df[('act', 'par', 'r')] = hourly.act.par.r.resample(df.index.freq).sum()
    # New information
    df[('act', 'covar', 'p')] = df.act.covar.r / df.act.offtake.q
    df[('act', 'par', 'p')] = df.act.par.r / df.act.offtake.q


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
r_covar = df.act.covar.r.sum() 
r_par = df.act.par.r.sum()
q = df.act.offtake.q.sum()

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
ax.plot(df.exp.envir.t, 'b-', linewidth=0.5, label='expected')
ax.plot(df.act.envir.t, 'r-', linewidth=0.5, label='actual')
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake")
ax.yaxis.label.set_text("MW")
ax.plot(df.exp.offtake.w, 'b-', linewidth=0.5, label='expected')
ax.plot(df.pf.hedge.w, 'b--', alpha=0.4, linewidth=0.5, label='hedge')
ax.plot(df.act.offtake.w, 'r-', linewidth=0.5, label='actual')
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Price")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(df.exp.spot.p, 'b-', linewidth=0.5, label='expected')
ax.plot(df.act.spot.p, 'r-', linewidth=0.5, label='actual')
ax.legend()

#PaR
#  Times with positive contribution (costs > 0).
times = ((df.act.par.r > 0).shift() - (df.act.par.r > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[3]
ax.title.set_text("Expected spot quantity vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(df.exp.spot.w.min(),0), 0), min(df.delta.spot.p.min(),0), 0, **pos)
ax.fill_between((max(df.exp.spot.w.max(),0), 0), max(df.delta.spot.p.max(),0), 0, **pos)
ax.scatter(df.exp.spot.w, df.delta.spot.p, c='orange', s=10, alpha=0.5,
           label=f'r_par   = {r_par/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_par = {r_par/q:.2f} Eur/MWh')
ax.legend()
#  MW.
ax = axes[4]
ax.title.set_text("Expected spot quantity (+ = buy)\n(unhedged volume before month)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.exp.spot.w, 'b-', linewidth=0.5, label='spot expected')
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text("Short-term change in price (+ = higher than expected)\n(spot vs pfc)")
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.spot.p, c='purple', linestyle='-', linewidth=0.5, label='diff')
ax.legend()

#Covar
#  Times with positive contribution (costs > 0).
times = ((df.act.covar.r > 0).shift() - (df.act.covar.r > 0)).dropna()
times.index -= times.index.freq * 0.5
start = times[times == -1].index
end = times[(times == 1) & (times.index > start[0])].index
#  Scatter.
ax = axes[6]
ax.title.set_text("Short-term change in offtake quantity vs short-term change in price")
ax.xaxis.label.set_text("MW")
ax.yaxis.label.set_text("Eur/MWh")
ax.fill_between((min(df.delta.offtake.w.min(),0), 0), min(df.delta.spot.p.min(),0), 0, **pos)
ax.fill_between((max(df.delta.offtake.w.max(),0), 0), max(df.delta.spot.p.max(),0), 0, **pos)
ax.scatter(df.delta.offtake.w, df.delta.spot.p, c='green', s=10, alpha=0.5,
           label=f'r_covar = {r_covar/1000:.0f} kEur\nofftake = {q/1000:.0f} GWh\np_covar = {r_covar/q:.2f} Eur/MWh')
ax.legend()
#  MW.
ax = axes[7]
ax.title.set_text("Short-term change in offtake quantity (+ = more than expected)\n(offake due to actual temperatures vs due to expected temperatures)")
ax.yaxis.label.set_text("MW")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.offtake.w, c='purple', linestyle='-', linewidth=0.5, label='diff')
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("Short-term change in price (+ = higher than expected)\n(spot vs pfc)")
ax.yaxis.label.set_text("Eur/MWh")
for s, e in zip(start, end):
    ax.axvspan(s, e, **pos)
ax.plot(df.delta.spot.p, c='purple', linestyle='-', linewidth=0.5, label='diff')
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
#  Eur/MWh
axes.append(plt.subplot2grid((3, 3), (1, 2), fig=fig, sharex=axes[0], sharey=axes[5]))

#Expected and actual
#  Temperatures.
ax = axes[0]
ax.title.set_text("Temperature, monthly average")
ax.yaxis.label.set_text("degC")
ax.plot(monthly.exp.envir.t, 'b-', linewidth=1, label='expected')
ax.plot(monthly.act.envir.t, 'r-', linewidth=1, label='actual')
ax.legend()
#  Offtake.
ax = axes[1]
ax.title.set_text("Offtake, per month")
ax.yaxis.label.set_text("MWh")
ax.plot(monthly.exp.offtake.q, 'b-', linewidth=1, label='expected')
ax.plot(monthly.pf.hedge.q, 'b--', alpha=0.4, linewidth=1, label='hedge')
ax.plot(monthly.act.offtake.q, 'r-', linewidth=1, label='actual')
ax.legend()
#  Prices.
ax = axes[2]
ax.title.set_text("Spot base price, per month")
ax.yaxis.label.set_text("Eur/MWh")
ax.plot(monthly.exp.spot.p, 'b-', linewidth=1, label='expected') 
ax.plot(monthly.act.spot.p, 'r-', linewidth=1, label='actual')
ax.legend()

#PaR
#  Distribution.
source_vals = yearly.act.par.p
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
ax.plot(monthly.act.par.r, color='orange', linestyle='-', linewidth=1, label='r_par')
ax.legend()
#  Eur/MWh.
ax = axes[5]
ax.title.set_text("par premium (+ = additional costs), per year")
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(yearly.index, yearly.act.par.p, width=365*0.9, color='orange', label='p_par')
ax.legend()

#Covar
#  Distribution.
source_vals = yearly.act.covar.p
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
ax.plot(monthly.act.covar.r, color='green', linestyle='-', linewidth=1, label='r_covar')
ax.legend()
#  Eur/MWh.
ax = axes[8]
ax.title.set_text("covar premium (+ = additional costs), per year")
ax.yaxis.label.set_text("Eur/MWh")
ax.bar(yearly.index, yearly.act.covar.p, width=365*0.9, color='green', label='p_covar')
ax.legend()


# %% CALCULATE: conditional value at risk

p_par_pricedin = 1.12
p_covar_pricedin = 1.56
yearly[('delta', 'par', 'r')] = yearly.act.par.r - yearly.act.offtake.q * p_par_pricedin
yearly[('delta', 'covar', 'r')] = yearly.act.covar.r - yearly.act.offtake.q * p_covar_pricedin


#%%

u = hourly[[('exp', 'offtake', 'w'), ('act', 'offtake', 'w'), ('delta', 'offtake', 'w')]]
ax = u.delta.offtake.w.abs().hist(bins=1000, cumulative=True, density=True, linewidth=1, histtype='step')
ax.set_xlabel('deviation [MW]')
ax.set_ylabel('cumulative fraction')
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

u.delta.offtake.w.mean()
