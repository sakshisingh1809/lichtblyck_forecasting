"""
Create standardized temperature year (res=1d, len=2020-2030) for 15 climate zones.
"""

import lichtblick as lb
from lichtblick.tools import tools
import pandas as pd
import numpy as np


# Get historic daily temperatures for each climate zones...
tmpr = pd.DataFrame()
for cz in range(1, 16):
    t = lb.historic.tmpr(cz)
    tmpr = tmpr.join(t.rename(cz), how='outer')
# NB: may contain gaps e.g. due to broken weather station.

# ...keep only 2005-2019...
tmpr = tmpr[(tmpr.index >= '2005') & (tmpr.index < '2020')]

# ...and find the most representative year for each month.
#    1: calculate monthly averages
mm_av = tmpr.groupby(tmpr.index.month).mean()
mm_av.index.rename('MM', inplace=True)
yymm_av = tmpr.groupby([tmpr.index.month, tmpr.index.year]).mean()
yymm_av.index.rename(['MM', 'YY'], inplace=True)
#    2: average; weigh with consumption / customer presence in each zone
weights = pd.DataFrame({'power': [60,717,1257,1548,1859,661,304,0,83,61,0,1324,1131,0,21],
                        'gas': [729,3973,13116,28950,13243,3613,2898,0,1795,400,0,9390,3383,9,113]},
                       index=range(1,16)) #MWh/a in each zone
weights = weights['power'] / weights['power'].sum() + weights['gas'] / weights['gas'].sum()
mm_av['germany'] = tools.wavg(mm_av, weights, axis=1)
yymm_av['germany'] = tools.wavg(yymm_av, weights, axis=1)
#    3: compare to find year lowest deviation for each month
yymm_av['delta'] = yymm_av.apply(
    lambda row: row['germany'] - mm_av['germany'][row.name[0]], axis=1)
idx = yymm_av['delta'].groupby('MM').apply(lambda df: df.apply(abs).idxmin())
bestfit = yymm_av.loc[idx, 'germany':'delta']

# Then, create single representative year from these individual months...
keep = tmpr.index.map(lambda idx: (idx.month, idx.year) in bestfit.index)
repryear = tmpr[keep]
repryear.index = repryear.index.map(lambda ts: (ts.month, ts.day))
repryear.index.rename(['MM', 'DD'], inplace=True)
if (2, 29) not in repryear.index:  #add 29 feb if doesn't exist yet.
    toadd = pd.Series(
        repryear[repryear.index.map(lambda idx: idx[0] == 2)].mean(),
        name = (2, 29))
    repryear = repryear.append(toadd)
repryear = repryear.sort_index()
# ...decompose into 'current' monthly averages and (zero-averaged) structure...
tmpr2012 = repryear.groupby('MM').mean()
tmprstruct = repryear - tmpr2012
# ...and also find the future monthly averages.
for cz in range(1, 16):
    t = lb.future.tmpr(cz).rename(cz)
    if cz == 1:
        tmpr2045 = pd.DataFrame(t)
    else:
        tmpr2045 = tmpr2045.join(t, how='outer')
tmpr2045 = tmpr2045.groupby('MM').mean() #Per month and climate zone.

# Finally, combine into a daily time series with 'standardized' temperatures.
year_start = 2020
year_end = 2030
idxTs = pd.date_range(start=pd.Timestamp(year=year_start, month=1, day=1),
                       end=pd.Timestamp(year=year_end+1, month=1, day=1),
                       closed='left', freq='D', tz='Europe/Berlin')
idxY = idxTs.map(lambda ts: ts.year).rename('YY')
idxM = idxTs.map(lambda ts: ts.month).rename('MM')
idxMD = idxTs.map(lambda ts: (ts.month, ts.day)).rename(['MM', 'DD'])
factor2045 = pd.Series((np.arange(2012, 2046) - 2012) / (2045 - 2012),
                       index=range(2012, 2046)).rename_axis('YY')
f = factor2045.loc[idxY]
f.index = idxTs
stdz_tmpr = tmprstruct.loc[idxMD].set_index(idxTs) \
    + tmpr2012.loc[idxM].set_index(idxTs).multiply(1 - f, axis=0) \
    + tmpr2045.loc[idxM].set_index(idxTs).multiply(f, axis=0)


stdz_tmpr.set_index(idxTs.map(lambda ts: (ts.year, ts.month, ts.day))).to_excel('tempr.xlsx')
