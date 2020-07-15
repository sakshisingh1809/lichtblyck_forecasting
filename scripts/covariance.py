"""
Module to calculate the historic covariance between spot price and tlp consumption.
"""

import lichtblyck as lb
from lichtblyck import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collect and massage input data...
#   1: prices
spot = lb.prices.spot()
#   2: temperatures
t = lb.historic.tmpr()
t = t[(t.index >= spot.index.min()) & (t.index <= spot.index.max())]
weights = pd.DataFrame({'power': [60,717,1257,1548,1859,661,304,0,83,61,0,1324,1131,0,21],
                        'gas': [729,3973,13116,28950,13243,3613,2898,0,1795,400,0,9390,3383,9,113]},
                       index=range(1,16)) #MWh/a in each zone
weights = weights['power'] / weights['power'].sum() + weights['gas'] / weights['gas'].sum()
t['t_germany'] = tools.wavg(t, weights, axis=1)
#   3: load
t2l = lb.tlp.standardized_tmpr_loadprofile('Avacon_HZ0')
load = lb.tlp.tmpr2load(t2l, tmpr['germany'], spec=1)
load = load.resample('H').mean()



plt.plot(spot, load, 'bo', markersize=1, alpha=0.1)

expected_tmpr = lb.historic.tmpr_struct()



# expected temperature -> 