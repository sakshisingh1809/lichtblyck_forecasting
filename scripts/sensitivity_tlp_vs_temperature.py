# -*- coding: utf-8 -*-
"""
Calculating the sensitivity of a tlp profile, in [%/degC]. 
Or, more precise: 1/Quantitiy * d(Quantity) / d(Temperature) in [1/degC]

2020-07-06 Ruud Wijtvliet
"""

import numpy as np
import pandas as pd
import lichtblick as lb
from lichtblick.tools import tools

# Get temperatures...
tmpr = lb.future.tmpr_standardized()
tmpr = tmpr[tmpr.index < '2021']

# ...and calculate geographic average; weigh with consumption / customer presence in each zone.
weights = pd.DataFrame({'power': [60,717,1257,1548,1859,661,304,0,83,61,0,1324,1131,0,21],
                        'gas': [729,3973,13116,28950,13243,3613,2898,0,1795,400,0,9390,3383,9,113]},
                       index=range(1,16)) #MWh/a in each zone
weights = weights['power'] / weights['power'].sum() + weights['gas'] / weights['gas'].sum()
tmpr['germany'] = tools.wavg(tmpr, weights, axis=1)

# Get the tlp profile... 
profile = lb.tlp.standardized_tmpr_loadprofile(3)
# ...and plot to verify.
wide = profile.unstack('tmpr') #or: wide = profile.reset_index().pivot(index='time_left_local', columns='tmpr', values='std_tmpr_lp')
wide.plot(colormap='coolwarm')

# Combine to get the expected consumption...
ref_consumption = lb.tlp.tmpr2load(profile, tmpr['germany'], 1).sum()
# ...as well as the consumption with temperature increase...
consumption = lb.tlp.tmpr2load(profile, tmpr['germany'] + 1, 1).sum()
# ...and calculate the sensitivity with them
sensitivity = consumption / ref_consumption - 1 #fraction per degC



# %% Additional analysis.

# Plot the dependence of consumption on temperature.
x = np.linspace(-5, 5, 101)
consumption = [lb.tlp.tmpr2load(profile, tmpr['germany'] + dt, 1).sum() for dt in x]
y = [c/ref_consumption - 1 for c in consumption]
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.plot(x, y)