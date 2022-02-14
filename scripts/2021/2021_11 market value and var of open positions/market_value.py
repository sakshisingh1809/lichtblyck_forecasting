"""
Create overview of the open positions of the portfolios. 
Step 1: volume and market value.
Step 2: value at risk.
"""

#%%

import lichtblyck as lb
import math
import pandas as pd
from scipy.stats import norm

#%% Volume and market value of open positions.
ts_left = lb.tools.stamps.ceil_ts(pd.Timestamp.now(), 0, 'AS')
lb.belvis.auth_with_password('Ruud.Wijtvliet', 'Ammm1mmm2mmm3mmm')
pfs = lb.portfolios.pfstate('power', 'B2C_P2H', ts_left)
unsourced_current = pfs.changefreq('MS').unsourced

#%% Value at risk.
u = pfs.hedge_of_unsourced('MS', 'val', True)
vola = 0.83 # /calyear
caldays = 3
calyears = caldays/365.24
# Distribution.
sigma = vola * math.sqrt(calyears) # as fraction
mu = - 0.5 * sigma ** 2 # as fraction
for label, quantile in {'min': 0.05, 'max': 0.95}.items():
    exponent = norm(mu, sigma).ppf(quantile)
    multiplication_factor = math.exp(exponent)
    change_factor = multiplication_factor - 1
    unsourced_var = unsourced_current.volume * (unsourced_current.price * change_factor)
        

# %%
