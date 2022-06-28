#%%

import lichtblyck as lb
import pandas as pd

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")


#%% CALCULATE SENSITIVITY

tempr = lb.tmpr.norm.tmpr("2023", "2023-04")
weights = {
    zone: weight
    for weight, zone in zip(
        [37.87, 4.14, 9.36, 42.04, 6.59], ["t_4", "t_5", "t_3", "t_13", "t_7"]
    )
}
t_norm = lb.wavg(
    tempr[weights.keys()], pd.Series(weights), axis=1
)  # averaged over relevant zones
hdd_norm = (18 - t_norm).clip(0, 1000).sum()
hdd_cold = (18 - (t_norm - 1)).clip(0, 1000).sum()

tlp = lb.tlp.power.fromsource(4, spec=1)
offtake_norm = tlp(t_norm).sum()
offtake_cold = tlp(t_norm - 1).sum()
sensitivity = (offtake_cold / offtake_norm - 1) / (hdd_cold - hdd_norm)


#%% CALCULATE VOLUME AND PORTFOLIO PRICE

current = lb.portfolios.pfstate("power", "B2C_P2H_LEGACY", "2023", "2023-04")
current = current.asfreq("MS")


# %%
