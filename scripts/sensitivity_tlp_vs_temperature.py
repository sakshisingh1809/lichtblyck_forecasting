# -*- coding: utf-8 -*-
"""
Calculating the sensitivity of a tlp profile, in [%/degC]. 
Or, more precise: 1/Quantitiy * d(Quantity) / d(Temperature) in [1/degC]

2020-07-06 Ruud Wijtvliet
"""

import numpy as np
import pandas as pd
import lichtblyck_forecasting as lf

# Get temperatures...
tmpr = lf.future.tmpr_standardized()
tmpr = tmpr[tmpr.index < "2021"]

# ...and calculate geographic average; weigh with consumption / customer presence in each zone.
weights = pd.DataFrame(
    {
        "power": [60, 717, 1257, 1548, 1859, 661, 304, 0, 83, 61, 0, 1324, 1131, 0, 21],
        "gas": [
            729,
            3973,
            13116,
            28950,
            13243,
            3613,
            2898,
            0,
            1795,
            400,
            0,
            9390,
            3383,
            9,
            113,
        ],
    },
    index=range(1, 16),
)  # MWh/a in each zone
weights = (
    weights["power"] / weights["power"].sum() + weights["gas"] / weights["gas"].sum()
)
tmpr["t_germany"] = lf.wavg(tmpr, weights, axis=1)

# Get the tlp profile...
t2l = 0.9 * lf.tlp.standardized_tmpr_loadprofile(
    "bayernwerk_nsp"
) + 0.1 * lf.tlp.standardized_tmpr_loadprofile(
    "bayernwerk_wp"
)  # 2=nsp, 3=wp
# ...and plot to verify.
wide = t2l.unstack(
    "t"
)  # or: wide = profile.reset_index().pivot(index='time_left_local', columns='tmpr', values='std_tmpr_lp')
wide.plot(colormap="coolwarm")

# Combine to get the expected consumption...
q_ref = lf.tlp.tmpr2load(t2l, tmpr["t_germany"], 0.79e6).mean() * 8760
# ...as well as the consumption with temperature increase...
q_warmer = lf.tlp.tmpr2load(t2l, tmpr["t_germany"] + 1, 0.79e6).mean() * 8760
# ...and calculate the sensitivity with them.
sensitivity = q_warmer / q_ref - 1  # fraction per degC


# %% Additional analysis.

# Plot the dependence of consumption on temperature.
delta_t = np.linspace(-2, 2, 51)
q = [
    lf.tlp.tmpr2load(t2l, tmpr["t_germany"] + dt, 0.79e6).mean() * 8760 / 1e6
    for dt in delta_t
]  # in /1e6 to make TWh
import matplotlib.pyplot as plt

plt.style.use("seaborn")
fig, ax = plt.subplots()
ax.plot(delta_t, q)
ax.title.set_text("Sensitivity of volume in Ludwig-PF on temperature")
ax.yaxis.label.set_text("Volume [TWh/a]")
ax.xaxis.label.set_text("temperature deviation [deg C]")
