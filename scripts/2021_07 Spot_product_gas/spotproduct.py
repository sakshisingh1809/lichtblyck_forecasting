"""
Calculate the "profile risk" of using an incorrect temperature station.

A TLP customer's consumption is calculated with the tmpr of a nearby weather
station, by the DSO. This profile is scaled with the customers cumulative consumption
at the end of the year. 
Lichtblick might take a different weather station AND a different TLP. The resulting
offtake profile is scaled to the same cumulative value [MWh/a], but the individual 
consumption values are incorrect (i.e., incorrectly distributed over the year).

Here we want to calculate, what the risk is, caused by this profile mismatch.

Method:
1. Get two offtake profiles to compare. For each:
   . Pick temperature profile.
   . Pick TLP.
2. Calculate their prices for each year: 
   . Combine temperatures and TL to calculate offtake volume [MW].
   . Combine with spot prices to calculate revenues [Eur].
   . Aggregate to yearly prices [Eur/MWh].
3. Calculate difference between the yearly prices for the two offtake profiles
   . One is the 'reference'. 
   . The difference is what must be added to the price of the reference to
     obtain the price of the other profile ('under investigation').
4. Calculate the premium.
   . From the distribution of the yearly differences, infer (a) the average 
     value (= 'cost component') [Eur/MWh] and (b) a certain quantile (to 
     calculate 'risk component') [Eur/MWh].
"""

#%%

import lichtblyck as lb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% Get input values.

quantile = 0.9

frankfurt = "tageswerte_KL_01420_19490101_20201231_hist.zip"

prices = lb.prices.gas_spot()
prices = lb.fill_gaps(prices, 5)
tmpr = lb.tmpr.hist.tmpr()
tmpr["t_ff"] = lb.tmpr.hist.climate_data(frankfurt)["TMK"].rename("t")  # add frankfurt

start = lb.floor(max([prices.index.min(), tmpr.index.min()]), 1, "AS")
end = lb.floor(min([prices.index.max(), tmpr.index.max()]), 0, "AS")

prices = prices[(prices.index >= start) & (prices.index < end)]
tmpr = tmpr[(tmpr.index >= start) & (tmpr.index < end)]


def charvals(distr, q=0.8):
    return {"mean": distr.mean(), "std": distr.std(), "qtl": distr.quantile(q)}


# %% Calculate portfolio prices.

weights_tlps = lb.tlp.gas.weights()["B2B"]
weights_tlps["base"] = 0
weights_tlps /= weights_tlps.sum()
relevant_tlps = weights_tlps[weights_tlps > 0.03].sort_values(ascending=False).index
relevant_tlps = [*relevant_tlps, "tlp_avg"]

weights_zones = lb.tmpr.weights()["gas"]
weights_zones["t_ff"] = 0
weights_zones /= weights_zones.sum()
relevant_zones = weights_zones[weights_zones > 0.1].sort_values(ascending=False).index
relevant_zones = [*relevant_zones, "t_avg"]

# Calculate yearly portfolio prices for each (climate zone, tlp)-combination...
offtake_records = {}
pfprice_records = {}
for code in weights_tlps.index:
    if code == "base":
        tlp = lambda t: pd.Series([1] * len(t), t.index)  # flat offtake
    else:
        tlp = lb.tlp.gas.fromsource(code, kw=1)
    for zone in weights_zones.index:
        temperature = tmpr[zone]
        offtake = tlp(temperature)  # MW
        offtake = offtake * offtake.duration  # MWh
        offtake = offtake.resample("AS").transform(lambda x: x / x.sum())  # scaled to 1
        revenue = offtake * prices  # Eur
        pfprice = lb.changefreq_sum(revenue, "AS") / 1  # as yearly price [Eur/MWh]
        offtake_records[(zone, code)] = offtake
        pfprice_records[(zone, code)] = pfprice

offtake = pd.DataFrame(offtake_records)
pfprice = pd.DataFrame(pfprice_records)

# ... and add average TLP (for each climate zone)...
avgtlp = pfprice.groupby(axis=1, level=0).apply(
    lambda df: lb.tools.wavg(df.droplevel(0, axis=1), weights_tlps, axis=1)
)
avgtlp.columns = pd.MultiIndex.from_product([avgtlp.columns, ["tlp_avg"]])
pfprice = pd.concat([pfprice, pfprice_avgtlp], axis=1)
# ... and add average climate zone (for each TLP).
avgzone = pfprice.groupby(axis=1, level=1).apply(
    lambda df: lb.tools.wavg(df.droplevel(1, axis=1), weights_zones, axis=1)
)
avgzone.columns = pd.MultiIndex.from_product([["t_avg"], avgzone.columns])
pfprice = pd.concat([pfprice, avgzone], axis=1)


relevant_years = pfprice.index.drop(pfprice.index[1])  # 2009 is extraordinary year


#%% Analysis A: PNL-uncertainty if climate zone or TLP are not same as reference.

# 1: Check influence of reference climate zone, for various climate zones.
# Compare each climatezone-pair. Assume TLP is same in each (i.e., average).

resultsA1 = {}
for zone1 in relevant_zones:
    resultsA1[zone1] = {}
    for zone2 in relevant_zones:
        diff = pfprice[(zone1, "tlp_avg")] - pfprice[(zone2, "tlp_avg")]
        resultsA1[zone1][zone2] = charvals(diff[relevant_years], quantile)
resultsA1 = pd.DataFrame(resultsA1).stack().apply(pd.Series)
resultsA1.index.names = ["customer zone", "reference zone"]
# show only relevant tlps:
A1_std = resultsA1["std"].unstack()
A1_std[["t_ff"]].style.background_gradient(cmap="YlOrRd", axis=None)
A1_mean = resultsA1["mean"].unstack()
A1_mean[["t_ff"]].style.background_gradient(cmap="BrBG", axis=None)


# 2: Check influence of reference TLP, for various TLPs.
# Compare each TLP-pair. Assume climate zone is same in each (i.e., average).

resultsA2 = {}
for code1 in relevant_tlps:
    resultsA2[code1] = {}
    for code2 in relevant_tlps:
        diff = pfprice[("t_avg", code1)] - pfprice[("t_avg", code2)]
        resultsA2[code1][code2] = charvals(diff[relevant_years], quantile)
resultsA2 = pd.DataFrame(resultsA2).stack().apply(pd.Series)
resultsA2.index.names = ["customer TLP", "reference TLP"]
# show only relevant tlps:
A2_std = resultsA2["std"].unstack()
A2_std[["BD4", "HA4", "HD3", "HD4"]].style.background_gradient(cmap="YlOrRd", axis=None)
A2_mean = resultsA2["mean"].unstack()
A2_mean[["BD4", "HA4", "HD3", "HD4"]].style.background_gradient(cmap="BrBG", axis=None)


# 3: Check influence of reference (TLP, climatezone) combination, for various (TLP, climatezone) combinations.
# Compare each (TLP, climatezone) combination with reference combination.

resultsA3 = {}
refprice = pfprice[("t_ff", "HD4")]
for zone in relevant_zones:
    resultsA3[zone] = {}
    for code in relevant_tlps:
        diff = pfprice[(zone, code)] - refprice
        resultsA3[zone][code] = charvals(diff[relevant_years], quantile)
resultsA3 = pd.DataFrame(resultsA3).stack().apply(pd.Series)
resultsA3.index.names = ["customer TLP", "customer zone"]
# show only relevant tlps:
A3_std = resultsA3["std"].unstack()
A3_std.style.background_gradient(cmap="YlOrRd", axis=None)
A3_mean = resultsA3["mean"].unstack()
A3_mean.style.background_gradient(cmap="BrBG", axis=None)


# %% Analysis B: PNL-distribution for average (tlp, climatezone) vs reference.

ref_zone = "t_ff"
ref_code = "HD4"

# 1: Show only distribution for 1 reference TLP (in Frankfurt).

avgprice = pfprice[("t_avg", "tlp_avg")]
refprice = pfprice[(ref_zone, ref_code)]
diff = avgprice - refprice
diff = diff[relevant_years]
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    "Influence of picking single reference climate zone and single reference tlp,\n"
    + "when customers are in an average climate zone and have an average tlp.\n"
    + f"(qlt = {quantile:.0%}-quantile)"
)
r = charvals(diff, quantile)
label = f"t_avg,tlp_avg vs {ref_zone},{ref_code}\nmean:{r['mean']:.2f}\nstd:{r['std']:.2f}\nqtl:{r['qtl']:.2f}"
ax.hist(diff, bins=200, cumulative=True, density=True, label=label, histtype="step")
ax.xaxis.label.set_text("Eur/MWh")
ax.yaxis.set_major_formatter("{:.0%}".format)
ax.legend()
fig.tight_layout()


# 2: Show only distribution for several reference TLPs (in Frankfurt), to see which is best.

avgprice = pfprice[("t_avg", "tlp_avg")]
kwargs = {'sharex' : True, 'sharey': True, 'figsize':(10, 20)}
fig, axes = plt.subplots(int(np.ceil(len(relevant_tlps)/2)), 2, **kwargs)
axes = axes.flatten()
fig.suptitle(
    "Influence of picking single reference climate zone and single reference tlp,\n"
    + "when customers are in an average climate zone and have an average tlp.\n"
    + f"values: (mean, standard dev, {quantile:.0%}-quantile)"
)
for code, ax in zip(relevant_tlps, axes):
    refprice = pfprice[(ref_zone, code)]
    diff = avgprice - refprice
    diff = diff[relevant_years]
    r = charvals(diff, quantile)
    label = f"t_avg,tlp_avg vs {ref_zone},{code}\n({r['mean']:.2f}, {r['std']:.2f}, {r['qtl']:.2f})"
    ax.hist(diff, bins=200, cumulative=True, density=True, label=label, histtype="step")
    ax.xaxis.label.set_text("Eur/MWh")
    ax.yaxis.set_major_formatter("{:.0%}".format)
    ax.legend()
