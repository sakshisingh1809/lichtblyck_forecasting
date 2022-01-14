# %%

import lichtblyck as lb

p = lb.prices.power_spot()

p2 = p.loc[p.index > "2020-10"]
p3 = lb.changefreq_avg(p2, "D").astype(float)
ax = p3.plot(figsize=(12, 8))
ax.set_xlabel("Date")
ax.set_ylabel("Avg. daily spot price [Eur/MWh]")


# %%

p4 = lb.changefreq_avg(p2, "MS").astype(float)
