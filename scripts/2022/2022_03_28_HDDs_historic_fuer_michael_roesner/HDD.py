#%%
import lichtblyck as lb
import pandas as pd

t = lb.tmpr.hist.tmpr()
t = lb.tmpr.hist.fill_gaps(t)

weights = {
    "t_4": 0.282,
    "t_5": 0.248,
    "t_3": 0.173,
    "t_12": 0.133,
    "t_13": 0.055,
    "t_7": 0.055,
    "t_2": 0.054,
}
t = t.loc["1980":, weights.keys()]
t["t_wavg"] = lb.wavg(t, weights, axis=1)

limit_HDD = 18

hdd = (limit_HDD - t).clip(0, 1000)

hdd = hdd.rename(columns={"t_wavg": "first_wavg_then_hdd"})
hdd["first_hdd_then_wavg"] = lb.wavg(hdd.loc[:, weights.keys()], weights, axis=1).clip(
    0, 1000
)

writer = pd.ExcelWriter("historic_climate_data2.xlsx")
t.tz_localize(None).to_excel(writer, "temperatures_per_day")
t.resample("MS").mean().tz_localize(None).to_excel(writer, "temperatures_per_month")
hdd.tz_localize(None).to_excel(writer, "hdd_per_day")
hdd.resample("MS").sum().tz_localize(None).to_excel(writer, "hdd_per_month")
writer.save()
writer.close()

#%%
