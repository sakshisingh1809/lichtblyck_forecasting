#%%

import lichtblyck as lb
from lichtblyck.prices import utils
import pandas as pd
import numpy as np

def get_wpresults_short(start, freq, length, tz, bpo, aggfreq):
    """For freq == 'H' or shorter.
    returns {(None or startts): {('val' or 'vol'): (peak, offpeak)}}
    """
    i = pd.date_range(start, freq=freq, periods=length, tz=tz)
    w_values = 100 + 100 * np.random.rand(len(i))
    p_values = 50 + 20 * np.random.rand(len(i))

    if aggfreq is None:
        resultkey = lambda ts: None
    else:
        # key is start of delivery period
        resultkey = lambda ts: utils.ts_deliv(ts, aggfreq.lower()[0], 0)[0]

    result = {}
    duration = 0.25 if freq == "15T" else 1
    for ts, w, p in zip(i, w_values, p_values):

        key = resultkey(ts)
        if key not in result:
            result[key] = {
                "w.d": np.array([0.0, 0.0]),
                "d": np.array([0.0, 0.0]),
                "w.pd": np.array([0.0, 0.0]),
                "pd": np.array([0.0, 0.0]),
            }

        if bpo and not utils.is_peak_hour(ts):
            result[key]["w.d"] += [0, w * duration]
            result[key]["d"] += [0, duration]
            result[key]["w.pd"] += [0, p * w * duration]
            result[key]["pd"] += [0, p * duration]
        else:
            result[key]["w.d"] += [w * duration, 0]
            result[key]["d"] += [duration, 0]
            result[key]["w.pd"] += [p * w * duration, 0]
            result[key]["pd"] += [p * duration, 0]

    return (
        pd.Series(w_values, i),
        pd.Series(p_values, i),
        {
            key: {
                "val": values["w.pd"] / values["pd"],
                "vol": values["w.d"] / values["d"],
            }
            for key, values in result.items()
        },
    )


w, p, ref_results = get_wpresults_short('2020', 'H', 4, 'Europe/Berlin', True, 'MS')

# u1 = lb.prices.w_hedge(w, p)
# u2 = lb.prices.w_hedge(w, p, how='val')
u3 = lb.prices.w_hedge(w, p, bpo=True)
# u4 = lb.prices.w_hedge(w, p, how='val', bpo=True)

#%%