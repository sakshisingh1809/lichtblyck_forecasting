import pandas as pd


# In some situations, I will have a series (or dataframe) with several values and a DatetimeIndex
# In other situtions, I will only a single value and a Timestamp.
# In both, I need to know, how much time the value(s) apply to.

# Example: running costs of a machine that is always on.
idx = pd.date_range("2022", freq="MS", periods=12, tz="Europe/Berlin")
costs = pd.DataFrame({"USD_per_h": 1.0}, idx)  # fixed running costs for sake of example

# The total costs can be calculated with this function.
def costs_USD(ts, costs_USD_per_h):
    timedelta = (ts + ts.freq) - ts
    hours = timedelta.total_seconds() / 3600
    return costs_USD_per_h * hours


# (A) As long as I'm working with entire series/dataframes, this works, as the `freq` attribute of the index is available:
costs["USD"] = costs_USD(costs.index, costs.USD_per_h)

# Note: correct, with odd number in March and Oct which have DST-change
# costs
# 	                        USD_per_h    USD
# 2022-01-01 00:00:00+01:00       1.0  744.0
# 2022-02-01 00:00:00+01:00       1.0  672.0
# 2022-03-01 00:00:00+01:00       1.0  743.0
# 2022-04-01 00:00:00+02:00       1.0  720.0
# 2022-05-01 00:00:00+02:00       1.0  744.0
# 2022-06-01 00:00:00+02:00       1.0  720.0
# 2022-07-01 00:00:00+02:00       1.0  744.0
# 2022-08-01 00:00:00+02:00       1.0  744.0
# 2022-09-01 00:00:00+02:00       1.0  720.0
# 2022-10-01 00:00:00+02:00       1.0  745.0
# 2022-11-01 00:00:00+01:00       1.0  720.0
# 2022-12-01 00:00:00+01:00       1.0  744.0

# (B) But, I might deal with a single timestamp and single value; and a 'lone timestamp' will be passed to the function:
costs_in_january = costs_USD(costs.index[0], costs.USD_per_h[0])  # FutureWarning

# (C) Or, I (for whatever reason) might want/have to apply a function row-wise; here too a 'lone timestamp' is passed:
costs["USD"] = costs.apply(lambda row: costs_USD(row.name, row.USD_per_h), axis=1)
# (which will lose its .freq attribute soon)


"""Surely, there are workarounds, but they are all cumbersome and remove flexibility.

My point concerning your suggestion was that, yes, I could add `old_freq` as an attribute to a timestamp. I could then change the `costs_USD` function like this:
"""


def costs_USD2(ts, costs_USD_per_h):
    if hasattr(ts, "old_freq"):  # or catch AttributeError
        freq = ts.old_freq
    else:
        freq = ts.freq
    timedelta = (ts + freq) - ts
    hours = timedelta.total_seconds() / 3600
    return costs_USD_per_h * hours


"""(A) would still work, and for (B) I could do some prep-work like so before calling:"""
ts = costs.index[0]
ts.old_freq = costs.index.freq
costs_in_january = costs_USD2(ts, costs.USD_per_h[0])  # No FutureWarning

"""However, (C) would not work anymore, and in general, I'd need to remember to add the `old_freq` attribute whenever I store an index value to a seperate variable or deal with individual timestamps."""
