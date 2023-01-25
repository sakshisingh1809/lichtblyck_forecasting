# lichtblyck-forecasting

Repository with functions to do temperature time-series analysis and forecasting.

---

# Time-index

The name, and the type of data, used for the index, is standardized in order to more quickly identify what its values mean. Exactly how depends on the situation, which is one of the following three.

## Specific, single moment or period in time

* These are values that describe a **specific moment in time** (e.g. 2020-04-21 15:32) or a **specific period in time** (e.g. 2020-04-21 15:00 till 16:00).
* The index values are encoded with **datetime timestamps**. Whenever possible, the timezone should be included.

* The following names can be used for the index:

  - `ts`: *timezone-aware* timestamp denoting a *moment* in time.
  - `ts_left`: *timezone-aware* timestamp denoting (the start of) a *period* in time. (`ts_right` if it's the end of the period; should be avoided.)
  - `ts_local`: *timezone-agnostic* timestamp denoting a *moment* in time.
  - `ts_left_local`: *timezone-agnostic* timestamp denoting (the start of) a *period* in time. (`ts_right_local` if it's the end of the period; should be avoided.)
