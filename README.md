# lichtblyck

Repository with functions to do time-series analysis.

Developer: Ruud Wijtvliet (ruud.wijtvliet@lichtblick.de)

---

---

Rules / Conventions:

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

* Example: time series or dataframe that contains historic temperatures, or the price-forward-curve, or historic prices. Like this historic spot price time series:

  ```
  ts_left
  2010-01-01 00:00:00+01:00    26.25
  2010-01-01 01:00:00+01:00    18.46
  2010-01-01 02:00:00+01:00    20.08
                               ...
  2019-12-31 21:00:00+01:00    39.74
  2019-12-31 22:00:00+01:00    38.88
  2019-12-31 23:00:00+01:00    37.39
  Freq: H, Name: p_spot, Length: 87648, dtype:float64
  ```

## Time of day

* These are values for which the **day is irrelevant**. They describe a time (e.g. 15:32) or time period (e.g. 15:00 till 16:00) within a (any) day.

* The index values are encoded with `datetime.time` objects. They do not (i.e., cannot) include timezone information.

* The following names can be used for the index:

  * `time_local`: denoting a *moment* during the day.
  * `time_left_local`: denoting a *period* during the day.

* A time is always local, i.e., timezone- and daylight-savings-time-unaware.

* Example: time series or dataframe describing how consumption of energy changes with (among other) the time of day. Like this temperature dependent load profile:

  ```
                        -17       -16  ...
  time_left_local                      ...
  00:00:00         3.698160  3.636973  ...
  00:15:00         3.927253  3.862267  ...
  00:30:00         3.927253  3.862267  ...
                    ...       ...      ...
  23:15:00         2.983240  2.866293  ...
  23:30:00         3.246533  3.136160  ...
  23:45:00         3.475627  3.373213  ...
  ```

## Missing date or time component

* If point in time contains an **undefined 'large-scale' time component**, while having a defined 'lower-scale' time component.

* For example, if the average monthly temperature over several years. In that case, the time series only has a 'month' component, while missing the 'year' component, so it cannot be specified as a datetime object. 

* Therefore, the index used is a (multi-)index, and its values are **integers**.

* The following names can be used for the index:

  * `YY` for the year
  * `QQ` for the quarter-of-year (1..4)
  * `MM` for the month-of-year (1..12)
  * `DD` for the day-of-month (1..31)
  * `H` for the hour-of-day (0..23)
  * `QH` for the quarterhour-of-day (0..95)
  * `M` for the minute-of-hour (0..59)
  * `S` for the second-of-minute (0..59)

* Example: time series with average monthly temperature over several years: index only has `MM` level. Or this time series with standard temperature year: index has `MM` and `DD` levels:

  ```
  MM  DD
  1   1     2.791667
      2     0.975000
      3    -0.766667
               ...
  12  29    6.875000
      30    8.820833
      31    7.120833
  Name: t, Length: 366, dtype: float64
  ```

* As noted above; a time or datetime value is preferably used instead of these integer indices, but sometimes a time or datetime needs to be converted to a multiindex of integers in order to do better merging.

# Column names

Dataframe column names, as well as Series names, are standardized in order to more quickly identify what its values mean, but also to be able to do a correct resampling of the timeseries (see resampling example below).

## *"Time-summable"* quantities

These are quantities that only make sense when the *duration* of the time period they apply to is also specified.

* Quantity, Volume (Menge)  
  - Name is `q` or starts with `q_`.
  - Unit is always **MWh**.
* Revenue, Value (Umsatz, Betrag)
  - Name is `r` or starts with `r_`.
  - Unit is always **Eur**.

These quantities can only be thought of as a *discrete list* instead of as a continuous function *f(t)*; each value applies to an entire time period, but not to any one moment within that time period. When downsampling, the values must be summed, see the example below.

To give another example of a time-summable quantity: consider the distances traveled by a car during various time intervals. This quantity has several characteristics. Firstly, the values do not apply to *moments* in time - to travel a distance, we need a *period* of time. Secondly, in order to judge if the distances are small or large, we must consider how much time they were driven in. Thirdly, if we combine several time periods, the distances must be summed to get the correct distance in the total period.

## *"Time-averagable"* quantities

There are quantities, that can be in principle be thought of as *continuous* functions. Changing the time step of the series does not change the magnitude of the values, see the resampling examples below.

* Power (Vermoegen):
  - Name is `w` or starts with `w_`.
  - Unit is always **MW**.
  - These values can be integrated over time. In our case this means: by multiplying the values (in MW) with the duration the corresponding index value (in h), we get the quantity (in MWh) in each time step. By summing these in a certain time interval, we get the volume in that interval. (See "Quantity, Volume", above.)
* Temperature:
  - Name is `t` or starts with `t_`.
  - Unit is always **degC**.

To give another example of a time-averagable quantity: consider the velocity of a car. This quantity has several characteristics that are different from the distance described above. Firstly, the velocity can be measured at every moment in time. We cannot directly measure the velocity during a period of time, we can only calculate it as the *average* value. Secondly, velocity values can be compared directly, even if they apply to time periods with unequal durations. Thirdly, if we combine several time periods, the velocities must be (weighted-)averaged to get the correct velocity during the total period.

## Derived quantities

* Specific price (spezifischer Preis)  
  - Name is `p` or starts with `p_`.
  - Unit is always **Eur/MWh**.
  - Price is always the revenue (`r`) divided by the quantity (`q`), and must be calculated again after resampling. Alternatively, it can be averaged by using the quantity (`q`) as weights.

# Checklist for timeseries and dataframes containing timeseries

* Index:
  * Index has name.
  * If index values are timestamps denoting a moment in time: 
    * Name is `ts` or `ts_local`.
  * If index values are timestamp denoting a time period:
    - Name is `ts_left` or `ts_left_local`;
    - `.index.freq` is set.
  * If index values are times denoting a moment during the (any) day:
    - Name is `time_local`.
  * If index values are times denoting a time period during the (any) day:
    - Name is `time_left_local`;
    - `.index.freq` is set.
  * If index values are a part of a timestamp or time:
    - Name is `YY`, `QQ`, `MM`, `DD`, `H`, `QH`, `M`, or `S`.
* Series name:
  * If values are in MWh, name should be `q` or start with `q_`
  * If values are in MW, name should be `w` or start with `w_`
  * If values are in Eur, name should be `r` or start with `r_`
  * If values are in Eur/MWh, name should be `p` or start with `p_`
  * If values are in degC, name should be `t` or start with `t_`
  * If values are not in one of these units, the name should not be `q`, `w`, `r`, `p`, or `t` and should not start with `q_`, `w_`, `r_`, `p_`, or `t_`.
* Column names for dataframe:
  * Same as series name.


# Resampling

Often, timeseries need to be resampled, i.e., their time step needs to be changed. E.g.: an hourly timeseries needs to be upsampled to quarterhourly timeseries. Or downsampled to a daily timeseries. What happens with the values in the series depends on the type of quantity (time-summable, time-averagable, or derived) they represent.

It is assumed that the conventions for column names are followed. That means:

| if column name is... | or starts with... | it is assumed that...         |
| -------------------- | ----------------- | ----------------------------- |
| `q`                  | `q_`              | this is a quantity in MWh     |
| `w`                  | `w_`              | this is a power in MW         |
| `p`                  | `p_`              | this is a price in Eur/MWh    |
| `r`                  | `r_`              | this is a revenue in Eur      |
| `t`                  | `t_`              | this is a temperature in degC |


## Upsampling example

If, in the year 2020, a customer consumes a volume of 1 GWh (`q = 1000`), i.e., has an average power of 0.11 MW (`w = 0.11`), and pays a price of 30 Eur/MWh (`p = 30`), then the revenue is 30 kEur (`r = 30000`). Let's say the average temperature in the year is 7.98 degC (`t = 7.98`):

```
                                q         w        r     p     t
ts_left                                                  
2020-01-01 00:00:00+01:00  1000.0  0.113843  30000.0  30.0  7.98
```

If we resample these values to a higher-frequency timeseries (e.g. quarteryearly), then the values of the summable quantities (`q` and `r`) become smaller, as their values need to be spread over the resulting rows. If nothing more is known about how the volume is consumed, we assume that the consumption rate is constant throughout the period. This means we have to **distribute** the values over the new rows, **in proportion to their duration**. (Because the 3rd and 4th quarter have more days than the 1st and 2nd quarter, they get a larger fraction of the original value.)

The values of the averagable quantities (`w` and `t`) are **unchanged**, i.e., they are simply copies of the original value. Also the value of the derived quantity `p` turns out to be unchanged. The resulting values are therefore:

```
                                q         w        r     p     t
ts_left                                                                
2020-01-01 00:00:00+01:00  248.52  0.113843  7455.60  30.0  7.98
2020-04-01 00:00:00+02:00  248.63  0.113843  7459.02  30.0  7.98
2020-07-01 00:00:00+02:00  251.37  0.113843  7540.98  30.0  7.98
2020-10-01 00:00:00+02:00  251.48  0.113843  7544.40  30.0  7.98
```

This is the best guess we can make without using any additional information about how the values are distributed throughout the year. Note that each row is consistent, i.e., `q` equals `w` times the duration in hours, and `r` equals `p` times `q`. 

## Downsampling example

Something similar happens when going in the reverse direction, but a bit more intricate. Let's start with these quarteryearly values:

```
                               q         w        r      p     t
ts_left                                                        
2020-01-01 00:00:00+01:00  300.0  0.137426  11330.1  37.77   1.3
2020-04-01 00:00:00+02:00  180.0  0.082418   4554.0  25.30  12.3
2020-07-01 00:00:00+02:00  200.0  0.090580   4260.0  21.30  15.1
2020-10-01 00:00:00+02:00  320.0  0.144862   9856.0  30.80   3.2
```

If we resample to a lower-frequency timeseries (e.g. yearly), we need to **sum** the values of the summable quantities `q` and `r` (the duration does not need to be considered). 

For the time-averagable quantities (`w` and `t`), the **average** of the individual values must be calculated, **weighted with the duration** of each row. (Alternatively, for the power `w`: this is always `q/duration` and can always be calculated from these values after *they* are downsampled.)

For the derived quantity `p`, this is also an average of the individual values, but weighted with the volume `q` of each row. (Alternatively: the price is always `r/q` and can always be calculated from these values after *they* are downsampled.)

The resulting downsampled values are:

```
                                q         w        r     p     t
ts_left                                                              
2020-01-01 00:00:00+01:00  1000.0  0.113843  30000.0  30.0  7.98
```

(Note that the 'simple row-average' of the power, temperature, and price result in incorrect values.)

---

---

Implementation details

# Extended `pandas` functionality

* `DataFrame` and `Series` are extended with a `.duration` property, which returns the duration of each current timestamp in hours.

  This removes the necessity of adding a dedicated column to the dataframe just to store this type of data (or to repeatedly calculate it manually).

---

(to do, but in separate object)

* `DataFrame` and `Series` are extended with a `.q` property, which returns a Series with the quantity [MWh] of each timestamp. It calculates these by, for a DataFrame, multiplying its column `'w'` with its`.index.duration`. And for a Series, multiplying it with its `.index.duration`, if its name is `'w'` or starts with `w_`. (For both: unless a column `'q'` exists; in that case, that column is returned.)

  This removes the necessity of creating and storing both power [MW] and quantity [MWh] columns, which are redundant.
  
* `DataFrame` is extended with a `.r` property, which returns a Series with the quantity [Eur] of each timestamp. It calculates these by multiplying its columns `'q'` and `'p'`. (Unless a column `'r'` exists; in that case, that column is returned.)

  This removes the necessity of storing this property if it can be both power [MW] and quantity [MWh] columns, which are redundant.

---

