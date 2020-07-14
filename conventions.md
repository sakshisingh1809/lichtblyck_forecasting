Rules / Conventions:

# Time-index

## Describing time-component of index

* If values describe a **particular/specific** point in time (2020-04-21 15:32) or a period in time (2020-04-21 15:00 till 16:00), in both cases the index is a `timestamp` index, i.e., the elements are datetime values and include the timezone information.
  - E.g.: time series with historic temperatures or historic prices or the price-forward-curve.
* If the **day is irrelevant**, and values describe a time (15:32) or time period (15:00 till 16:00) within a (any) day, the index is a `time` index, i.e., the elements are time values and do not (cannot) include timezone information.
  - E.g.: Standardized temperature load profile: index has time to indicate the load variation throughout the day.
* If point in time contains an **undefined 'large-scale' time component**, while having a defined 'lower-scale' time component - for example, if the average monthly temperature over several years is calculated.
In that case, the time series only has a 'month' component, while missing the 'year' component, so it cannot be specified as a datetime object. Therefore, the index used is a (multi-)index, and the names of the individual levels are picked from [`YY`, `QQ`, `MM`, `DD`, `H`, `M`, `S`].  
  - E.g.: time series with average monthly temperature over several years: index only has `MM` level.  
  - E.g.: time series with standard temperature year: index has `MM` and `DD` levels.

## Interpreting time-component of index
All time and date timestamps which are used to denote a **period** rather than a **moment** in time (which is almost always the case in our situation), are considered to be left-bound. If this is not the case, i.e. during import, this is corrected before the data is used.

## Index names

* For indices that are **datetime timestamps**, the name starts with `ts`. To be more specific: if it denotes a *period*, it's called `ts_left`; if it denotes a *moment* in time, it's just `ts`. Also, if no timezone information is included, `_local` is appended; otherwise, nothing is appended. So: 
  - `ts`: timezone-aware timestamp denoting a moment in time.
  - `ts_left`: timezone-aware timestamp denoting the start of a period in time.
  - `ts_local`: timezone-agnostic timestamp denoting a moment in time.
  - `ts_left_local`: timezone-agnostic timestamp denoting the start of a period in time.

* For indices that are **times**, the name starts with `time`. To be more specific: if it denotes a time *period*, it's called `time_left_local`; if it denotes a *moment* in time, it's just `time_local`. A time is always local, i.e., timezone- and daylight-savings-time-unaware.

* For indices that indicate only a part/component of a timestamp or time, an abbreviation is used:
  - `YY` for the year
  - `QQ` for the quarter-of-year (1..4)
  - `MM` for the month-of-year (1..12)
  - `DD` for the day-of-month (1..31)
  - `H` for the hour-of-day (0..23)
  - `QH` for the quarterhour-of-day (0..95)
  - `M` for the minute-of-hour (0..59)
  - `S` for the second-of-minute (0..59)

  The values for these indices can only be integers.
  
  These usually denote time periods (rather than moments in time), and the length can be determined from the name of the levels. E.g., if the index name is `MM`, the values denote the monthly aggregates (sum or mean) over multiple years; if the index name is `QH`, the values denote quarterhourly aggregates over multiple days. 

  As noted above; a time or datetime value is preferably used instead of these integer indices, but sometimes a time or datetime needs to be converted to a multiindex of integers in order to do better merging.



# Column names

Column names are standardized in order to more quickly identify what its values mean, but also to be able to do a correct resampling of the timeseries (see resampling example below).

## Discrete, time-integrated quantities
These are quantities that only make sense when the *duration* of the time period they apply to is also specified.

* Quantity, Volume (Menge)  
  - Name starts with `q_`.
  - `q_xxx` --> unit is always **MWh**.
* Revenue, Value (Umsatz, Betrag)
  - Name starts with `r_`.
  - `r_xxx` --> unit is always **Eur**.

These quantities that can only be thought of as a *discrete list* instead of as a continuous function *f(t)*. Also, when resampling, the values must be changed, see the example below.

## Continuous, time-averaged quantities
There are quantities, that can be in principle be thought of as *continuous* functions. Changing the time step of the series does not change the magnitude of the values, see the resampling examples below.

* Power (Vermoegen):
  - Name starts with `w_`.
  - `w_xxx` --> unit is always **MW**.
  - These values can be integrated over time. In our case this means: by multiplying the values (in MW) with the time step of the index (in h), we get the quantity (in MWh) in each time step. By summing these in a certain time interval, we get the volume in that interval. (See "Quantity, Volume", above.)
* Temperature:
  - Name starts with `tmpr_`.
  - `tmpr_` --> unit is always **degC**.

## Derived quantities

* Speficic price (spezifischer Preis)  
  - Name starts with `p_`.
  - `p_xxx` --> unit is always **Eur/MWh**.
  - Price is always the revenue (`r`) divided by the quantity (`q`), and must 

# Resampling

Often, timeseries need to be resampled, i.e., their time step needs to be changed. E.g.: an hourly timeseries needs to be upsampled to quarterhourly timeseries. Or downsampled to a daily timeseries. What happens with the values in the series depends on the type of quantity (discrete or continuous) they represent.

## Upsampling example

If we have an hour where a customer consumes a volume of 6 MWh (`q = 6`), i.e., has an average power of 6 MW (`w = 6`), and pays a price of 30 Eur/MWh (`p = 30`), then the revenue is 180 Eur (`r = 180`). Let's say the average temperature in that hour is 8 degC (`tmpr = 8`):
```
                     q  w   p    r  tmpr
2020-01-01 00:00:00  6  6  30  180     8
```
If we resample these values to a higher-frequency timeseries (e.g. quarter-hourly), then the values of the discrete quantities (`q` and `r`) become smaller, as their values need to be spread over the resulting rows. The values of the continuous quantities (`w`, `p` and `tmpr`) are more or less unchanged, as they are averages.  
If nothing more is known about how the volume is consumed throughout the day, the best estimate for the discrete quantities is to simply divide the values by the number of new rows (4 in this case), and the best estimate for the continous quantities is to simply repeat the value:
```
                       q  w   p     r  tmpr
2020-01-01 00:00:00  1.5  6  30  45.0     8
2020-01-01 00:15:00  1.5  6  30  45.0     8
2020-01-01 00:30:00  1.5  6  30  45.0     8
2020-01-01 00:45:00  1.5  6  30  45.0     8
```
Note that in this case, due to lack of information about the sub-hour granularity, all rows are identical.

## Downsampling example

Something similar happens when going in the reverse direction, but a bit more intrecate. Let's start with these quarterhourly values:
```
                       q  w     p   r  tmpr
2020-01-01 00:00:00  2.0  8  40.0  80     8
2020-01-01 00:15:00  1.5  6  20.0  30     6
2020-01-01 00:30:00  1.0  4  25.0  25     7
2020-01-01 00:45:00  1.5  6  30.0  45    11
```
If we resample to a lower-frequency timeseries (hourly), we need to **sum** the values of the discrete quantities. We can **average** the values of the continuous quantities, but, for the price, the average must be weighed with the volume (`q`) or power (`w`) in order to get the correct value. Alternatively, the price is always `r/q` and can be calculated from these values after *they* are upsampled:
```
                     q  w   p    r  tmpr
2020-01-01 00:00:00  6  6  30  180     8
```
(Note that the 'simple row-average' of the price is the incorrect 28.75 Eur/MWh, instead of the (correct) weighted average of 30 Eur/MWh.)
