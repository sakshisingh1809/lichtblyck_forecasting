## Goal

I'm considering creating a class for timeZone-aware periods (which I'll call `Zeriod` and accompanying `ZeriodIndex`).

Before I start coding, though, I want to discuss my ideas here. Dealing with timezones is notoriously difficult, so I'd appreciate your help in identifying possible problems and making the concept water-tight.

Also, the ambition would be to have this as an index class next to `DatetimeIndex`, `IntervalIndex` and `PeriodIndex` - I'd appreciate feed-back on how likely/possible/difficult this might be.

## Motivation

`Period` and `PeriodIndex` are not timezone-aware, this is [out of scope](https://github.com/pandas-dev/pandas/issues/45736#issuecomment-1036152148) for these classes. When timezone awareness is required in dealing with time periods, one can creatively use `Timestamp` and `DatetimeIndex` instead, as detailed [here](https://github.com/pandas-dev/pandas/issues/45736). However, this is not ideal; firstly, it is conceptually incorrect (a timestamp is an instant, not an interval), and secondly, changes to the `Timestamp` class make this impossible in the future.

## Example use-case

For a concrete example, consider a European powerplant is producing 1 MW of electrical power - year-in, year-out, "24/7". We'd like to calculate how much energy that is, during various time periods, by multiplying it with the number of hours. We get non-timezone-related variation due to the varying lengths of the calendar periods:

* 720 MWh in June, but 744 MWh in July;
* 2184 MWh in Q2, but 2208 MWh in Q3;
* 8760 MWh in 2022, but 8784 MWh in 2024;

However, we also get some timezone-related variation - due to the daylight savings-time (DST) changeover:

* 24 MWh on every calendar day, except for 23 MWh on 2022-03-27 and 25 MWh on 2022-10-30;
* 744 MWh in every 31-day month, except for 743 MWh in March and 745 MWh in October.

For the former variation, `Period` is perfect, but the latter is out of its scope.

In general: `Period` and `PeriodIndex` are insufficient in use-cases that (a) deal with a timezone or geographic location that changes its UTC-offset, while also (b) depending on the duration of a time period - e.g. when aggregating rates of change, like [meters per second] to [meters], or [Euro per hour] to [Euro], or, as in the example [MW] to [MWh].

Another use-case is when comparing a timestamp with a time period - e.g., to see if the former lies within the latter. If the timestamp is timezone-aware, it can only be compared with a time period that is also timezone-aware. The timestamp `2022-03-11T01:00:00+04:00` falls within the calendar day `2022-03-11` of the Aukland timezone, but not of the New York timezone.

## Concepts

Let's consider the "universal time axis", and let's see how people, on planet earth, in modern times, indicate points and intervals on it.

Points in time are called timestamps, and we can specify them unambiguously by adding their UTC-offset. Depending on who you ask, there are 38 distinct points on the time axis that may be called `2022-03-11T15:00:00`, but everyone agrees which point is meant by `2022-02-11T15:00:00+01:00`.

One way to unambiguously specify an interval on the time axis would be with its start and end timestamp (both including a UTC-offset). When done this way, the duration of the interval is easily calculated, and it's also easy to see if a certain timestamp lies inside the interval or not. However, time intervals are rarely specified this way.

More often, an interval is selected from many commonly understood subdivisions of the time axis. These intervals are things like calender years, weeks, days, and seconds, and they have some complications:

* As noted, not all subdivisions are equally long. January and February are both months, but of unequal duration. When DST is considered, the only periods with a fixed duration are `hour` and shorter.

* Some subdivisions fit neatly inside one another ("box-in-box"), like for each pair in the chain `second -> minute -> hour -> day -> month -> quarter -> year`. The time interval describing any calendar day lies *entirely* within the time interval describing a *single* calendar month. Other subdivisions are not so neat; a week can cut across month, quarter, and year boundaries.

* Just like a timestamp needs a UTC-offset to be unambiguous, a period needs a timezone to be unambiguous. There are dozens of interpretations of which interval on the time axis is meant with "the calendar day `2022-03-11`", but there is only one "calendar day `2022-03-11 Europe/Berlin`".

## Scope

Here is the scope I imagine for the MVP. Might be expanded later.

#### Internals

A `Zeriod` is fully defined with a starting `pandas.Timestamp` and a `pandas` frequency string. The timestamp must include a UTC-offset. It must also include either a timezone (like `Europe/Berlin`) or else a fixed offset (like `pytz.FixedOffset(60)`).

I'll just concatenate this for now as e.g. `2022-03-11T15:00:00+01:00 Europe/Berlin 'H'` or `2022-03-11T0:00:00+01:00 Europe/Berlin 'D'`, even though in some cases this can be shortened without loss of information.

A `ZeriodIndex` is fully defined with a list of `pandas.Timestamp`s, indicating the starting points, as well as a timezone and frequency; all this information can internally be held in a `pandas.DatetimeIndex`.

#### Closed/Open

A `Zeriod` is left-closed and right-open. 

Consequence: 

* There is *no gap* and also *no overlap* between any 2 consecutive `Zeriod`s of a given timezone and frequency.

* When considering any timestamp, there is always exactly one `Zeriod` (of a given timezone and frequency) that contains it.

#### Frequencies

'box-in-box' periods are in scope. This includes the frequencies: `year`, `quarter`, `month`, `day`, `hour`, `minute`, `second`. Shorter frequencies than second might be included; no complications are expected.

With `day` a calendar day, starting and ending at midnight, is meant; an `hour` is meant to start and end at the full hour, etc. A `quarter` is meant to start with `Q1` in January; other options might be considered later. 

Also in-scope are "divisors of frequency pairs that have a fixed ratio". There are always 60 minutes in an hour, and always 60 seconds in a minute. For these, any divisor might be acceptable, so a frequency of `15 * minute` or `20 * second` are in-scope, but `23 * minute` is not. [^1]

[^1]: there are not always 24 hours in a day, so a frequency of `6 * hour` is also out of scope.

Out of scope is frequency `week`.

Consequence: 

* When considering any `Zeriod` of a given timezone and frequency, then, for the same timezone and any longer frequency, there is always *exactly one* `Zeriod` that *entirely* contains it, and *no* `Zeriod` that *partly* contains it.

* This greatly simplifies resampling.

#### Attributes and functionality

* `.start_time` and `.end_time` return the start (incl) and end (excl) timestamps.
* `.freq` returns the frequency object.
* `.length` returns duration of the `Zeriod` as a timedelta.
  This might be changed to a float in hours or seconds, as timedeltas have a fixed-length day of 24 hours. This does not introduce errors, but might be confusing.

#### Timezone conversions

Converting between timezones is not generally possible, and not the focus of the implementation.

(Some `Zeriod`s can be converted into other timezones. For example, the following `Zeriod`s are equal: `2022-03-11T15:00:00+01:00 Europe/Berlin 'H'`, `2022-03-11T16:00:00+02:00 Europe/Athens 'H'`, `2022-03-11T14:00:00+00:00 UTC 'H'`. In the `Tehran` timezone, however it is the time interval `['2022-03-11T17:30:00+03:30 Asia/Tehran', '2022-03-11T18:30:00+03:30 Asia/Tehran')`)

## Sample use case

Using the example above, I imagine something like this

```python
i = pd.zeriod_range('2022', periods=365, freq='D', tz='Europe/Berlin')
i
# ZeriodIndex(['2022-01-01 00:00:00+01:00', '2022-01-02 00:00:00+01:00',
#              '2022-01-03 00:00:00+01:00', '2022-01-04 00:00:00+01:00',
#              ...
#              '2022-03-27 00:00:00+01:00', '2022-03-28 00:00:00+02:00',
#              ...
#              '2022-10-30 00:00:00+02:00', '2022-10-31 00:00:00+01:00',
#              ...
#              '2022-12-28 00:00:00+01:00', '2022-12-29 00:00:00+01:00',
#              '2022-12-30 00:00:00+01:00', '2022-12-31 00:00:00+01:00'],
#             dtype='datetime64[ns, Europe/Berlin]', length=365, freq='D')
i.end_time - i.start_time
# TimedeltaIndex(['1 days', '1 days', ... , '1 days', '0 days 23:00:00', '1 days', ...
#                 ... '1 days', '1 days 01:00:00', '1 days', ..., '1 days', '1 days'],
#                dtype='timedelta64[ns]', length=365, freq=None)
power = pd.Series(1.0, i)
power
# 2022-01-01 00:00:00+01:00    1.0
# 2022-01-02 00:00:00+01:00    1.0
#                              ... 
# 2022-03-27 00:00:00+01:00    1.0
# 2022-03-28 00:00:00+02:00    1.0
#                              ... 
# 2022-10-30 00:00:00+02:00    1.0
# 2022-10-31 00:00:00+01:00    1.0
#                              ... 
# 2022-12-30 00:00:00+01:00    1.0
# 2022-12-31 00:00:00+01:00    1.0
# Freq: D, Length: 365, dtype: float64
energy = power * power.index.length.total_seconds() / 3600
energy
# 2022-01-01 00:00:00+01:00    24.0
# 2022-01-02 00:00:00+01:00    24.0
#                              ... 
# 2022-03-27 00:00:00+01:00    23.0 #!
# 2022-03-28 00:00:00+02:00    24.0
#                              ... 
# 2022-10-30 00:00:00+02:00    25.0 #!
# 2022-10-31 00:00:00+01:00    24.0
#                              ... 
# 2022-12-30 00:00:00+01:00    24.0
# 2022-12-31 00:00:00+01:00    24.0
# Freq: D, Length: 365, dtype: float64
monthly_energy = energy.resample('M').sum()
monthly_energy
# 2022-01-01 00:00:00+01:00    744.0
# 2022-02-01 00:00:00+01:00    672.0
# 2022-03-01 00:00:00+01:00    743.0 #!
# 2022-04-01 00:00:00+02:00    720.0
# 2022-05-01 00:00:00+02:00    744.0
# 2022-06-01 00:00:00+02:00    720.0
# 2022-07-01 00:00:00+02:00    744.0
# 2022-08-01 00:00:00+02:00    744.0
# 2022-09-01 00:00:00+02:00    720.0
# 2022-10-01 00:00:00+02:00    745.0 #!
# 2022-11-01 00:00:00+01:00    720.0
# 2022-12-01 00:00:00+01:00    744.0
# Freq: M, dtype: float64
```