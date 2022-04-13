import pandas as pd
import pytz


def floor2(self, freq):
    if self.tz is None:
        return self.__class__(self.floor(freq), freq=self.freq)

    # All timestamps are converted to the anchor's UTC offset.
    anchor = self if isinstance(self, pd.Timestamp) else self[0]

    # Turn into fixed offset to eliminate ambiguity...
    fo = pytz.FixedOffset(anchor.utcoffset().total_seconds() / 60)
    # (fo = 'UTC' gives incorrect result if UTC-offset is not integer number of hours)
    # ...then floor and convert back to original timezone...
    newinstance = self.tz_convert(fo).floor(freq).tz_convert(self.tz)
    # ...and keep original frequency.
    return self.__class__(newinstance, freq=self.freq)


pd.DatetimeIndex.floor2 = floor2
pd.Timestamp.floor2 = floor2

# Works for DatetimeIndex
# -----------------------
i0 = pd.date_range("2020-10-25 1:11", freq="H", periods=4)
i0.floor2("15T")
# DatetimeIndex(['2020-10-25 01:00:00', '2020-10-25 02:00:00',
#               '2020-10-25 03:00:00', '2020-10-25 04:00:00'],
#              dtype='datetime64[ns]', freq='H')

i1 = pd.date_range("2020-10-25 1:11", freq="H", periods=4, tz="Europe/Berlin")
i1.floor2("15T")
# DatetimeIndex(['2020-10-25 01:00:00+02:00', '2020-10-25 02:00:00+02:00',
#                '2020-10-25 02:00:00+01:00', '2020-10-25 03:00:00+01:00'],
#               dtype='datetime64[ns, Europe/Berlin]', freq='H')

i2 = pd.date_range("2020-10-25 1:11", freq="H", periods=4, tz="Asia/Kolkata")
i2.floor2("15T")
# DatetimeIndex(['2020-10-25 01:00:00+05:30', '2020-10-25 02:00:00+05:30',
#                '2020-10-25 03:00:00+05:30', '2020-10-25 04:00:00+05:30'],
#               dtype='datetime64[ns, Asia/Kolkata]', freq='H')

# Works for Timestamp
# -------------------
i0[0].floor2("15T")
# Timestamp('2020-10-25 01:00:00', freq='H')
i1[0].floor2("15T")
# Timestamp('2020-10-25 01:00:00+0200', tz='Europe/Berlin', freq='H')
i2[0].floor2("15T")
# Timestamp('2020-10-25 01:00:00+0530', tz='Asia/Kolkata', freq='H')
