from lichtblyck.core import utils
from lichtblyck.tools.stamps import freq_up_or_down
from lichtblyck.tools.frames import set_ts_index, wavg
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import pytest
import functools

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


@pytest.fixture(params=freqs_small_to_large)
def freq(request):
    return request.param

freq1 = freq2 = freq


@functools.lru_cache
def aggdata():
    # Sample data
    i_15T = pd.date_range("2020", "2022", freq="15T", tz="Europe/Berlin", closed="left")

    def value_func(mean, ampl_a, ampl_m, ampl_d):
        start = pd.Timestamp("2020", tz="Europe/Berlin")
        end = pd.Timestamp("2021", tz="Europe/Berlin")

        def value(ts):
            angle = 2 * np.pi * (ts - start) / (end - start)
            return (
                mean
                + ampl_a * np.cos(angle + np.pi / 12)
                + ampl_m * np.cos(angle * 12)
                + ampl_d * np.cos(angle * 365)
            )

        return np.vectorize(value)  # make sure it accepts arrays

    f = value_func(500, 300, 150, 50)
    values = np.random.normal(f(i_15T), 10)  # added noise
    source = set_ts_index(pd.Series(values, i_15T))

    # Seperate the values in bins for later aggregation.
    def isstart_f(freq):
        if freq == "AS":
            return lambda ts: ts.floor("D") == ts and ts.is_year_start
        if freq == "QS":
            return lambda ts: ts.floor("D") == ts and ts.is_quarter_start
        if freq == "MS":
            return lambda ts: ts.floor("D") == ts and ts.is_month_start
        if freq == "D":
            return lambda ts: ts.floor("D") == ts
        if freq == "H":
            return lambda ts: ts.minute == 0
        raise ValueError("Invalid value for `freq`.")

    agg_data = {
        freq: {"values": [], "index": [], "durations": [], "new": isstart_f(freq)}
        for freq in ["H", "D", "MS", "QS", "AS"]
    }
    for ts, val, dur in zip(source.index, source.values, source.index.duration):
        for freq, dic in agg_data.items():
            if dic["new"](ts):
                dic["index"].append(ts)
                dic["values"].append([])
                dic["durations"].append([])
            dic["values"][-1].append(val)
            dic["durations"][-1].append(dur)
    agg_data["15T"] = {
        "values": [[v] for v in source.values],
        "durations": [[d] for d in source.index.duration],
        "index": source.index,
    }
    return agg_data


@functools.lru_cache
def combis_downsampling():
    # series-pairs, where one can be turned into the other by downsampling
    agg_data = aggdata()
    summed, avged = {}, {}
    for freq, dic in agg_data.items():
        summ = [sum(vals) for vals in dic["values"]]
        avg = [
            wavg(pd.Series(values), durations)
            for values, durations in zip(dic["values"], dic["durations"])
        ]

        for vals, coll in [(summ, summed), (avg, avged)]:
            coll[freq] = set_ts_index(
                pd.Series(vals, dic["index"]).resample(freq).asfreq()
            )

    sumcombis, avgcombis = [], []
    for coll, combis in [(summed, sumcombis), (avged, avgcombis)]:
        for freq1, s1 in coll.items():
            for freq2, s2 in coll.items():
                if freq_up_or_down(freq1, freq2) > 0:
                    continue
                # freq1 to freq2 means downsampling
                combis.append((s1, s2))

    return sumcombis, avgcombis


@functools.lru_cache
def combis_upsampling():
    # series-pairs, where one can be turned into the other by upsampling.
    agg_data = aggdata()
    sumcombis, avgcombis = [], []
    for freq1, dic1 in agg_data.items():
        for freq2, dic2 in agg_data.items():
            if freq_up_or_down(freq1, freq2) < 0:
                continue
            # freq1 to freq2 means upsampling

            # Find the two series, value-by-value.
            sumrecords1, sumrecords2, avgrecords1, avgrecords2 = {}, {}, {}, {}
            i2 = 0
            for ts1, vals1, durs1 in zip(
                dic1["index"], dic1["values"], dic1["durations"]
            ):
                len1 = len(vals1)
                sumval1 = sum(vals1)
                avgval1 = wavg(pd.Series(vals1), durs1)

                # For each datapoint in long frequency, find corresponing datapoints in shorter frequency.
                tss2, durss2 = [], []
                len2 = 0
                # ts1 is single timestamp; vals1 and durs1 are nonnested lists.
                # tss2 is list of timestamps; valss2 and durss2 are nested lists.
                while len2 < len1:
                    tss2.append(dic2["index"][i2])
                    durss2.append(dic2["durations"][i2])
                    len2 += len(dic2["durations"][i2])
                    i2 += 1

                durs2 = np.array([sum(durs) for durs in durss2])
                durfractions = durs2 / durs2.sum()
                sumvals2 = sumval1 * durfractions

                assert sum(durs1) == sum(
                    durs2
                )  # just small check (not part of pytests)

                sumrecords1[ts1] = sumval1
                avgrecords1[ts1] = avgval1
                for ts, sumval in zip(tss2, sumvals2):
                    sumrecords2[ts] = sumval
                    avgrecords2[ts] = avgval1  # same value copied to all children

            # Add the pair to the combinations
            for combis, records1, records2 in [
                (sumcombis, sumrecords1, sumrecords2),
                (avgcombis, avgrecords1, avgrecords2),
            ]:
                s1 = set_ts_index(pd.Series(records1).resample(freq1).asfreq())
                s2 = set_ts_index(pd.Series(records2).resample(freq2).asfreq())
                combis.append((s1, s2))

    return sumcombis, avgcombis


def summable():
    combis = []
    sum_up, _ = combis_upsampling()
    sum_down, _ = combis_downsampling()
    for key, sumcombis in (("up", sum_up), ("down", sum_down)):
        for s1, s2 in sumcombis:
            combis.append((s1, s2, f"{key}-s-{freq1}-{freq2}"))
            combis.append(
                (
                    pd.DataFrame({"a": s1}),
                    pd.DataFrame({"a": s2}),
                    f"{key}-df-{freq1}-{freq2}",
                )
            )
    return combis


def avgable():
    combis = []
    _, avg_up = combis_upsampling()
    _, avg_down = combis_downsampling()
    for key, avgcombis in (("up", avg_up), ("down", avg_down)):
        for s1, s2 in avgcombis:
            combis.append((s1, s2, f"{key}-s-{freq1}-{freq2}"))
            combis.append(
                (
                    pd.DataFrame({"a": s1}),
                    pd.DataFrame({"a": s2}),
                    f"{key}-df-{freq1}-{freq2}",
                )
            )
    return combis


@pytest.mark.parametrize("fr1,fr2,descr", summable())
def test_changefreq_sum(fr1, fr2, descr):
    testfr = utils.changefreq_sum(fr1, fr2.index.freq)
    if isinstance(fr1, Series):
        pd.testing.assert_series_equal(testfr, fr2)
    else:
        pd.testing.assert_frame_equal(testfr, fr2)


@pytest.mark.parametrize("fr1,fr2,descr", avgable())
def test_changefreq_avg(fr1, fr2, descr):
    testfr = utils.changefreq_avg(fr1, fr2.index.freq)
    if isinstance(fr1, Series):
        pd.testing.assert_series_equal(testfr, fr2)
    else:
        pd.testing.assert_frame_equal(testfr, fr2)


# # TODO: test where timeseries is downsampled that starts/ends in e.g. middle of month.

