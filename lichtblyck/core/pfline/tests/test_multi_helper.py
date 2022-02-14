from lichtblyck import testing
from lichtblyck.core.pfline import multi_helper
from lichtblyck.core.develop import dev
import pandas as pd
import pytest


@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("kind1", ["p", "q", "all"])
@pytest.mark.parametrize("kind2", ["p", "q", "all", None])
@pytest.mark.parametrize("kind3", ["p", "q", "all", None])
def test_makedataframe_consistency(freq, kind1, kind2, kind3):
    """Test if conversions are done correctly and inconsistent data raises error."""

    i = dev.get_index(freq, "Europe/Berlin")

    kinds, dic = [], {}
    for k, kind in enumerate([kind1, kind2, kind3]):
        if kind is not None:
            kinds.append(kind)
            dic[f"part_{k}"].append(dev.get_singlepfline(i, kind))

    if len(dic) == 1:
        pass

    elif len(dic) == 2:
        if len(set(kinds)) != 1 and set(kinds) != set(["p", "q"]):
            # Can only combine 2 pflines if they have the same kind or are 'q' and 'p'
            with pytest.raises(ValueError):
                _ = multi_helper.make_childrendict(dic)
            return

    elif len(dic) == 3:
        if len(set(kinds)) != 1:
            # Can only combine 3 pflines if they have the same kind.
            with pytest.raises(ValueError):
                _ = multi_helper.make_childrendict(dic)
            return

    result = multi_helper.make_childrendict(dic)
    assert result == dic


@pytest.mark.parametrize("freq1", ["15T", "D", "MS", "QS"])  # don't do all - many!
@pytest.mark.parametrize("freq2", ["15T", "H", "D", "MS", "QS"])
def test_makedict_unequalfrequencies(freq1, freq2):
    """Test if error is raised when creating a dictionary from pflines with unequal frequencies."""

    kwargs = {"start": "2020", "end": "2021", "closed": "left", "tz": "Europe/Berlin"}
    i1 = pd.date_range(**kwargs, freq=freq1)
    i2 = pd.date_range(**kwargs, freq=freq2)

    spfl1 = dev.get_singlepfline(i1, "all")
    spfl2 = dev.get_singlepfline(i2, "all")

    dic = {"PartA": spfl1, "PartB": spfl2}

    if freq1 != freq2:
        with pytest.raises(ValueError):
            _ = multi_helper.make_childrendict(dic)


@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize("overlap", [True, False])
def test_pfline_unequaltimeperiods(freq, overlap):
    """Test if only intersection is kept for overlapping pflines, and error is raised
    for non-overlapping pflines."""

    i1 = pd.date_range(
        start="2020-01-01",
        end="2020-06-01",
        freq=freq,
        closed="left",
        tz="Europe/Berlin",
    )
    start = "2020-03-01" if overlap else "2020-07-01"
    i2 = pd.date_range(
        start=start,
        end="2020-09-01",
        freq=freq,
        closed="left",
        tz="Europe/Berlin",
    )

    spfl1 = dev.get_singlepfline(i1, "all")
    spfl2 = dev.get_singlepfline(i2, "all")
    dic = {"PartA": spfl1, "PartB": spfl2}

    intersection = spfl1.index.intersection(spfl2.index)

    if not overlap:
        # raise ValueError("The two portfoliolines do not have anything in common.")
        with pytest.raises(ValueError):
            result = multi_helper.make_childrendict(dic)
        return

    result = multi_helper.make_childrendict(dic)
    for name, child in result.items():
        testing.assert_series_equal(child.q, dic[name].loc[intersection].q)
        testing.assert_series_equal(child.r, dic[name].loc[intersection].r)
        testing.assert_index_equal(child.index, intersection)
