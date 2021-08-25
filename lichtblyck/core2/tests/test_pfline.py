# Assert correct working of _make_df:
# . can be called with dictionary, with dataframe, with pfline, with named tuple.
# . check with various combinations of keys: p, q, w, p and q, q and w, etc.
# . check that inconsisten data raises error.
# . check with keys having unequal indexes: unequal freq, timeperiod.
# . check if missing values have expected result.

# Assert correct working of pfline:
# . initialisation with dictionary, with dataframe, with named tuple.
# . initialisation with pfline returns identical pfline.
# . .kind property always correctly set.
# . timeseries can be accessed with .q, .p, .r, .w, ['q'], ['p'], etc. Check for various kinds.
# . check correct working of attributes .df() and .changefreq().
# . check correct working of dunder methods. E.g. assert correct addition:
# . . pflines having same or different kind
# . . pflines having same or different frequency
# . . pflines covering same or different time periods

from lichtblyck.core2.pfline import PfLine, _make_df
from lichtblyck.core2 import dev
from typing import Union
import pandas as pd
import numpy as np
import pytest



@pytest.mark.parametrize("tz", ['Europe/Berlin', None])
@pytest.mark.parametrize("freq", ['MS', 'D'])
def test_makedf1(freq, tz):
    i = dev.get_index(tz, freq)
    q = dev.get_series(i, 'q')
    testresult1 = _make_df({'q': q})

    expected = pd.DataFrame({'q': q})
    if tz is None:
        expected = expected.tz_localize('Europe/Berlin')
    expected.index.freq = freq
    
    pd.testing.assert_frame_equal(testresult1, expected, check_names=False)
   
    if tz:
        w = q / q.duration
        testresult2 = _make_df({'w': w})
        pd.testing.assert_frame_equal(testresult2, expected, check_names=False)


# def test_makedf1():
#     i = dev.get_index()
#     q = dev.get_series(i, 'q')
#     expected = pd.DataFrame(q)
    

# def test_makedf(data, expected):
#     pass
