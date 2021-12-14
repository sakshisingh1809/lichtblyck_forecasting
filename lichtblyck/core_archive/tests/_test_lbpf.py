"""Testing LbPf."""


from lichtblyck import SinglePf, MultiPf, LbPf
from lichtblyck.core.dev import (
    get_index,
    get_dataframe,
    get_singlepf,
    get_multipf_standardcase,
    get_multipf_allcases,
    get_lbpf_nosubs,
    get_lbpf_subs_standardcase,
    get_lbpf_subs_allcases,
    OK_FREQ,
    OK_COL_COMBOS,
)
from typing import Union
import pandas as pd
import numpy as np
import pytest




@pytest.mark.parametrize("tz", ["Europe/Berlin"])
@pytest.mark.parametrize("freq", OK_FREQ)
@pytest.mark.parametrize(
    "pffunction", [get_singlepf, get_multipf_standardcase, get_multipf_allcases]
)
def test_lbpf_equalindex(tz, freq, pffunction):

    i = get_index(freq, tz)
    offtake = pffunction(i)
    sourced = pffunction(i)

    # Specify offtake only.

    lbpf = LbPf(offtake=offtake, name='test')
    pd.testing.assert_frame_equal(lbpf.Offtake.df(), offtake.df())
    assert lbpf.Sourced == 0
    pd.testing.assert_frame_equal(lbpf.Unhedged.df('w'), offtake.df('w'))
    pd.testing.assert_series_equal(lbpf.Unhedged.w, lbpf.w)

    # Specify sourced only.

    lbpf = LbPf(sourced=sourced, name='test')
    assert lbpf.Offtake == 0
    pd.testing.assert_frame_equal(lbpf.Sourced.df(), sourced.df())
    pd.testing.assert_frame_equal(lbpf.Unhedged.df('w'), sourced.df('w'))
    pd.testing.assert_series_equal(lbpf.Unhedged.w, lbpf.w)

    # Specify both.

    lbpf = LbPf(offtake=offtake, sourced=sourced, name='test')
    pd.testing.assert_frame_equal(lbpf.Offtake.df(), offtake.df())
    pd.testing.assert_frame_equal(lbpf.Sourced.df(), sourced.df())
    pd.testing.assert_frame_equal(lbpf.Unhedged.df('w'), sourced.df('w') + offtake.df('w'))
    pd.testing.assert_series_equal(lbpf.Unhedged.w, lbpf.w)

    

    # for column in ["q", "w"]:
    #     df = get_dataframe(i, column)
    #     sp = SinglePf(df, "test")
    #     assert_w_q_compatible(sp)
    #     assert_raises_attributeerror(sp, no="wpqr")
    #     assert sp.r.isna().all()
    #     assert sp.p.isna().all()
    #     assert sp.index.freq == freq
    #     assert sp.index.tz is not None

    # for column in ["p", "r"]:
    #     with pytest.raises(ValueError):  # information about power is missing.
    #         df = get_dataframe(get_index(freq, tz), column)
    #         SinglePf(df, "test").duration

    # # Specify two. That's good, if it's not (w and q).

    # for columns in ["pr", "qr", "pq", "wp", "wr"]:
    #     df = get_dataframe(get_index(freq, tz), columns)
    #     sp = SinglePf(df, "test")
    #     assert_w_q_compatible(sp)
    #     assert_p_q_r_compatible(sp)
    #     assert_raises_attributeerror(sp, no="wpqr")
    #     assert sp.index.freq == freq
    #     assert sp.index.tz is not None

    # with pytest.raises(ValueError):
    #     df = get_dataframe(get_index(freq, tz), "wq")
    #     SinglePf(df, "test").duration

    # # Specify three or four. Always incompatible.

    # for columns in ["pqr", "wpr", "qwp", "qwr", "pqrw"]:
    #     with pytest.raises(ValueError):
    #         df = get_dataframe(get_index(freq, tz), columns)
    #         SinglePf(df, "test").duration
