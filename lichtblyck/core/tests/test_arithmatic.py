from lichtblyck.core.pfline import PfLine
from lichtblyck.tools import nits
from lichtblyck.core import dev
import pandas as pd
import numpy as np
import pytest


i = pd.date_range("2020", periods=20, freq="MS")  # reference
i1 = pd.date_range("2021", periods=20, freq="MS")  # same frequency, part overlap
i2 = pd.date_range("2022-04", periods=20, freq="MS")  # same frequency, no overlap
i3 = pd.date_range("2020-04", periods=8000, freq="H")  # shorter frequency, part overlap
i3 = pd.date_range("2020", periods=8, freq="QS")  # longer frequency, part overlap


@pytest.mark.parametrize(
    ("pfl_in", "value", "returntype", "returnkind"),
    [
        (dev.get_pfline(i, "p"), 12, PfLine, "p", "more"),
        (dev.get_pfline(i, "p"), nits.ureg("12 Eur/MWh"), PfLine, "p"),
        (dev.get_pfline(i, "p"), nits.ureg("12 Eur/kWh"), PfLine, "p"),
        (dev.get_pfline(i, "p"), nits.ureg("12 cent/kWh"), PfLine, "p"),
        (dev.get_pfline(i, "p"), nits.ureg("12 Eur"), None),
        (dev.get_pfline(i, "q"), 8.1, PfLine, "q"),
        (dev.get_pfline(i, "q"), nits.ureg("8.1 GWh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), nits.ureg("8.1 MWh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), nits.ureg("-8.1 kWh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), nits.ureg("-8.1 Wh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), nits.ureg("12 Eur"), None),
        (dev.get_pfline(i, "all"), 5.9, None),
        (dev.get_pfline(i, "all"), nits.ureg("12 Eur"), None),
        (dev.get_pfline(i, "all"), nits.ureg("6 Eur/MWh"), None),
        (dev.get_pfline(i, "all"), nits.ureg("6 MW"), None),
        (dev.get_pfline(i, "p"), dev.get_series(i, "p").pint.magnitude, PfLine, "p"),
        (dev.get_pfline(i, "p"), dev.get_series(i, "p"), PfLine, "p"),
        (dev.get_pfline(i, "p"), dev.get_pfline(i, "p"), PfLine, "p"),
        (dev.get_pfline(i, "p"), dev.get_series(i, "q"), None),
        (dev.get_pfline(i, "p"), dev.get_pfline(i, "q"), None),
        (dev.get_pfline(i, "p"), dev.get_pfline(i, "all"), None),
        (dev.get_pfline(i, "q"), dev.get_series(i, "q").pint.magnitude, PfLine, "q"),
        (dev.get_pfline(i, "q"), dev.get_series(i, "q"), PfLine, "q"),
        (dev.get_pfline(i, "q"), dev.get_pfline(i, "q"), PfLine, "q"),
        (dev.get_pfline(i, "all"), dev.get_pfline(i, "all"), None),
    ],
)
def test_pfl_addition(pfl_in, value, returntype, returnkind=None):
    # Check error is raised.
    if returntype is None:
        with pytest.raises:
            _ = pfl_in + value
        with pytest.raises:
            _ = value + pfl_in
    # Check return type.
    for out in (pfl_in + value, value + pfl_in):
        assert isinstance(out, returntype)
        if returntype is PfLine:
            if returnkind:
                assert out.kind == returnkind
