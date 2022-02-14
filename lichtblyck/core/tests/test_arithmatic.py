from lichtblyck.core.pfline import PfLine
from lichtblyck.tools.nits import Q_
from lichtblyck.core import dev
import pandas as pd
import numpy as np
import pytest

tz = "Europe/Berlin"
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference
i1 = pd.date_range("2021", periods=20, freq="MS", tz=tz)  # same freq, part overlap
i2 = pd.date_range("2022-04", periods=20, freq="MS", tz=tz)  # same freq, no overlap
i3 = pd.date_range(
    "2020-04", periods=8000, freq="H", tz=tz
)  # shorter freq, part overlap
i4 = pd.date_range("2020", periods=8, freq="QS", tz=tz)  # longer freq, part overlap


# . check correct working of dunder methods. E.g. assert correct addition:
# . . pflines having same or different kind
# . . pflines having same or different frequency
# . . pflines covering same or different time periods


@pytest.mark.parametrize(
    ("pfl_in", "value", "returntype", "returnkind"),
    [
        (dev.get_pfline(i, "p"), 12, PfLine, "p"),
        (dev.get_pfline(i, "p"), Q_(12, "Eur/MWh"), PfLine, "p"),
        (dev.get_pfline(i, "p"), Q_(12, "Eur/kWh"), PfLine, "p"),
        (dev.get_pfline(i, "p"), Q_(12, "cent/kWh"), PfLine, "p"),
        (dev.get_pfline(i, "p"), Q_(12, "Eur"), None, None),
        (dev.get_pfline(i, "q"), 8.1, PfLine, "q"),
        (dev.get_pfline(i, "q"), Q_(8.1, "GWh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), Q_(8.1, "MWh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), Q_(-8.1, "kWh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), Q_(-8.1, "Wh"), PfLine, "q"),
        (dev.get_pfline(i, "q"), Q_(12, "Eur"), None, None),
        (dev.get_pfline(i, "all"), 5.9, None, None),
        (dev.get_pfline(i, "all"), Q_(12, "Eur"), None, None),
        (dev.get_pfline(i, "all"), Q_(6, "Eur/MWh"), None, None),
        (dev.get_pfline(i, "all"), Q_(6, "MW"), None, None),
        (dev.get_pfline(i, "p"), dev.get_series(i, "p").pint.magnitude, PfLine, "p"),
        (dev.get_pfline(i, "p"), dev.get_series(i, "p"), PfLine, "p"),
        (dev.get_pfline(i, "p"), dev.get_pfline(i, "p"), PfLine, "p"),
        (dev.get_pfline(i, "p"), dev.get_series(i, "q"), None, None),
        (dev.get_pfline(i, "p"), dev.get_pfline(i, "q"), None, None),
        (dev.get_pfline(i, "p"), dev.get_pfline(i, "all"), None, None),
        (dev.get_pfline(i, "q"), dev.get_series(i, "q").pint.magnitude, PfLine, "q"),
        (dev.get_pfline(i, "q"), dev.get_series(i, "q"), PfLine, "q"),
        (dev.get_pfline(i, "q"), dev.get_pfline(i, "q"), PfLine, "q"),
        (dev.get_pfline(i, "all"), dev.get_pfline(i, "all"), None, None),
        (dev.get_pfline(i, "all"), dev.get_pfline(i2, "all"), PfLine, "all"),
        (dev.get_pfline(i, "all"), dev.get_pfline(i3, "all"), PfLine, "all"),
    ],
)
def test_pfl_addition(pfl_in, value, returntype, returnkind):
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
