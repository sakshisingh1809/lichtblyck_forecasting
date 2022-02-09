from lichtblyck.core.pfline.base import PfLine
from lichtblyck.core.pfline.single import SinglePfLine
from lichtblyck.core.pfline.multi import MultiPfLine
from lichtblyck.core.develop import dev
import pytest


@pytest.mark.parametrize("inputtype", ["df", "dict", "pfline"])
@pytest.mark.parametrize("outputtype", [SinglePfLine, MultiPfLine, None])
def test_pfline_init(inputtype, outputtype):
    """Test if object can be initialized correctly."""

    if outputtype is SinglePfLine:
        spfl = dev.get_singlepfline()
        if inputtype == "df":
            data_in = spfl.df()
        elif inputtype == "dict":
            data_in = {name: s for name, s in spfl.df().items()}
        else:  # pfline
            data_in = spfl
    elif outputtype is MultiPfLine:
        mpfl = dev.get_pfline()
        if inputtype == "df":
            return  # no way to call with dataframe
        elif inputtype == "dict":
            data_in = {name: pfl for name, pfl in mpfl.items()}
        else:  # pfline
            data_in = mpfl
    else:  # Error
        if inputtype == "df":
            # dataframe with columns that don't make sense
            data_in = dev.get_dataframe().rename(columns=dict(zip("wqpr", "abcd")))
        elif inputtype == "dict":
            # dictionary with mix of series and pflines
            data_in = {"p": dev.get_series(name="p"), "partA": dev.get_singlepfline()}
        else:  # pfline
            return

    if outputtype is None:  # expect error
        with pytest.raises(NotImplementedError):
            _ = PfLine(data_in)
    else:
        result = PfLine(data_in)
        assert type(result) is outputtype
