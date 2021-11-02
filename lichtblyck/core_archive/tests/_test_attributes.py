from lichtblyck.core import attributes as a
import pandas as pd
import numpy as np
import pytest

# Extend functionality for testing
pd.DataFrame.w = a._w
pd.DataFrame.p = a._p


def dotest_w(i):
    sw = pd.Series(np.random.rand(len(i)), i)
    sp = pd.Series(np.random.rand(len(i)), i)
    # Series has no column 'w'.
    with pytest.raises(AttributeError):
        sw.w
    # Dataframe without column 'w'.
    df1 = pd.DataFrame({"a": sw})
    with pytest.raises(AttributeError):
        df1.w
    # Dataframe with 'w' as one of its colums.
    df2 = pd.DataFrame({"w": sw})
    np.testing.assert_allclose(df2.w.values, sw.values)
    df3 = pd.DataFrame({"w": sw, "p": sp})
    np.testing.assert_allclose(df3.w.values, sw.values)
    # Dataframe with 'w' as column of the 'subframes'.
    df4 = pd.DataFrame({("pfA", "w"): sw})
    np.testing.assert_allclose(df4.w.values, sw.values)
    df5 = pd.DataFrame(
        {("pfA", "w"): sw, ("pfA", "p"): sp, ("pfB", "w"): sw * 2, ("pfB", "p"): sp + 2}
    )
    np.testing.assert_allclose(df5.w.values, sw.values * 3)
    df6 = pd.DataFrame(
        {
            ("pfA", "w"): sw,
            ("pfA", "p"): sp,
            ("pfB", "w"): sw * 2,
            ("pfC", "w"): sw * 3,
            ("pfC", "p"): sp + 3,
        }
    )
    np.testing.assert_allclose(df6.w.values, sw.values * 6)
    df7 = pd.DataFrame({("pfA", "w"): sw, ("pfA", "p"): sp, ("pfB", "p"): sp + 2})
    with pytest.raises(AttributeError):
        df7.w
    df8 = pd.DataFrame(
        {("pfA", "w"): sw, ("pfA", "p"): sp, ("pfB", "p"): sp + 2, ("pfC", "w"): sw * 3}
    )
    with pytest.raises(AttributeError):
        df8.w
    # Dataframe with 'w' column at various levels.
    df9 = pd.DataFrame(
        {
            ("pfA", "", "w"): sw,
            ("pfA", "", "p"): sp,
            ("pfB", "pfB1", "w"): sw * 2,
            ("pfB", "pfB1", "p"): sp + 2,
            ("pfB", "pfB2", "w"): sw * 3,
            ("pfB", "pfB2", "p"): sp + 3,
        }
    )
    np.testing.assert_allclose(df9.w.values, sw.values * 6)
    np.testing.assert_allclose(df9.pfB.w.values, sw.values * 5)
    df10 = pd.DataFrame(
        {
            ("pfA", "w", ""): sw,
            ("pfA", "p", ""): sp,
            ("pfB", "pfB1", "w"): sw * 2,
            ("pfB", "pfB1", "p"): sp + 2,
            ("pfB", "pfB2", "w"): sw * 3,
            ("pfB", "pfB2", "p"): sp + 3,
        }
    )
    np.testing.assert_allclose(df10.w.values, sw.values * 6)
    np.testing.assert_allclose(df10.pfB.w.values, sw.values * 5)


def dotest_p(i):
    sw = pd.Series(np.random.rand(len(i)), i)
    sp = pd.Series(np.random.rand(len(i)), i)
    # Series.
    with pytest.raises(AttributeError):
        sp.p
    # Dataframe without column 'p'.
    df1 = pd.DataFrame({"a": sp})
    with pytest.raises(AttributeError):
        df1.p
    # Dataframe with 'p' as one of its colums.
    df2 = pd.DataFrame({"p": sp})
    np.testing.assert_allclose(df2.p.values, sp.values)
    df3 = pd.DataFrame({"w": sw, "p": sp})
    np.testing.assert_allclose(df3.p.values, sp.values)
    # Dataframe with 'p' as column of the 'subframes'.
    df4 = pd.DataFrame({("pfA", "p"): sp})
    np.testing.assert_allclose(df4.p.values, sp.values)
    df5 = pd.DataFrame(
        {("pfA", "w"): sw, ("pfA", "p"): sp, ("pfB", "w"): sw * 2, ("pfB", "p"): sp + 2}
    )
    np.testing.assert_allclose(
        df5.p.values, ((sw * sp + (sw * 2) * (sp + 2)) / (sw * 3)).values
    )
    df6 = pd.DataFrame(
        {
            ("pfA", "w"): sw,
            ("pfA", "p"): sp,
            ("pfB", "w"): sw * 2,
            ("pfC", "w"): sw * 3,
            ("pfC", "p"): sp + 3,
        }
    )
    with pytest.raises(AttributeError):
        df6.p
    df7 = pd.DataFrame({("pfA", "w"): sw, ("pfA", "p"): sp, ("pfB", "p"): sp + 2})
    with pytest.raises(AttributeError):
        df7.p
    df8 = pd.DataFrame(
        {("pfA", "w"): sw, ("pfA", "p"): sp, ("pfB", "p"): sp + 2, ("pfC", "w"): sw * 3}
    )
    with pytest.raises(AttributeError):
        df8.p
    # Dataframe with 'p' column at various levels.
    df9 = pd.DataFrame(
        {
            ("pfA", "", "w"): sw,
            ("pfA", "", "p"): sp,
            ("pfB", "pfB1", "w"): sw * 2,
            ("pfB", "pfB1", "p"): sp + 2,
            ("pfB", "pfB2", "w"): sw * 3,
            ("pfB", "pfB2", "p"): sp + 3,
        }
    )
    np.testing.assert_allclose(
        df9.p.values,
        ((sw * sp + (sw * 2) * (sp + 2) + (sw * 3) * (sp + 3)) / (sw * 6)).values,
    )
    np.testing.assert_allclose(
        df9.pfB.p.values,
        (((sw * 2) * (sp + 2) + (sw * 3) * (sp + 3)) / (sw * 5)).values,
    )
    df10 = pd.DataFrame(
        {
            ("pfA", "w", ""): sw,
            ("pfA", "p", ""): sp,
            ("pfB", "pfB1", "w"): sw * 2,
            ("pfB", "pfB1", "p"): sp + 2,
            ("pfB", "pfB2", "w"): sw * 3,
            ("pfB", "pfB2", "p"): sp + 3,
        }
    )
    np.testing.assert_allclose(
        df10.p.values,
        ((sw * sp + (sw * 2) * (sp + 2) + (sw * 3) * (sp + 3)) / (sw * 6)).values,
    )
    np.testing.assert_allclose(
        df10.pfB.p.values,
        (((sw * 2) * (sp + 2) + (sw * 3) * (sp + 3)) / (sw * 5)).values,
    )


def test_w():
    for tz in [None, "Europe/Berlin"]:
        dotest_w(
            pd.date_range(
                "2020-01-01", freq="M", periods=np.random.randint(10, 100), tz=tz
            )
        )
        dotest_w(
            pd.date_range(
                "2020-01-01", freq="D", periods=np.random.randint(100, 1000), tz=tz
            )
        )
        dotest_w(
            pd.date_range(
                "2020-01-01", freq="H", periods=np.random.randint(1000, 100_000), tz=tz
            )
        )
        dotest_w(
            pd.date_range(
                "2020-01-01",
                freq="15T",
                periods=np.random.randint(1000, 100_000),
                tz=tz,
            )
        )
