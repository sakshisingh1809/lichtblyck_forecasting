import pandas as pd
import pint
import pint_pandas

ureg = pint_pandas.PintType.ureg = pint.UnitRegistry()
ureg.default_format = "~L"

df = pd.DataFrame({"a": [1000, 2000], "b": [3, 5]}).astype(
    {"a": "pint[meter/s]", "b": "pint[km/day]"}
)
q = f"{ureg('1000 meter/second')}"

df.pint.dequantify()


class A:

    def __init__(self, val):
        self.val = val

    def __add__(self, other):
        return A(self.val + other)

    __radd__ = __add__
    def __sub__(self, other):
        return A(self.val - other)

    __rsub__ = __sub__

    def __repr__(self):
        return str(self.val)

