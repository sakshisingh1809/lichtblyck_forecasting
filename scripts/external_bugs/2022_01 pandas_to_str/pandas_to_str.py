import pandas as pd

df = pd.DataFrame({"a": [1, 34, -120]})
df.index.name = "idx"


value = 12345.6789

fstrings = {"a": "{:,.2f}", "b": "{:,.5f}"}
print(fstrings["a"].format(value).replace(",", " "))  # 12 345.68

ffuncs = {}
for key, fstring in fstrings.items():

    def ffunc(fstring):
        return lambda v: fstring.format(v).replace(",", " ")

    ffuncs[key] = ffunc(fstring)
print(ffuncs["a"](value))  # 12 345.67890 <-- ...but 5 decimal places??
