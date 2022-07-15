from setuptools import setup
from setuptools import find_packages

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
# https://github.com/scikit-learn/scikit-learn/blob/sklearn/setup.py

#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases

# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'

setup(
    name="Lichtblyck",
    version="0.1.dev04",
    author="Ruud Wijtvliet",
    zip_safe=False,
    packages=find_packages(exclude=["tests"]),
    description="Analysing and manipulating timeseries related to power and gas offtake portfolios.",
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
    python_requires=">=3.8",
    include_package_data=True,
    long_description=open("README.md").read(),
)
