from setuptools import setup
from setuptools import find_packages

setup(
    name="lichtblyck",
    version="0.1.dev2",
    author="Ruud Wijtvliet",
    packages=find_packages(exclude=["*tests.*", "*tests"]),
    description="Analysing and manipulating timeseries related to power and gas offtake portfolios.",
    long_description=open("README.md").read(),
    long_description_content_type="text/x-rst",
    python_requires=">=3.8",
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
    # package_data is data that is deployed within the python package on the
    # user's system. setuptools will get whatever is listed in MANIFEST.in
    include_package_data=True,
)
