from setuptools import setup
from setuptools import find_packages

setup(
    name="Lichtblyck",
    version="0.1.dev1",
    packages=find_packages(exclude=["*tests.*", "*tests"]),
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
    python_requires=">=3.8",
    include_package_data=True,
    long_description=open("README.md").read(),
)
