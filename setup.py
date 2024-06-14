from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Maestro'
LONG_DESCRIPTION = 'Implementation of Maestro in PyTorch'

setup(
    name="maestro",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
)
