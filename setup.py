#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="katsdpimager",
    version="0.1.dev0",
    description="GPU-accelerated radio-astronomy imager",
    author="Bruce Merry and Ludwig Schwardt",
    packages=find_packages(),
    scripts=["scripts/imager.py"],
    install_requires=[
        'numpy', 'katsdpsigproc', 'python-casacore', 'astropy', 'progress',
        'numba', 'pycuda', 'scikits.cuda>=0.5.0b1', 'h5py', 'ansicolors'
    ]
)
