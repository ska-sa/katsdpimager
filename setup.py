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
        'numpy', 'scipy', 'katsdpsigproc', 'python-casacore', 'astropy', 'progress',
        'numba', 'pycuda', 'scikit-cuda', 'h5py', 'ansicolors'
    ]
)
