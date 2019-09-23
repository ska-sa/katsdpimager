#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import glob

import pkgconfig


eigen3 = pkgconfig.parse('eigen3')
tests_require = ['nose', 'scipy', 'fakeredis']

extensions = [
    Extension(
        '_preprocess',
        sources=['katsdpimager/preprocess.cpp'],
        language='c++',
        include_dirs=['3rdparty/pybind11/include'] + list(eigen3.get('include_dirs', [])),
        depends=glob.glob('katsdpimager/*.h'),
        extra_compile_args=['-std=c++1y', '-g0', '-fvisibility=hidden'],
        libraries=list(eigen3.get('libraries', []))
    )
]

setup(
    name="katsdpimager",
    version="0.1.dev0",
    description="GPU-accelerated radio-astronomy imager",
    author="Bruce Merry and Ludwig Schwardt",
    packages=find_packages(),
    package_data={'': ['imager_kernels/*.mako', 'imager_kernels/*/*.mako']},
    scripts=["scripts/imager.py", "scripts/imager-mkat-pipeline.py"],
    ext_package='katsdpimager',
    ext_modules=extensions,
    python_requires='>=3.5',       # Somewhat arbitrary choice; only tested with 3.6+
    install_requires=[
        'numpy>=1.10.0', 'katsdpsigproc', 'katpoint', 'astropy>=1.3', 'progress',
        'pycuda', 'scikit-cuda', 'h5py', 'ansicolors'
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
        'doc': ['sphinx>=1.3.0', 'sphinxcontrib-tikz', 'sphinx-rtd-theme'] + tests_require,
        'cpu': ['numba'],
        'report': ['aplpy', 'matplotlib', 'mako'],
        'ms': ['python-casacore'],
        'katdal': ['katdal', 'scipy>=0.17'],
        'benchmark': ['katpoint'],
        'pipeline': ['katsdpservices']
    }
)
