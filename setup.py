#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import ctypes.util
import sys
try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:
    numpy_include = None

tests_require = ['nose', 'mock']

# Different OSes install the Boost.Python library under different names
bp_library_names = [
    'boost_python-py{0}{1}'.format(sys.version_info.major, sys.version_info.minor),
    'boost_python{0}'.format(sys.version_info.major),
    'boost_python',
    'boost_python-mt']
for name in bp_library_names:
    if ctypes.util.find_library(name):
        bp_library = name
        break
else:
    raise RuntimeError('Cannot find Boost.Python library')

extensions = [
    Extension(
        '_preprocess',
        sources=['katsdpimager/preprocess.cpp'],
        language='c++',
        include_dirs=[numpy_include],
        extra_compile_args=['-std=c++0x', '-g0'],
        libraries=[bp_library, 'boost_system'])
]

setup(
    name="katsdpimager",
    version="0.1.dev0",
    description="GPU-accelerated radio-astronomy imager",
    author="Bruce Merry and Ludwig Schwardt",
    packages=find_packages(),
    scripts=["scripts/imager.py"],
    ext_package='katsdpimager',
    ext_modules=extensions,
    setup_requires=['numpy'],
    install_requires=[
        'numpy', 'scipy', 'katsdpsigproc', 'python-casacore', 'astropy', 'progress',
        'pycuda', 'scikit-cuda', 'h5py', 'ansicolors'
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
        'doc': ['sphinx>=1.3.0'],
        'cpu': ['numba'],
        'report': ['aplpy', 'matplotlib', 'mako', 'katpoint']
    }
)
