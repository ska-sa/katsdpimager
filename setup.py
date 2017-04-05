#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import ctypes.util
import sys
import glob
try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:
    numpy_include = None
try:
    import pkgconfig
    eigen3 = pkgconfig.parse('eigen3')
except ImportError:
    eigen3 = {'include_dirs': set()}

tests_require = ['nose', 'mock', 'scipy']

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
        include_dirs=[numpy_include] + list(eigen3.get('include_dirs', [])),
        depends=glob.glob('katsdpimager/*.h'),
        extra_compile_args=['-std=c++0x', '-g0'],
        libraries=[bp_library, 'boost_system'] + list(eigen3.get('libraries', []))
    )
]

setup(
    name="katsdpimager",
    version="0.1.dev0",
    description="GPU-accelerated radio-astronomy imager",
    author="Bruce Merry and Ludwig Schwardt",
    packages=find_packages(),
    package_data={'': ['imager_kernels/*.mako', 'imager_kernels/*/*.mako']},
    scripts=["scripts/imager.py"],
    ext_package='katsdpimager',
    ext_modules=extensions,
    setup_requires=['numpy', 'pkgconfig'],
    install_requires=[
        'numpy>=1.10.0', 'katsdpsigproc', 'astropy>=1.3', 'progress',
        'pycuda', 'scikit-cuda', 'h5py', 'ansicolors'
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
        'doc': ['sphinx>=1.3.0', 'sphinxcontrib-tikz'] + tests_require,
        'cpu': ['numba'],
        'report': ['aplpy', 'matplotlib', 'mako'],
        'ms': ['python-casacore'],
        'katdal': ['katdal', 'katpoint'],
        'benchmark': ['katpoint']
    }
)
