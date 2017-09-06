#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import ctypes.util
import sys
import glob
import importlib
try:
    import pkgconfig
    eigen3 = pkgconfig.parse('eigen3')
except ImportError:
    eigen3 = {'include_dirs': set()}

tests_require = ['nose', 'mock', 'scipy']

class get_include(object):
    """Helper class to defer importing a module until build time for fetching
    the include directory.
    """
    def __init__(self, module, *args, **kwargs):
        self.module = module
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        module = importlib.import_module(self.module)
        return getattr(module, 'get_include')(*self.args, **self.kwargs)

extensions = [
    Extension(
        '_preprocess',
        sources=['katsdpimager/preprocess.cpp'],
        language='c++',
        include_dirs=[
            get_include('pybind11'),
            get_include('pybind11', user=True)] + list(eigen3.get('include_dirs', [])),
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
    scripts=["scripts/imager.py"],
    ext_package='katsdpimager',
    ext_modules=extensions,
    setup_requires=['pkgconfig', 'pybind11>=2.2.0'],
    install_requires=[
        'numpy>=1.10.0', 'katsdpsigproc', 'astropy>=1.3', 'progress',
        'pycuda', 'scikit-cuda', 'h5py', 'ansicolors', 'six'
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
        'doc': ['sphinx>=1.3.0', 'sphinxcontrib-tikz', 'sphinx-rtd-theme'] + tests_require,
        'cpu': ['numba'],
        'report': ['aplpy', 'matplotlib', 'mako'],
        'ms': ['python-casacore'],
        'katdal': ['katdal', 'katpoint', 'scipy>=0.17'],
        'benchmark': ['katpoint']
    }
)
