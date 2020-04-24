#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import glob
import os


class MissingPkgconfig:
    """Raise an exception only when trying to convert it to a string.

    This allows commands like ``setup.py clean`` to work even when pkgconfig
    is not present, but makes it fail when trying to build the extension.
    """
    def __str__(self):
        raise RuntimeError(
            'The pkgconfig module was not found. Try upgrading pip to the latest '
            'version and using it to install.')


try:
    import pkgconfig
    eigen3 = pkgconfig.parse('eigen3')
except ImportError:
    eigen3 = {'include_dirs': {MissingPkgconfig()}}

tests_require = ['nose', 'scipy', 'fakeredis[lua]']

root_dir = os.path.dirname(__file__)
pybind11_dir = os.path.join(root_dir, '3rdparty', 'pybind11', 'include', 'pybind11')
if not os.path.exists(pybind11_dir):
    raise RuntimeError(
        'pybind11 directory not found in source tree. If this is a git checkout, you '
        'can probably fix it by running "git submodule update --init --recursive".')

with open(os.path.join(root_dir, 'README.rst')) as f:
    long_description = f.read()

extensions = [
    Extension(
        '_preprocess',
        sources=['katsdpimager/preprocess.cpp'],
        language='c++',
        include_dirs=['3rdparty/pybind11/include'] + list(eigen3.get('include_dirs', [])),
        depends=glob.glob('katsdpimager/*.h'),
        extra_compile_args=['-std=' + os.environ.get('KATSDPIMAGER_STD_CXX', 'c++1y'),
                            '-g0', '-fvisibility=hidden'],
        libraries=list(eigen3.get('libraries', []))
    )
]

setup(
    name="katsdpimager",
    description="GPU-accelerated radio-astronomy spectral line imager",
    long_description=long_description,
    author="MeerKAT SDP Team",
    author_email="sdpdev+katsdpimager@ska.ac.za",
    url="https://github.com/ska-sa/katsdpimager/",
    packages=find_packages(),
    package_data={'': ['imager_kernels/*.mako', 'imager_kernels/*/*.mako',
                       'models/*/*/*/*.h5']},
    scripts=["scripts/imager.py",
             "scripts/imager-mkat-pipeline.py",
             "scripts/fits-video.py",
             "scripts/fits-image.py"],
    ext_package='katsdpimager',
    ext_modules=extensions,
    python_requires='>=3.6',
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
        'pipeline': ['katsdpservices', 'matplotlib']
    },
    use_katversion=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Environment :: Console"
    ]
)
