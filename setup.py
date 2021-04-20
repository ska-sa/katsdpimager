#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import glob
import os


class Missing:
    """Raise an exception only when trying to convert it to a string.

    This allows commands like ``setup.py clean`` to work even when the package
    is not present, but makes it fail when trying to build the extension.
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        raise RuntimeError(
            ("The {name} module was not found. You should use pip to install, "
             "which will automatically install {name} in the build environment. "
             "If you are using pip, check that you're using at least "
             "version 19.0.").format(name=self.name))


try:
    import pkgconfig
    eigen3 = pkgconfig.parse('eigen3')
except ImportError:
    eigen3 = {'include_dirs': {Missing('pkgconfig')}}

try:
    import pybind11
    pybind11_include_dirs = [pybind11.get_include()]
except ImportError:
    pybind11_include_dirs = [Missing('pybind11')]

tests_require = ['nose', 'scipy', 'fakeredis[lua]']
install_requires = [
    'ansicolors',
    'astropy>=1.3',
    'contextvars; python_version<"3.7"',
    'h5py',
    'katsdpsigproc[CUDA]>=1.2',
    'katpoint<1',
    'numba',
    'numpy>=1.17.0',
    'progress>=1.5',
    'scikit-cuda',
    'scipy'
]

cffi_modules = []
if os.path.exists('/usr/local/cuda/include/nvtx3/nvToolsExt.h'):
    cffi_modules.append('katsdpimager/nvtx_build.py:ffibuilder')
if cffi_modules:
    install_requires.append('cffi')


root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, 'README.rst')) as f:
    long_description = f.read()

extensions = [
    Extension(
        '_preprocess',
        sources=['katsdpimager/preprocess.cpp'],
        language='c++',
        include_dirs=pybind11_include_dirs + list(eigen3.get('include_dirs', [])),
        depends=glob.glob('katsdpimager/*.h'),
        extra_compile_args=['-std=' + os.environ.get('KATSDPIMAGER_STD_CXX', 'c++1y'),
                            '-g0', '-fvisibility=hidden', '-fopenmp'],
        extra_link_args=['-fopenmp'],
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
                       'models/*/*/*/*.h5', 'templates/*', 'static/*']},
    scripts=["scripts/imager.py",
             "scripts/imager-mkat-pipeline.py",
             "scripts/imager-mkat-report.py"],
    ext_package='katsdpimager',
    ext_modules=extensions,
    cffi_modules=cffi_modules,
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
        'doc': ['sphinx>=1.3.0', 'sphinxcontrib-tikz', 'sphinx-rtd-theme'] + tests_require,
        'cpu': [],
        'report': ['aplpy', 'matplotlib', 'mako'],
        'ms': ['python-casacore'],
        'katdal': ['katdal>=0.18', 'katsdpmodels[requests]'],
        'benchmark': ['katpoint'],
        'pipeline': [
            'bokeh>=2.0.0',
            'jinja2>=2.11',
            'katdal[s3credentials]',
            'katsdpservices',
            'katsdpimageutils',
            'matplotlib>=3.1'
        ]
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
