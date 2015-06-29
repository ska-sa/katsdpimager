#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install -r requirements.txt
pip install sphinxcontrib-napoleon sphinxcontrib-tikz
pip install 'sphinx>=1.3.0'  # napoleon forces an older version of sphinx
rm -rf doc/_build
make -C doc html
