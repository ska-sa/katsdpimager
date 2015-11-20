#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
    -r requirements.txt
install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
    -r doc-requirements.txt
pip install --no-index '.[doc]'
rm -rf doc/_build
make -C doc html
