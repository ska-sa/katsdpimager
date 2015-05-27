#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install -r requirements.txt
pip install coverage
nosetests --with-coverage --cover-package=katsdpimager --cover-xml
