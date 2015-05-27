#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install -r requirements.txt
pip install coverage
nosetests --with-coverage --cover-erase --cover-package=katsdpimager --cover-xml
# Hack to try to make Jenkins Cobertura plugin find the source
sed -i -e 's!katsdppipelines/katsdpimager</source>!katsdppipelines</source>!' \
       -e 's!filename="katsdpimager/!filename="katsdpimager/katsdpimager/!g' \
       coverage.xml
