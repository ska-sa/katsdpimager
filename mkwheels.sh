#!/bin/sh
set -e

sudo docker build --pull -t ska-sa/katsdpimager/manylinux -f manylinux/Dockerfile .
mkdir -p wheelhouse
sudo docker run --rm -v "$PWD/wheelhouse:/wheelhouse" ska-sa/katsdpimager/manylinux sh -c 'cp -v /output/*.whl /wheelhouse'
sudo chown `id -u`:`id -g` wheelhouse/*.whl
