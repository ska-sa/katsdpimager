#!/bin/bash
set -e

cd /tmp/katsdpimager
mkdir -p /output
for d in /opt/python/cp{36,37,38}*; do
    git clean -xdf
    # We know the compiler supports C++17, and using it avoids the need for Boost
    KATSDPIMAGER_STD_CXX=c++17 $d/bin/pip wheel --no-deps .
    auditwheel repair --plat manylinux2010_x86_64 -w /output katsdpimager-*-`basename $d`-linux_*.whl
done
