#!/bin/bash
set -e -x

pip install -U pip setuptools wheel
install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
    -r requirements.txt
if [ "$1" = "" ]; then
    install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
        -r test-requirements.txt
    pip install -e --no-index '.[test]'
    nosetests --with-xunit --with-coverage --cover-erase --cover-package=katsdpimager --cover-xml
    # Hack to make Jenkins Cobertura plugin find the source
    sed -i -e 's!katsdppipelines/katsdpimager</source>!katsdppipelines</source>!' \
           -e 's!filename="katsdpimager/!filename="katsdpimager/katsdpimager/!g' \
           coverage.xml
elif [ "$1" = "images" ]; then
    install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
        -r report-requirements.txt
    pip install -e --no-index '.[report]'
    cd tests
    rm -rf simple.ms simple.ms_p0 simple.lsm.html report
    makems
    mv simple.ms_p0 simple.ms
    # makems hard-codes LOFAR antenna information; replace with MeerKAT
    rm -r simple.ms/ANTENNA
    cp -a MeerKAT64_ANTENNAS simple.ms/ANTENNA
    # Create the Sky model
    tigger-convert --rename --format "name ra_h ra_m ra_s dec_d dec_m dec_s i q u v" -f simple.lsm.txt simple.lsm.html
    # Meqtrees won't run against our virtualenv, since it uses system Python
    # packages. We strip the virtualenv off PATH.
    PATH=${PATH#*:} meqtree-pipeliner.py -c batch.tdlconf '[turbo-sim]' ms_sel.msname=simple.ms /usr/lib/python2.7/dist-packages/Cattery/Siamese/turbo-sim.py =_tdl_job_1_simulate_MS
    ./images_report.py simple.ms report
    cp simple.lsm.txt simple.lsm.html makems.cfg batch.tdlconf report/
fi
