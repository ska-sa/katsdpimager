FROM quay.io/pypa/manylinux2010_x86_64

# yum provides only an old version of Eigen3 (too old)
RUN cd /tmp && \
    curl https://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2 | tar -jx && \
    cd eigen-eigen-323c052e1731 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr && \
    make install

COPY . /tmp/katsdpimager
RUN /tmp/katsdpimager/manylinux/generate_wheels.sh
