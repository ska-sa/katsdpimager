FROM quay.io/pypa/manylinux2010_x86_64

# yum provides only an old version of Eigen3 (too old)
RUN cd /tmp && \
    curl https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.bz2 | tar -jx && \
    cd eigen-3.3.9 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr && \
    make install

COPY . /tmp/katsdpimager
RUN /tmp/katsdpimager/manylinux/generate_wheels.sh
