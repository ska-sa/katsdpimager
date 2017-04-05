FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu

LABEL com.nvidia.volumes.needed="nvidia_driver" com.nvidia.cuda.version="8.0"

# Install system packages
USER root
RUN apt-get -y update && apt-get -y --no-install-recommends install \
        casacore-dev
USER kat

# Install dependencies
RUN mkdir -p /tmp/install/katsdpimager
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpimager
RUN cd /tmp/install/katsdpimager && \
    python ./setup.py clean && \
    pip install --no-deps .[ms,katdal] && \
    pip check
