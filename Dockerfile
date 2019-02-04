FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-build as build

# Enable Python 2 ve
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"

# Install dependencies
RUN mkdir -p /tmp/install/katsdpimager
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpimager
WORKDIR /tmp/install/katsdpimager
RUN python ./setup.py clean
RUN pip install --no-deps .[ms,katdal]
RUN pip check

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-runtime:18.04

COPY --from=build --chown=kat:kat /home/kat/ve /home/kat/ve
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"
