ARG KATSDPDOCKERBASE_REGISTRY=sdp-docker-registry.kat.ac.za:5000

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build as build

# Build ffmpeg from source. The Ubuntu package has all sorts of dependencies
# (including X11) that we'd rather avoid. We can also build a minimal ffmpeg
# that just does the encoding we want.
USER root
RUN apt-get update && apt-get -y install nasm libx264-dev
WORKDIR /tmp
RUN wget --progress=dot:mega https://ffmpeg.org/releases/ffmpeg-4.2.2.tar.bz2
RUN tar -jxf ffmpeg-4.2.2.tar.bz2
WORKDIR /tmp/ffmpeg-4.2.2
RUN ./configure --prefix=/usr \
    --enable-gpl --enable-libx264 \
    --disable-autodetect \
    --disable-doc \
    --disable-programs --enable-ffmpeg \
    --disable-decoders --enable-decoder=rawvideo \
    --disable-encoders --enable-encoder=libx264 \
    --disable-indevs \
    --disable-outdevs \
    --disable-hwaccels \
    --disable-filters --enable-filter=scale \
    --disable-muxers --enable-muxer=mp4 \
    --disable-demuxers --enable-demuxer=rawvideo
RUN make -j
RUN make DESTDIR=/tmp/ffmpeg-install install-progs install-data
USER kat

# Enable Python 3 ve
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install dependencies
RUN mkdir -p /tmp/install/katsdpimager
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpimager
WORKDIR /tmp/install/katsdpimager
RUN python ./setup.py clean
RUN pip install --no-deps .[ms,katdal,pipeline]
RUN pip check

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime
LABEL maintainer="sdpdev+katsdpimager@ska.ac.za"

# ffmpeg is linked against libx264
USER root
RUN apt-get update && apt-get -y --no-install-recommends install libx264-152 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
USER kat

COPY --from=build /tmp/ffmpeg-install /
COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
