ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build AS build

# Build ffmpeg from source. The Ubuntu package has all sorts of dependencies
# (including X11) that we'd rather avoid. We can also build a minimal ffmpeg
# that just does the encoding we want.
USER root
RUN apt-get update && apt-get -y install nasm libx264-dev


# Unpin cffi to allow uv to resolve a version with pre-built wheels
# cffi==1.14.5 does not have a pre-built wheel for Python 3.10 and
# requires libffi-dev to compile from source. Unpinning allows uv
# to resolve a newer version that ships a wheel for Python 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends libffi-dev && apt-get clean

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


ENV PATH="/home/kat/.local/bin:$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
# Install dependencies
RUN mkdir -p /tmp/install/katsdpimager
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
# Replace install_pinned.py with uv
RUN uv pip compile /tmp/install/requirements.txt \
        -c /home/kat/docker-base/base-requirements.lock \
        -c /home/kat/docker-base/gpu-requirements.lock \
        -o /tmp/install/requirements.lock && \
    uv pip install --no-deps -r /tmp/install/requirements.lock && \
    uv pip check


# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpimager
WORKDIR /tmp/install/katsdpimager

# Remove setup.py clean because uv handles this automatically
RUN uv pip install --no-deps .[ms,katdal,pipeline] && \
    uv pip check


#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime
LABEL maintainer="sdpdev+katsdpimager@ska.ac.za"

# ffmpeg is linked against libx264
USER root
# Upgrade libx264 from libx264-155 to libx264-163 for Ubuntu 22.04
RUN apt-get update && apt-get -y --no-install-recommends install  libx264-163 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
USER kat

COPY --from=build /tmp/ffmpeg-install /
COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
