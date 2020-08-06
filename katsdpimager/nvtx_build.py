"""Build the _nvtx module."""

from cffi import FFI

ffibuilder = FFI()
ffibuilder.set_unicode(True)
ffibuilder.set_source(
    "._nvtx", "#include <nvtx3/nvToolsExt.h>",
    include_dirs=['/usr/local/cuda/include']
)
ffibuilder.cdef(
    r"""
    void nvtxInitialize(const void *reserved);
    int nvtxRangePushW(const wchar_t *message);
    int nvtxRangePop(void);
    """
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
