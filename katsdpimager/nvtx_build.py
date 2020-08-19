"""Build the _nvtx module."""

from cffi import FFI


ffibuilder = FFI()
ffibuilder.set_unicode(True)
ffibuilder.set_source(
    "._nvtx", "#include <nvtx3/nvToolsExt.h>\n",
    include_dirs=['/usr/local/cuda/include']
)
ffibuilder.cdef(
    r"""
    typedef ...* nvtxDomainHandle_t;
    typedef ...* nvtxStringHandle_t;
    #define NVTX_VERSION ...
    #define NVTX_EVENT_ATTRIB_STRUCT_SIZE ...

    typedef union {
        const char *ascii;
        const wchar_t *unicode;
        nvtxStringHandle_t registered;
        ...;
    } nvtxMessageValue_t;

    typedef enum {
        NVTX_COLOR_UNKNOWN,
        NVTX_COLOR_ARGB,
        ...
    } nvtxColorType_t;

    typedef enum {
        NVTX_MESSAGE_UNKNOWN,
        NVTX_MESSAGE_TYPE_ASCII,
        NVTX_MESSAGE_TYPE_UNICODE,
        NVTX_MESSAGE_TYPE_REGISTERED,
        ...
    } nvtxMessageType_t;

    typedef enum {
        NVTX_PAYLOAD_UNKNOWN,
        NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
        NVTX_PAYLOAD_TYPE_INT64,
        NVTX_PAYLOAD_TYPE_DOUBLE,
        NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
        NVTX_PAYLOAD_TYPE_INT32,
        NVTX_PAYLOAD_TYPE_FLOAT,
        ...
    } nvtxPayloadType_t;

    typedef struct {
        uint16_t version;
        uint16_t size;
        uint32_t category;
        int32_t colorType;
        uint32_t color;
        int32_t payloadType;
        union payload_t
        {
            uint64_t ullValue;
            int64_t llValue;
            double dValue;
            uint32_t uiValue;
            int32_t iValue;
            float fValue;
        } payload;
        int32_t messageType;
        nvtxMessageValue_t message;
        ...;
    } nvtxEventAttributes_t;

    void nvtxInitialize(const void *reserved);
    nvtxStringHandle_t nvtxDomainRegisterStringW(nvtxDomainHandle_t domain, const wchar_t* string);
    int nvtxDomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);
    int nvtxRangePushEx(const nvtxEventAttributes_t *eventAttrib);
    int nvtxRangePushW(const wchar_t *message);
    int nvtxDomainRangePop(nvtxDomainHandle_t domain);
    int nvtxRangePop(void);
    nvtxDomainHandle_t nvtxDomainCreateW(const wchar_t *name);
    void nvtxDomainDestroy(nvtxDomainHandle_t domain);
    """
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
