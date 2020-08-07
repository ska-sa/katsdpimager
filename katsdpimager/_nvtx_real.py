import contextlib
from typing import Optional, Generator, Union

from ._nvtx import ffi, lib


def initialize() -> None:
    lib.nvtxInitialize(ffi.NULL)


class RegisteredString:
    def __init__(self, handle: object) -> None:
        self._handle = handle


class Domain:
    def __init__(self, name: str) -> None:
        self._handle = lib.nvtxDomainCreateW(name)

    def __del__(self, _nvtxDomainDestroy=lib.nvtxDomainDestroy) -> None:
        _nvtxDomainDestroy(self._handle)


def register_string(value: str, domain: Optional[Domain] = None) -> RegisteredString:
    domain_handle = domain._handle if domain is not None else ffi.NULL
    handle = lib.nvtxDomainRegisterStringW(domain_handle, value)
    return RegisteredString(handle)


@contextlib.contextmanager
def thread_range(name: Union[str, RegisteredString], *,
                 payload: Union[None, int, float] = None,
                 domain: Optional[Domain] = None) -> Generator[None, None, None]:
    event = ffi.new("nvtxEventAttributes_t *")
    event.version = lib.NVTX_VERSION
    event.size = lib.NVTX_EVENT_ATTRIB_STRUCT_SIZE
    if isinstance(name, RegisteredString):
        event.messageType = lib.NVTX_MESSAGE_TYPE_REGISTERED
        event.message.registered = name._handle
    else:
        c_name = ffi.new("wchar_t[]", name)  # Assign to a var to keep it alive
        event.messageType = lib.NVTX_MESSAGE_TYPE_UNICODE
        event.message.unicode = c_name
    if payload is not None:
        if isinstance(payload, int):
            event.payloadType = lib.NVTX_PAYLOAD_TYPE_INT64
            event.payload.llValue = payload
        elif isinstance(payload, float):
            event.payloadType = lib.NVTX_PAYLOAD_TYPE_DOUBLE
            event.payload.dValue = payload
        else:
            raise TypeError('Payload must be None, int or float')

    # Note: we don't check the return values for errors, because if no
    # profiling tool is active then these functions "fail".
    domain_handle = domain._handle if domain is not None else ffi.NULL
    lib.nvtxDomainRangePushEx(domain_handle, event)
    try:
        yield
    finally:
        lib.nvtxDomainRangePop(domain_handle)
