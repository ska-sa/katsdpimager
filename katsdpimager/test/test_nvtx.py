"""Tests for :mod:`katsdpimager.nvtx`.

It tests both the real and the fallback implementation. It is unable to test
that it actually inserts the intended values into the NVIDIA tools, but it
can check that nothing crashes and that the fallback implementation is
complete.
"""

from .. import nvtx, _nvtx_fallback


class TestNvtx:
    MOD = nvtx

    @classmethod
    def setupClass(cls) -> None:
        # Needs to be done only once for testing with nsys - creating and
        # destroying the same named domain causes problems otherwise.
        cls.domain = cls.MOD.Domain('test_domain')

    def setup(self) -> None:
        self.rstring1 = self.MOD.register_string('hello')
        self.rstring2 = self.MOD.register_string('world', domain=self.domain)

    def test_initialize(self) -> None:
        self.MOD.initialize()

    def test_thread_range_basic(self) -> None:
        with self.MOD.thread_range('basic'):
            pass

    def test_thread_range_payload(self) -> None:
        with self.MOD.thread_range('int', payload=1):
            with self.MOD.thread_range('float', payload=2.5):
                pass

    def test_thread_range_domain(self) -> None:
        with self.MOD.thread_range('domain', domain=self.domain):
            pass

    def test_thread_range_registered_string(self) -> None:
        with self.MOD.thread_range(self.rstring1):
            with self.MOD.thread_range(self.rstring2, domain=self.domain):
                pass


class TestNvtxFallback(TestNvtx):
    MOD = _nvtx_fallback
