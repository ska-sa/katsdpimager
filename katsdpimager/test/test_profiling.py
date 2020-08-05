"""Tests for :py:mod:`katsdpimager.profiling`."""

import contextvars
from unittest import mock

from nose.tools import assert_equal, assert_not_equal, assert_is_not

from ..profiling import Record, Profiler, profile, labels


class TestRecord:
    def setup(self) -> None:
        self.record = Record('foo', 12.5, 13.75, {'str': 'hello', 'int': 1, 'float': 2.5})

    def test_construct(self) -> None:
        assert_equal(self.record.name, 'foo')
        assert_equal(self.record.start_time, 12.5)
        assert_equal(self.record.stop_time, 13.75)
        assert_equal(self.record.labels, {'str': 'hello', 'int': 1, 'float': 2.5})

    def test_elapsed(self) -> None:
        assert_equal(self.record.elapsed, 1.25)

    def test_eq_hash(self) -> None:
        # Hash != comparisons could fail by chance, but it's very unlikely
        other = Record(self.record.name, self.record.start_time, self.record.stop_time,
                       {'str': 'other', 'int': 1, 'float': 2.5})
        assert_not_equal(self.record, other)
        assert_not_equal(hash(self.record), hash(other))
        other = Record('othername', self.record.start_time, self.record.stop_time,
                       self.record.labels)
        assert_not_equal(self.record, other)
        assert_not_equal(hash(self.record), hash(other))
        other = Record(self.record.name, self.record.start_time, self.record.stop_time,
                       self.record.labels)
        assert_equal(self.record, other)
        assert_equal(hash(self.record), hash(other))


@mock.patch('time.monotonic', return_value=0.0)
class TestProfiler:
    def setUp(self) -> None:
        self.profiler = Profiler()

    def test_simple(self, monotonic):
        with self.profiler.profile('foo'):
            monotonic.return_value = 1.0
        assert_equal(self.profiler.records, [Record('foo', 0.0, 1.0, {})])

    def test_labels(self, monotonic):
        with labels(x=0, z=2.5):
            with self.profiler.profile('foo', x=1, y='hello'):
                monotonic.return_value = 1.0
        assert_equal(self.profiler.records, [
            Record('foo', 0.0, 1.0, {'x': 1, 'y': 'hello', 'z': 2.5})
        ])

    def test_nested(self, monotonic):
        with self.profiler.profile('foo', x=1):
            monotonic.return_value += 1.0
            with self.profiler.profile('bar', y=2):
                monotonic.return_value += 1.0
            monotonic.return_value += 1.0
        assert_equal(self.profiler.records, [
            Record('bar', 1.0, 2.0, {'y': 2}),
            Record('foo', 0.0, 3.0, {'x': 1})
        ])

    def test_pause(self, monotonic):
        with self.profiler.profile('foo', x=1) as stopwatch:
            monotonic.return_value += 1.0
            stopwatch.stop()
            monotonic.return_value += 1.0
            stopwatch.start()
            monotonic.return_value += 1.0
        assert_equal(self.profiler.records, [
            Record('foo', 0.0, 1.0, {'x': 1}),
            Record('foo', 2.0, 3.0, {'x': 1})
        ])

    def test_context(self, monotonic):
        def inner():
            Profiler.set_profiler(self.profiler)
            with profile('foo', x=1):
                monotonic.return_value += 1.0

        context = contextvars.copy_context()
        context.run(inner)
        assert_is_not(Profiler.get_profiler(), self.profiler)
        assert_equal(self.profiler.records, [Record('foo', 0.0, 1.0, {'x': 1})])
