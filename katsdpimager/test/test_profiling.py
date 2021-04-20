"""Tests for :py:mod:`katsdpimager.profiling`."""

import contextvars
import functools
import inspect
import io
from unittest import mock
from typing import Generator

from nose.tools import (
    assert_equal, assert_not_equal, assert_greater_equal,
    assert_is, assert_is_none, assert_is_not
)
import katsdpsigproc.abc
from katsdpsigproc.test.test_accel import device_test

from ..profiling import (
    Frame, Record, DeviceRecord, Profiler, CollectProfiler, FlamegraphProfiler,
    profile, profile_function, profile_generator
)


class TestFrame:
    def setup(self) -> None:
        self.frame1 = Frame('foo', {'x': 1, 'y': 2})
        self.frame2 = Frame('bar', {'x': 3, 'z': 'hello'}, self.frame1)

    def test_properties(self) -> None:
        assert_equal(self.frame1.name, 'foo')
        assert_equal(self.frame1.labels, {'x': 1, 'y': 2})
        assert_is_none(self.frame1.parent)
        assert_is(self.frame2.parent, self.frame1)

    def test_reuse(self) -> None:
        frame = Frame('bar', {'x': 3, 'z': 'hello'}, self.frame1)
        assert_is(frame, self.frame2)
        frame = Frame('bar', {'x': 3, 'z': 'goodbye'}, self.frame1)
        assert_is_not(frame, self.frame2)
        assert_equal(frame.labels, {'x': 3, 'z': 'goodbye'})

    def test_eq_hash(self) -> None:
        # Separate tree with identical properties
        frame1 = Frame('foo', {'x': 1, 'y': 2})
        frame2 = Frame('bar', {'x': 3, 'z': 'hello'}, frame1)
        assert_equal(frame1, self.frame1)
        assert_equal(frame2, self.frame2)
        assert_equal(hash(frame1), hash(self.frame1))
        assert_equal(hash(frame2), hash(self.frame2))
        # Same properties as frame2, but no parent
        frame = Frame('bar', {'x': 3, 'z': 'hello'})
        assert_not_equal(frame, frame2)
        assert_not_equal(hash(frame), hash(frame2))
        # Different properties, same parent
        frame = Frame('bar', {}, frame1)
        assert_not_equal(frame, frame2)
        assert_not_equal(hash(frame), hash(frame2))

    def test_stack(self) -> None:
        assert_equal(list(self.frame2.stack()), [self.frame1, self.frame2])

    def test_all_labels(self) -> None:
        assert_equal(self.frame1.all_labels(), {'x': 1, 'y': 2})
        assert_equal(self.frame2.all_labels(), {'x': 3, 'y': 2, 'z': 'hello'})


class TestRecord:
    def setup(self) -> None:
        self.frame = Frame('foo', {'str': 'hello', 'int': 1, 'float': 2.5})
        self.record = Record(self.frame, 12.5, 13.75)

    def test_construct(self) -> None:
        assert_equal(self.record.frame, self.frame)
        assert_equal(self.record.start_time, 12.5)
        assert_equal(self.record.stop_time, 13.75)

    def test_elapsed(self) -> None:
        assert_equal(self.record.elapsed, 1.25)

    def test_eq_hash(self) -> None:
        # Hash != comparisons could fail by chance, but it's very unlikely
        other_frame = Frame('bar', {})
        other = Record(other_frame, self.record.start_time, self.record.stop_time)
        assert_not_equal(self.record, other)
        assert_not_equal(hash(self.record), hash(other))
        other = Record(self.record.frame, self.record.start_time, self.record.stop_time)
        assert_equal(self.record, other)
        assert_equal(hash(self.record), hash(other))


def temporary_context(func):
    """Run a function with its own context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        context = contextvars.copy_context()
        return context.run(func, *args, **kwargs)

    return wrapper


def use_self_profiler(func):
    """Modify a class or member to use self.profiler in all tests."""
    if inspect.isclass(func):
        for name, member in func.__dict__.items():
            if name.startswith('test_') and inspect.isfunction(member):
                setattr(func, name, use_self_profiler(member))
        return func

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        Profiler.set_profiler(self.profiler)
        return func(self, *args, **kwargs)

    return temporary_context(wrapper)


@profile_function()
def empty():
    pass


@mock.patch('time.monotonic', return_value=0.0)
@use_self_profiler
class TestProfiler:
    def setup(self) -> None:
        self.profiler = CollectProfiler()

    def test_simple(self, monotonic) -> None:
        with self.profiler.profile('foo'):
            monotonic.return_value = 1.0
        assert_equal(self.profiler.records, [Record(Frame('foo'), 0.0, 1.0)])

    def test_nested(self, monotonic) -> None:
        with self.profiler.profile('foo', {'x': 1}):
            monotonic.return_value += 1.0
            with self.profiler.profile('bar', {'y': 2}):
                monotonic.return_value += 1.0
            monotonic.return_value += 1.0
        frame1 = Frame('foo', {'x': 1})
        frame2 = Frame('bar', {'y': 2}, frame1)
        assert_equal(self.profiler.records, [
            Record(frame2, 1.0, 2.0),
            Record(frame1, 0.0, 3.0)
        ])

    @device_test
    def test_profile_device(self, monotonic, ctx: katsdpsigproc.abc.AbstractContext,
                            queue: katsdpsigproc.abc.AbstractCommandQueue) -> None:
        with self.profiler.profile('foo', {'x': 1}):
            with self.profiler.profile_device(queue, 'bar', {'y': 2}):
                pass
        assert_equal(len(self.profiler.device_records), 1)
        assert_equal(
            self.profiler.device_records[0].frame,
            Frame('bar', {'y': 2}, Frame('foo', {'x': 1}))
        )
        assert_greater_equal(self.profiler.device_records[0].elapsed, 0.0)

    def test_pause(self, monotonic) -> None:
        with self.profiler.profile('foo', {'x': 1}) as stopwatch:
            monotonic.return_value += 1.0
            stopwatch.stop()
            monotonic.return_value += 1.0
            stopwatch.start()
            monotonic.return_value += 1.0
        frame = Frame('foo', {'x': 1})
        assert_equal(self.profiler.records, [
            Record(frame, 0.0, 1.0),
            Record(frame, 2.0, 3.0)
        ])

    def test_context(self, monotonic) -> None:
        other_profiler = CollectProfiler()

        def inner():
            Profiler.set_profiler(other_profiler)
            with profile('foo', {'x': 1}):
                monotonic.return_value += 1.0

        context = contextvars.copy_context()
        context.run(inner)
        assert_is_not(Profiler.get_profiler(), other_profiler)
        frame = Frame('foo', {'x': 1})
        assert_equal(other_profiler.records, [Record(frame, 0.0, 1.0)])

    def test_profile_function_sequence_labels(self, monotonic):
        @profile_function('blah', ('x', 'y'))
        def inner(y, z, x=4):
            pass

        inner(2, 3)
        inner(y=5, z=6, x=7)
        assert_equal(self.profiler.records, [
            Record(Frame('blah', {'x': 4, 'y': 2}), 0.0, 0.0),
            Record(Frame('blah', {'x': 7, 'y': 5}), 0.0, 0.0)
        ])

    def test_profile_function_mapping_labels(self, monotonic):
        @profile_function('blah',
                          {'x': 'xx', 'y': lambda bound_args: 2 * bound_args.arguments['y']})
        def inner(xx, y):
            pass

        inner(2, 3)
        assert_equal(self.profiler.records, [
            Record(Frame('blah', {'x': 2, 'y': 6}), 0.0, 0.0)
        ])

    def test_profile_function_auto_name(self, monotonic):
        empty()
        assert_equal(self.profiler.records, [
            Record(Frame('katsdpimager.test.test_profiling.empty', {}), 0.0, 0.0)
        ])

    def test_profile_generator(self, monotonic):
        @profile_generator(name='slow_range', labels=('reps',))
        def slow_range(reps: int) -> Generator[int, None, None]:
            for i in range(reps):
                monotonic.return_value += 1.0
                yield i
            monotonic.return_value += 0.5

        result = []
        for item in slow_range(3):
            with profile('other_work'):
                monotonic.return_value += 2.0
                result.append(item)

        assert_equal(result, [0, 1, 2])
        frame = Frame('slow_range', {'reps': 3})
        frame2 = Frame('other_work')
        assert_equal(self.profiler.records, [
            Record(frame, 0.0, 1.0),
            Record(frame2, 1.0, 3.0),
            Record(frame, 3.0, 4.0),
            Record(frame2, 4.0, 6.0),
            Record(frame, 6.0, 7.0),
            Record(frame2, 7.0, 9.0),
            Record(frame, 9.0, 9.5)
        ])


class DummyEvent(katsdpsigproc.abc.AbstractEvent):
    def __init__(self, timestamp: float) -> None:
        self._timestamp = timestamp

    def wait(self) -> None:
        pass

    def time_since(self, prior_event: 'DummyEvent') -> float:
        return self._timestamp - prior_event._timestamp

    def time_till(self, next_event: 'DummyEvent') -> float:
        return next_event._timestamp - self._timestamp


class TestFlamegraphProfiler:
    def setup(self) -> None:
        self.profiler = FlamegraphProfiler()
        self.parent_frame = Frame('parent', {})
        self.child1_frame = Frame('child1', {}, self.parent_frame)
        self.child2_frame = Frame('child2', {}, self.parent_frame)

    def test_host(self) -> None:
        self.profiler.add_record(Record(self.child1_frame, 2.0, 3.0))
        self.profiler.add_record(Record(self.child2_frame, 4.0, 4.5))
        self.profiler.add_record(Record(self.parent_frame, 1.0, 11.0))
        out = io.StringIO()
        self.profiler.write_flamegraph(out)
        assert_equal(
            sorted(out.getvalue().splitlines()),
            [
                'parent 8500000',
                'parent;child1 1000000',
                'parent;child2 500000'
            ]
        )

    def test_device(self) -> None:
        self.profiler.add_device_record(
            DeviceRecord(self.child1_frame, DummyEvent(1.0), DummyEvent(3.0))
        )
        self.profiler.add_device_record(
            DeviceRecord(self.child2_frame, DummyEvent(5.0), DummyEvent(5.5))
        )
        out = io.StringIO()
        self.profiler.write_device_flamegraph(out)
        assert_equal(
            sorted(out.getvalue().splitlines()),
            [
                'parent;child1 2000000',
                'parent;child2 500000'
            ]
        )
