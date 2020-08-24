"""Functions for recording profiling information."""

import time
import collections.abc
import contextlib
from contextvars import ContextVar
import csv
import functools
import inspect
import io
import pathlib
import weakref
import typing
from typing import (
    List, Tuple, Dict, Mapping, Sequence, Callable, TextIO,
    Generator, Union, Optional, TypeVar, Any
)

import katsdpsigproc.abc

from . import nvtx


_T = TypeVar('_T')
_LabelSpec = Union[Sequence[str],
                   Mapping[str, Union[Callable[[inspect.BoundArguments], Any], str]]]
_domain = nvtx.Domain(__name__)


class Frame:
    """Stack frame.

    Where possible, identical frames are represented by the same object to save
    memory.

    Stack frames may have arbitrary key/value labels associated to assist with
    aggregation. Label values must be CSV-compatible e.g. strings, ints and
    floats. The following labels are standardised:

    - channel
    - start_channel, stop_channel
    """

    name: str
    labels: Dict[str, Any]
    parent: Optional['Frame']
    _name_handle: nvtx.RegisteredString
    _hash: int
    _children: weakref.WeakValueDictionary

    def __new__(cls, name, labels: Mapping[str, Any] = {},
                parent: Optional['Frame'] = None) -> 'Frame':
        labels = dict(labels)
        key = (name, tuple(sorted(labels.items())))
        if parent is not None:
            frame = parent._children.get(key)
            if frame is not None:
                return frame
        frame = super().__new__(cls)
        frame.name = name
        frame.labels = labels
        frame.parent = parent
        frame._name_handle = nvtx.register_string(name, domain=_domain)
        frame._hash = hash((key, parent))
        frame._children = weakref.WeakValueDictionary()
        if parent is not None:
            parent._children[key] = frame
        return frame

    def __eq__(self, other: object) -> bool:
        if type(other) != Frame:
            return NotImplemented
        assert isinstance(other, Frame)    # To keep mypy happy
        if self._hash != other._hash:
            return False
        return (self.name == other.name and self.labels == other.labels
                and self.parent == other.parent)

    def __hash__(self) -> int:
        return self._hash

    def stack(self) -> Generator['Frame', None, None]:
        """Iterate over the frames in the stack, from root to leaf."""
        if self.parent is not None:
            yield from self.parent.stack()
        yield self

    def all_labels(self) -> Mapping[str, Any]:
        """Collapse all labels in the stack to a single dictionary.

        Labels closer to the top of the stack update those below.
        """
        labels = {}
        for frame in self.stack():
            labels.update(frame.labels)
        return labels

    def __repr__(self) -> str:
        if self.parent is None:
            return f'Frame({self.name!r}, {self.labels!r})'
        else:
            return f'Frame({self.name!r}, {self.labels!r}, {self.parent!r})'


class Record:
    """A record of some process with a start and a stop time.

    The start and stop times are as returned by :func:`time.monotonic`.
    """

    def __init__(self, frame: Frame, start_time: float, stop_time: float) -> None:
        self.frame = frame
        self.start_time = start_time
        self.stop_time = stop_time

    @property
    def elapsed(self) -> float:
        return self.stop_time - self.start_time

    def __eq__(self, other: object) -> bool:
        if type(other) == Record:
            return vars(self) == vars(other)
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.frame, self.start_time, self.stop_time))

    def __repr__(self) -> str:
        return f'Record({self.frame!r}, {self.start_time!r}, {self.stop_time!r})'


class DeviceRecord:
    """A record of some on-device process with a start and a stop event."""

    def __init__(self, frame: Frame,
                 start_event: katsdpsigproc.abc.AbstractEvent,
                 stop_event: katsdpsigproc.abc.AbstractEvent) -> None:
        self.frame = frame
        self.start_event = start_event
        self.stop_event = stop_event

    @property
    def elapsed(self) -> float:
        """Get elapsed time.

        Note that this may block until the stop event completes.
        """
        return self.stop_event.time_since(self.start_event)


class Stopwatch(contextlib.ContextDecorator):
    """Measures intervals of time and add records of them to a profiler.

    Each interval is bracketed by calls to :meth:`start` and :meth:`stop`.
    It can also be (and is designed to be) used as a context manager, in
    which case it will start on entry and stop (if necessary) on exit.
    Within the context manager it is possible to use stop/start to pause
    the stopwatch.
    """

    def __init__(self, profiler: 'Profiler', frame: Frame):
        self._profiler = profiler
        self._frame = frame
        self._start_time: Optional[float] = None
        self._nvtx_ctx = None

    def __enter__(self) -> 'Stopwatch':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        if self._start_time is not None:
            self.stop()

    def start(self) -> None:
        if self._start_time is not None:
            raise RuntimeError(f'Stopwatch for {self._frame.name} is already running')
        self._start_time = time.monotonic()
        self._nvtx_range = nvtx.thread_range(self._frame._name_handle, domain=_domain)
        self._nvtx_range.__enter__()

    def stop(self) -> None:
        if self._start_time is None:
            return
        self._nvtx_range.__exit__(None, None, None)
        stop_time = time.monotonic()
        record = Record(self._frame, self._start_time, stop_time)
        self._profiler.records.append(record)
        self._start_time = None


class NullStopwatch(Stopwatch):
    """Stopwatch that doesn't do anything."""

    def start(self) -> None:
        if self._start_time is not None:
            raise RuntimeError(f'Stopwatch {self._frame.name} is already running')
        self._start_time = 0.0   # Just for the error checking

    def stop(self) -> None:
        self._start_time = None


class Profiler:
    """Top-level class to hold profiling results."""

    def __init__(self) -> None:
        self.records: List[Record] = []
        self.device_records: List[DeviceRecord] = []

    @contextlib.contextmanager
    def profile(self, name: str,
                labels: Mapping[str, Any] = {}) -> Generator[Stopwatch, None, None]:
        """Context manager that runs code under a :class:`Stopwatch` with a new Frame."""
        frame = Frame(name, labels, _current_frame.get())
        token = _current_frame.set(frame)
        try:
            with Stopwatch(self, frame) as stopwatch:
                yield stopwatch
        finally:
            _current_frame.reset(token)

    @contextlib.contextmanager
    def profile_device(self, queue: katsdpsigproc.abc.AbstractCommandQueue,
                       name: str, labels: Mapping[str, Any] = {}):
        frame = Frame(name, labels, _current_frame.get())
        start_event = queue.enqueue_marker()
        yield
        stop_event = queue.enqueue_marker()
        self.device_records.append(DeviceRecord(frame, start_event, stop_event))

    def write_csv(self, filename: Union[str, pathlib.Path]) -> None:
        labelset = set()
        for record in self.records:
            for frame in record.frame.stack():
                labelset |= set(frame.labels)
        labels = sorted(labelset)
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, ['name', 'start', 'stop', 'elapsed'] + labels)
            writer.writeheader()
            for record in self.records:
                row = {
                    'name': record.frame.name,
                    'start': record.start_time,
                    'stop': record.stop_time,
                    'elapsed': record.elapsed,
                    **record.frame.all_labels()
                }
                writer.writerow(row)

    def write_flamegraph(self, file: Union[TextIO, io.TextIOBase]) -> None:
        """Writes data in a form that can be converted to a flame graph.

        Specifically, use https://github.com/brendangregg/FlameGraph to
        post-process the output.

        Labels are discarded.
        """
        samples: typing.Counter[Tuple[str, ...]] = collections.Counter()
        for record in self.records:
            stack = tuple(frame.name for frame in record.frame.stack())
            # flamegraph.pl wants integers, so convert to microseconds
            elapsed = round(1000000 * record.elapsed)
            samples[stack] += elapsed
            # We need to produce exclusive counts (time with no child frames
            # active), so subtract from parent frame.
            if len(stack) >= 2:
                samples[stack[:-1]] -= elapsed

        for names, value in samples.items():
            print(';'.join(names), value, file=file)

    def write_device_flamegraph(self, file: Union[TextIO, io.TextIOBase]) -> None:
        """Writes device records in a form that can be converted to a flame graph.

        See :meth:`write_flamegraph` for details. The full stack frames are
        emitted, but only time spent in device functions is shown. The
        flamegraph is only meaningful if kernels do not execute concurrently.
        """
        samples: typing.Counter[Tuple[str, ...]] = collections.Counter()
        for record in self.device_records:
            stack = tuple(frame.name for frame in record.frame.stack())
            # flamegraph.pl wants integers, so convert to microseconds
            elapsed = round(1000000 * record.elapsed)
            samples[stack] += elapsed

        for names, value in samples.items():
            print(';'.join(names), value, file=file)

    @staticmethod
    def get_profiler() -> 'Profiler':
        return _current_profiler.get()

    @staticmethod
    def set_profiler(profiler: 'Profiler') -> None:
        _current_profiler.set(profiler)


# Create a default profiler that any context should inherit
_current_profiler: ContextVar[Profiler] = ContextVar('_current_profiler', default=Profiler())
_current_frame: ContextVar[Optional[Frame]] = ContextVar('_current_frame', default=None)


@contextlib.contextmanager
def profile(name: str, labels: Mapping[str, Any] = {}) -> Generator[Stopwatch, None, None]:
    """Context manager that runs code under a :class:`Stopwatch`.

    See also
    --------
    :meth:`Profiler.profile`
    """
    with Profiler.get_profiler().profile(name, labels) as stopwatch:
        yield stopwatch


def _extract_labels(sig: inspect.Signature, labels: _LabelSpec, *args, **kwargs):
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    result = {}
    if isinstance(labels, collections.abc.Mapping):
        for label, spec in labels.items():
            if callable(spec):
                result[label] = spec(bound)
            else:
                result[label] = bound.arguments[spec]
    else:
        result = {label: bound.arguments[label] for label in labels}
    return result


def profile_function(name: Optional[str] = None, labels: _LabelSpec = ()) \
        -> Callable[[Callable], Callable]:
    """Decorator for profiling functions.

    It defaults to using the name of the function as the name for the frame,
    and the labels are extracted from the arguments passed to the function.
    The labels can either be a sequence or a mapping. If a sequence, each
    element names a function argument to be used as a label. If a mapping,
    the keys define the labels. Each values can either be the name of a
    function argument, or a callable that takes
    :class:`inspect.BoundArguments` and returns the value for the label.
    """
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        try:
            func_name = f'{func.__module__}.{func.__qualname__}'
        except AttributeError:
            func_name = '<anonymous function>'
        frame_name = name if name is not None else func_name

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label_dict = _extract_labels(sig, labels, *args, **kwargs)
            with profile(frame_name, label_dict):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def profile_generator(name: Optional[str] = None, labels: _LabelSpec = ()) \
        -> Callable[[Callable[..., Generator[_T, None, None]]],
                    Callable[..., Generator[_T, None, None]]]:
    """Decorator for profiling generator functions.

    It functions similarly to :func:`profile_function`.

    It is only suitable for simple generators intended to be used as
    iterables. It does not support sending values or exceptions into the
    wrapped generator.

    The stopwatch is stopped when the generator yields and restarted when it
    resumes.
    """

    labels = list(labels)

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        try:
            func_name = f'{func.__module__}.{func.__qualname__}'
        except AttributeError:
            func_name = '<anonymous function>'
        frame_name = name if name is not None else func_name

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label_dict = _extract_labels(sig, labels, *args, **kwargs)
            gen = func(*args, **kwargs)
            try:
                with profile(frame_name, label_dict) as stopwatch:
                    for item in gen:
                        stopwatch.stop()
                        yield item
                        stopwatch.start()
            finally:
                gen.close()

        return wrapper
    return decorator


class NullProfiler(Profiler):
    """Implements the :class:`Profiler` interface but does not record anything."""

    @contextlib.contextmanager
    def profile(self, name: str,
                labels: Mapping[str, Any] = {}) -> Generator[Stopwatch, None, None]:
        yield NullStopwatch(self, Frame(name, {}))


@contextlib.contextmanager
def profile_device(queue: katsdpsigproc.abc.AbstractCommandQueue,
                   name: str, labels: Mapping[str, Any] = {}):
    with Profiler.get_profiler().profile_device(queue, name, labels):
        yield
