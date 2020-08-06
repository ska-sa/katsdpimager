"""Functions for recording profiling information."""

import time
import collections
import contextlib
from contextvars import ContextVar
import csv
import pathlib
import weakref
import typing
from typing import List, Tuple, Dict, Mapping, Generator, Union, Optional, Any


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

    def stop(self) -> None:
        if self._start_time is None:
            raise RuntimeError(f'Stopwatch {self._frame.name} is already stopped')
        stop_time = time.monotonic()
        record = Record(self._frame, self._start_time, stop_time)
        self._profiler.records.append(record)
        self._start_time = None


class NullStopwatch(Stopwatch):
    """Stopwatch that doesn't do anything."""

    def start(self) -> None:
        if self._start_time is not None:
            raise RuntimeError(f'Stopwatch {self._name} is already running')
        self._start_time = 0.0   # Just for the error checking

    def stop(self) -> None:
        if self._start_time is None:
            raise RuntimeError(f'Stopwatch {self._name} is already stopped')
        self._start_time = None


class Profiler:
    """Top-level class to hold profiling results."""

    def __init__(self) -> None:
        self.records: List[Record] = []

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

    def write_flamegraph(self, filename: Union[str, pathlib.Path]) -> None:
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
            # active), so subtract from parent frames.
            for i in range(1, len(stack) - 1):
                samples[stack[:i]] -= elapsed

        with open(filename, 'w') as f:
            for names, value in samples.items():
                print(';'.join(names), value, file=f)

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


class NullProfiler(Profiler):
    """Implements the :class:`Profiler` interface but does not record anything."""

    @contextlib.contextmanager
    def profile(self, name: str,
                labels: Mapping[str, Any] = {}) -> Generator[Stopwatch, None, None]:
        yield NullStopwatch(self, name, Frame('', {}))
