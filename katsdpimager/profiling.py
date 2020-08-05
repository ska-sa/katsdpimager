"""Functions for recording profiling information."""

import time
import collections
import contextlib
from contextvars import ContextVar
import csv
import pathlib
from typing import List, Dict, Mapping, Generator, Union, Optional, Any


class Record:
    """A record of some process with a start and a stop time.

    Records may have arbitrary key/value labels associated to assist with
    aggregation. The following labels are standardised:

    - channel
    - start_channel, stop_channel

    The start and stop times are as returned by :func:`time.monotonic`.

    Label values must be CSV-compatible e.g. strings, ints and floats.
    """

    def __init__(self, name: str, start_time: float, stop_time: float,
                 labels: Mapping[str, Any] = {}) -> None:
        self.name = name
        self.start_time = start_time
        self.stop_time = stop_time
        self.labels = dict(labels)

    @property
    def elapsed(self) -> float:
        return self.stop_time - self.start_time

    def __eq__(self, other: object) -> bool:
        if type(other) == Record:
            return vars(self) == vars(other)
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.name, self.start_time, self.stop_time,
                     tuple(sorted(self.labels.items()))))


class Stopwatch(contextlib.ContextDecorator):
    """Measures intervals of time and add records of them to a profiler.

    Each interval is bracketed by calls to :meth:`start` and :meth:`stop`.
    It can also be (and is designed to be) used as a context manager, in
    which case it will start on entry and stop (if necessary) on exit.
    Within the context manager it is possible to use stop/start to pause
    the stopwatch.
    """

    def __init__(self, profiler: 'Profiler', name: str, labels: Mapping[str, Any]):
        self._profiler = profiler
        self._name = name
        self._labels = labels
        self._start_time: Optional[float] = None

    def __enter__(self) -> 'Stopwatch':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        if self._start_time is not None:
            self.stop()

    def start(self) -> None:
        if self._start_time is not None:
            raise RuntimeError(f'Stopwatch {self._name} is already running')
        self._start_time = time.monotonic()

    def stop(self) -> None:
        if self._start_time is None:
            raise RuntimeError(f'Stopwatch {self._name} is already stopped')
        stop_time = time.monotonic()
        record = Record(self._name, self._start_time, stop_time, self._labels)
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
    def profile(self, name: str, **kwargs) -> Generator[Stopwatch, None, None]:
        """Context manager that runs code under a :class:`Stopwatch`."""
        default_labels = _default_labels.get({})
        labels = {**default_labels, **kwargs}
        with Stopwatch(self, name, labels) as stopwatch:
            yield stopwatch

    def write_csv(self, filename: Union[str, pathlib.Path]) -> None:
        labelset = set()
        for record in self.records:
            labelset |= set(record.labels.keys())
        labels = sorted(labelset)
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, ['name', 'start', 'stop', 'elapsed'] + labels)
            writer.writeheader()
            for record in self.records:
                row = {
                    'name': record.name,
                    'start': record.start_time,
                    'stop': record.stop_time,
                    'elapsed': record.elapsed,
                    **record.labels
                }
                writer.writerow(row)

    def write_flamegraph(self, filename: Union[str, pathlib.Path]) -> None:
        """Writes data in a form that can be converted to a flame graph.

        Specifically, use https://github.com/brendangregg/FlameGraph to
        post-process the output.

        Labels are discarded.

        TODO: The implementation currently assumes that all events come from
        the same context and that time strictly increases between events.
        Stack frames should instead be explicitly maintained as events are
        collected.
        """
        events = []
        for record in self.records:
            events.append((record.start_time, True, record))
            events.append((record.stop_time, False, record))
        events.sort(key=lambda event: event[0])
        stack = []
        last_time = None
        samples = collections.Counter()
        for new_time, start, record in events:
            if last_time is not None:
                stack_names = tuple(record.name for record in stack)
                samples[stack_names] += int(1000000 * (new_time - last_time))
            if start:
                stack.append(record)
            else:
                if not stack or stack[-1] is not record:
                    raise ValueError('Records are not correctly nested')
                stack.pop()
            last_time = new_time
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
_default_labels: ContextVar[Dict[str, Any]] = ContextVar('_default_labels')


@contextlib.contextmanager
def profile(name: str, **kwargs) -> Generator[Stopwatch, None, None]:
    """Context manager that runs code under a :class:`Stopwatch`.

    See also
    --------
    :meth:`Profiler.profile`
    """
    with Profiler.get_profiler().profile(name, **kwargs) as stopwatch:
        yield stopwatch


@contextlib.contextmanager
def labels(**kwargs) -> Generator[None, None, None]:
    """Context manager that sets and restores labels."""
    old_labels = _default_labels.get({})
    new_labels = {**old_labels, **kwargs}
    token = _default_labels.set(new_labels)
    try:
        yield
    finally:
        _default_labels.reset(token)


class NullProfiler(Profiler):
    """Implements the :class:`Profiler` interface but does not record anything."""

    @contextlib.contextmanager
    def profile(self, name: str, **kwargs) -> Generator[Stopwatch, None, None]:
        yield NullStopwatch(self, name, {})
