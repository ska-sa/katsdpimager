"""Functions for recording profiling information."""

import time
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
    """Top-level class to hold profiling results.

    Labels that will be applied to all records can be set in
    :attr:`default_labels`; the :func:`labels` context manager is a better
    alternative in most cases though.
    """

    _current: ContextVar['Profiler'] = ContextVar('Profiler._current')

    def __init__(self) -> None:
        self.records: List[Record] = []
        self.default_labels: Dict[str, Any] = {}

    @contextlib.contextmanager
    def profile(self, name: str, **kwargs) -> Generator[Stopwatch, None, None]:
        labels = {**self.default_labels, **kwargs}
        with Stopwatch(self, name, labels) as stopwatch:
            yield stopwatch

    @contextlib.contextmanager
    def labels(self, **kwargs) -> Generator[None, None, None]:
        old_labels = self.default_labels
        self.default_labels.update(kwargs)
        yield
        self.default_labels = old_labels

    def write(self, filename: Union[str, pathlib.Path]) -> None:
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

    @staticmethod
    def get_profiler() -> 'Profiler':
        return Profiler._current.get()

    @staticmethod
    def set_profiler(profiler: 'Profiler') -> None:
        Profiler._current.set(profiler)


# Create a default profiler that any context should inherit
Profiler._current.set(Profiler())


@contextlib.contextmanager
def profile(name: str, **kwargs) -> Generator[Stopwatch, None, None]:
    with Profiler.get_profiler().profile(name, **kwargs) as stopwatch:
        yield stopwatch


class NullProfiler(Profiler):
    """Implements the :class:`Profiler` interface but does not record anything."""

    @contextlib.contextmanager
    def profile(self, name: str, **kwargs) -> Generator[Stopwatch, None, None]:
        yield NullStopwatch(self, name, {})
