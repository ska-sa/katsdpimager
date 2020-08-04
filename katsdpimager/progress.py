"""Progress bar functions.

This wraps the :py:mod:`progress` module to add certain enhancements:

- A helper is provided for steps where no partial progress will be reported, but
  goes from empty to full in one step.
- A fallback implementation is used when stderr is not a TTY.
"""

import sys
from contextlib import contextmanager

import progress
import progress.bar


class NoTtyMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._written_length = 0

    def update(self):
        filled_length = int(self.width * self.progress)
        message = self.message % self
        bar = self.fill * filled_length
        line = ''.join([message, self.bar_prefix, bar])
        if len(line) > self._written_length:
            self.file.write(line[self._written_length:])
            self.file.flush()
            self._written_length = len(line)

    def finish(self):
        filled_length = int(self.width * self.progress)
        empty_length = self.width - filled_length
        empty = self.empty_fill * empty_length
        suffix = self.suffix % self
        self.file.write(''.join([empty, self.bar_suffix, suffix, '\n']))
        self.file.flush()


class NoTtyBar(NoTtyMixin, progress.bar.Bar):
    pass


def make_progressbar(name, *args, **kwargs):
    if sys.stderr.isatty():
        bar = progress.bar.Bar("{:20}".format(name),
                               suffix='%(percent)3d%% [%(eta_td)s]',
                               *args, **kwargs)
    else:
        bar = NoTtyBar("{:20}".format(name), suffix='', *args, **kwargs)
    bar.update()
    return bar


@contextmanager
def step(name):
    bar = make_progressbar(name, max=1)
    with bar:
        yield
        bar.next()
