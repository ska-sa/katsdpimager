"""Data loading frontend"""

from __future__ import print_function, division
import warnings


_loader_classes = []


def _register_loader(loader_class):
    _loader_classes.append(loader_class)


def load(filename, options):
    for cls in _loader_classes:
        if cls.match(filename):
            return cls(filename, options)
    raise ValueError('No loader class is registered for "{}"'.format(filename))


try:
    import katsdpimager.loader_ms
    _register_loader(katsdpimager.loader_ms.LoaderMS)
except ImportError:
    warnings.warn("Failed to load loader_ms. Possibly python-casacore is missing or broken.",
                  ImportWarning)

try:
    import katsdpimager.loader_katdal
    _register_loader(katsdpimager.loader_katdal.LoaderKatdal)
except ImportError:
    warnings.warn("Failed to load loader_katdal. Possibly katdal is missing or broken.",
                  ImportWarning)
