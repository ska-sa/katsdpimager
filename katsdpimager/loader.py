"""Data loading frontend"""

import warnings


_loader_classes = []


def _register_loader(loader_class):
    _loader_classes.append(loader_class)


def load(filename, options, start_channel=0, stop_channel=None):
    """Load a dataset.

    The class is automatically determined from the registered loaders.

    Parameters
    ----------
    filename : str
        Name of the data set to load (does not necessarily have to be a filename
        e.g., it could be an URL).
    options : Dict[str, Any]
        Loader-specific command-line options.
    start_channel,stop_channel : int, optional
        Channel range of interest. This serves as a hint only, to optimise
        loading, and does not change the channel numbers used to interact with
        the data set.
    """
    for cls in _loader_classes:
        if cls.match(filename):
            return cls(filename, options, start_channel, stop_channel)
    raise ValueError('No loader class is registered for "{}"'.format(filename))


def data_iter(dataset, vis_limit, vis_load, start_channel, stop_channel):
    """Wrapper around :py:meth:`katsdpimager.loader_core.LoaderBase.data_iter`
    that handles truncation to a number of visibilities specified on the
    command line.
    """
    N = vis_limit
    it = dataset.data_iter(start_channel, stop_channel, vis_load)
    for chunk in it:
        if N is not None:
            if N < len(chunk['uvw']):
                for key in ['uvw', 'baselines']:
                    if key in chunk:
                        chunk[key] = chunk[key][:N]
                for key in ['weights', 'vis']:
                    if key in chunk:
                        chunk[key] = chunk[key][:, :N]
                chunk['progress'] = chunk['total']
        yield chunk
        if N is not None:
            N -= len(chunk['uvw'])
            if N == 0:
                it.close()
                return


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
