# -*- coding: utf-8 -*-
"""Load a local sky model from file and transfer it to a model image

At present the only file format supported is a katpoint catalogue file, but
other formats (e.g. Tigger) may be added later.
"""

import itertools
import logging
import urllib

import numpy as np
import astropy.units as units
import katpoint


logger = logging.getLogger(__name__)


class SkyModel:
    """A sky model which can be used to generate images at specified frequencies.

    This is an abstract base class.
    """
    def __len__(self):
        """Get number of sources in catalogue"""
        raise NotImplementedError         # pragma: nocover

    @units.quantity_input(phase_centre=units.rad)
    def lmn(self, phase_centre):
        """Get direction cosine coordinates of the sources

        Parameters
        ----------
        phase_centre : Quantity
            RA and DEC of phase centre

        Returns
        -------
        array
            The l/m/n coordinates, shape N×3
        """
        raise NotImplementedError         # pragma: nocover

    @units.quantity_input(wavelength=units.m, equivalencies=units.spectral())
    def flux_density(self, wavelength):
        """Get flux densities for the sources at the given wavelength/frequency.

        Parameters
        ----------
        wavelength : Quantity
            Wavelength or frequency

        Returns
        -------
        array
            Flux densities in Jy, shape N×4 for Stokes IQUV
        """
        raise NotImplementedError         # pragma: nocover


class KatpointSkyModel(SkyModel):
    """Sky model created from a :class:`katpoint.Catalogue`."""

    def __init__(self, catalogue):
        self._catalogue = catalogue
        positions = units.Quantity(np.empty((len(catalogue), 2)),
                                   unit=units.deg)
        for i, source in enumerate(catalogue):
            positions[i] = units.Quantity(source.astrometric_radec(), unit=units.rad)
        self._positions = positions

    def __len__(self):
        return len(self._catalogue)

    @units.quantity_input(phase_centre=units.rad)
    def lmn(self, phase_centre):
        phase_centre = phase_centre.to(units.rad).value
        phase_centre = katpoint.construct_radec_target(phase_centre[0], phase_centre[1])
        return np.array([phase_centre.lmn(*source.astrometric_radec())
                         for source in self._catalogue])

    @units.quantity_input(wavelength=units.m, equivalencies=units.spectral())
    def flux_density(self, wavelength):
        freq_MHz = wavelength.to(units.MHz, equivalencies=units.spectral()).value
        out = np.stack([source.flux_density_stokes(freq_MHz) for source in self._catalogue])
        return np.nan_to_num(out, copy=False)


def catalogue_from_telstate(telstate, capture_block_id, continuum, target):
    """Extract a katpoint catalogue written by katsdpcontim.

    Parameters
    ----------
    capture_block_id : str
        Capture block ID
    continuum : str
        Name of the continuum imaging task (used to form the telstate view).
    target : :class:`katpoint.Target`
        Target field

    Returns
    -------
    katpoint.Catalogue
    """
    # katsdpcontim is still Python 2, so it needs TelstateToStr to work with
    # if we are run under Python 3.
    from katdal.sensordata import TelstateToStr

    prefix = telstate.join(capture_block_id, continuum)
    view = TelstateToStr(telstate.view(prefix))
    for i in itertools.count():
        key = 'target{}_clean_components'.format(i)
        data = view.get(key)
        if data is None:
            break
        try:
            if katpoint.Target(data['description']) == target:
                return katpoint.Catalogue(data['components'])
        except KeyError:
            logger.warning('Failed to access %s from telescope state',
                           telstate.join(prefix, key), exc_info=True)
    logger.warning('Sky model for target %s not found', target.name)
    return katpoint.Catalogue()


def open_sky_model(url):
    """Load a sky model from an external resource.

    The format is encoded in the URL as a `format` query parameter, and
    defaults to ``katpoint``.

    Parameters
    ----------
    url : str
        Either a catalogue file to load (only local files are currently
        supported) or an URL of the form
        :samp:`redis://{host}:{port}/?format=katpoint&capture_block_id={id}&continuum={name}&target={description}`,
        which will read clean components written by katsdpcontim.

    Raises
    ------
    ValueError
        if `format` was not recognised, the URL doesn't contain the
        expected query parameters, or the URL scheme is not supported
    IOError, OSError
        if there was a low-level error reading a file
    redis.RedisError
        if a ``redis://`` url was used and there was an error accessing redis
    katsdptelstate.telescope_state.TelstateError
        if a ``redis://`` url was used and there was an error accessing the telescope state

    Returns
    -------
    SkyModel
    """
    parts = urllib.parse.urlparse(url, scheme='file')
    params = urllib.parse.parse_qs(parts.query)
    model_format = params.pop('format', ['katpoint'])[0]
    if model_format != 'katpoint':
        raise ValueError('format "{}" is not recognised'.format(model_format))

    if parts.scheme == 'file':
        with open(parts.path) as f:
            catalogue = katpoint.Catalogue(f)
        return KatpointSkyModel(catalogue)
    elif parts.scheme == 'redis':
        import katsdptelstate
        try:
            capture_block_id = params.pop('capture_block_id')[0]
            continuum = params.pop('continuum')[0]
            target = katpoint.Target(params.pop('target')[0])
        except KeyError:
            raise ValueError('URL must contain capture_block_id, continuum and target')
        # Reconstruct the URL without the query components we've absorbed
        redis_url = urllib.parse.urlunparse((
            parts.scheme, parts.netloc, parts.path, parts.params,
            urllib.parse.urlencode(params, doseq=True),
            parts.fragment))
        telstate = katsdptelstate.TelescopeState(redis_url)
        catalogue = catalogue_from_telstate(telstate, capture_block_id, continuum, target)
        return KatpointSkyModel(catalogue)
    else:
        raise ValueError('Unsupported URL scheme {}'.format(parts.scheme))
