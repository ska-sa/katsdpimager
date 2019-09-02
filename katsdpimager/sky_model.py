# -*- coding: utf-8 -*-
"""Load a local sky model from file and transfer it to a model image

At present the only file format supported is a katpoint catalogue file, but
other formats (e.g. Tigger) may be added later.
"""

import logging
import urllib

import numpy as np
from astropy import units
import katpoint


logger = logging.getLogger(__name__)


class NoSkyModelError(Exception):
    """Attempted to load a sky model for continuum subtraction but there isn't one"""
    pass


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
    telstate : :class:`katsdptelstate.TelescopeState` or :class:`katdal.sensordata.TelstateToStr`
        Telescope state
    capture_block_id : str
        Capture block ID
    continuum : str or ``None``
        Name of the continuum imaging stream (used to form the telstate view).
        If ``None``, there must be exactly one continuum imaging stream in the
        data set, which is used.
    target : :class:`katpoint.Target`
        Target field

    Raises
    ------
    NoSkyModelError
        If no sky model could be found for the given parameters

    Returns
    -------
    katpoint.Catalogue
    """
    # katsdpcontim is still Python 2, so it needs TelstateToStr to work with
    # if we are run under Python 3.
    from katdal.sensordata import TelstateToStr

    telstate = TelstateToStr(telstate)
    try:
        # Find the continuum image stream
        if continuum is None:
            archived_streams = telstate['sdp_archived_streams']
            for stream_name in archived_streams:
                view = telstate.view(stream_name, exclusive=True)
                view = view.view(telstate.join(capture_block_id, stream_name))
                stream_type = view.get('stream_type', 'unknown')
                if stream_type != 'sdp.continuum_image':
                    continue
                if continuum is not None:
                    raise NoSkyModelError(
                        'Multiple continuum image streams found - need to select one')
                continuum = stream_name
            if continuum is None:
                raise NoSkyModelError('No continuum image streams found')

        view = telstate.view(continuum, exclusive=True)
        view = view.view(telstate.join(capture_block_id, continuum))
        target_namespace = view['targets'][target.description]
        prefix = telstate.join(capture_block_id, continuum, target_namespace, 'target0')
        data = view.view(prefix)['clean_components']
        # Should always match, but for safety
        if katpoint.Target(data['description']) == target:
            return katpoint.Catalogue(data['components'])
    except KeyError:
        logger.debug('KeyError', exc_info=True)
    raise NoSkyModelError('Sky model for target {} not found'.format(target.name))


def open_sky_model(url):
    """Load a sky model from an external resource.

    The format is encoded in the URL as a `format` query parameter, and
    defaults to ``katpoint``.

    Parameters
    ----------
    url : str
        The interpretation depends on the format.

        katpoint
            A ``file://`` URL for a katpoint catalogue file
        katdal
            A katdal MVFv4 URL. The following extra query parameters are interpreted
            and removed before they are passed to katdal:

            target (required)
                Description of the katpoint target whose LSM should be loaded
            continuum (optional)
                Name of the continuum image stream. If not specified, there
                must be exactly one in the file.

    Raises
    ------
    ValueError
        if `format` was not recognised, the URL doesn't contain the
        expected query parameters, or the URL scheme is not supported
    IOError, OSError
        if there was a low-level error reading a file
    Exception
        any exception raised by katdal in opening the file

    Returns
    -------
    SkyModel
    """
    parts = urllib.parse.urlparse(url, scheme='file')
    params = urllib.parse.parse_qs(parts.query)
    model_format = params.pop('format', ['katpoint'])[0]

    if model_format == 'katdal':
        import katdal
        try:
            target = katpoint.Target(params.pop('target')[0])
        except KeyError:
            raise ValueError('URL must contain target')
        try:
            continuum = params.pop('continuum')[0]
        except KeyError:
            continuum = None
        # Reconstruct the URL without the query components we've absorbed
        new_url = urllib.parse.urlunparse((
            parts.scheme, parts.netloc, parts.path, parts.params,
            urllib.parse.urlencode(params, doseq=True),
            parts.fragment))
        data_source = katdal.open_data_source(new_url, chunk_store=None, upgrade_flags=False)
        catalogue = catalogue_from_telstate(data_source.telstate, data_source.capture_block_id,
                                            continuum, target)
        return KatpointSkyModel(catalogue)
    elif model_format == 'katpoint':
        if parts.scheme != 'file':
            raise ValueError('Only file:// URLs are supported for katpoint sky model format')
        with open(parts.path) as f:
            catalogue = katpoint.Catalogue(f)
        return KatpointSkyModel(catalogue)
    else:
        raise ValueError('format "{}" is not recognised'.format(model_format))
