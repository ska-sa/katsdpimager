"""Render images using matplotlib."""

import numpy as np
from astropy import units
from astropy.wcs import WCS
import astropy.io.fits as fits
import matplotlib
matplotlib.use('Agg')     # noqa: E402
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import katsdpsigproc.zscale as zscale


DEFAULT_DPI = 96


def _prepare_axes(wcs, width, height, image_width, image_height, dpi):
    fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(projection=wcs, slices=('x', 'y', 0, 0))
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_xlim(-0.5, image_width - 0.5)
    ax.set_ylim(-0.5, image_height - 0.5)
    return fig, ax


def _plot(hdus, caption, ax, extent, vmin, vmax):
    bunit = hdus[0].header['BUNIT']
    if bunit == 'JY/BEAM':
        # This was used in older versions of katsdpimager, but is not FITS-standard
        unit = units.Jy / units.beam
    else:
        unit = units.Unit(bunit)
    data = hdus[0].data[0, 0, :, :] << unit
    vmin <<= unit
    vmax <<= unit
    # If the flux is low, use µJy/beam or mJy/beam to keep scale sane
    if vmax < 100 * (units.uJy / units.beam):
        data = data.to(units.uJy / units.beam)
    elif vmax < 100 * (units.mJy / units.beam):
        data = data.to(units.mJy / units.beam)
    vmin = vmin.to(data.unit)
    vmax = vmax.to(data.unit)
    if not ax.images:
        im = ax.imshow(data.value, origin='lower', cmap='afmhot', aspect='equal',
                       vmin=vmin.value, vmax=vmax.value, extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', pad='3%', size='5%', axes_class=matplotlib.axes.Axes)
        cbar = ax.get_figure().colorbar(im, cax=cax, orientation='vertical')
        # simply using data.unit.to_string('unicode') draws an ASCII-art
        # fraction, which doesn't end up looking very good. But we want
        # unicode to properly render µJy (rather than uJy).
        unit_label = (data.unit * units.beam).to_string('unicode') + ' / beam'
        cbar.set_label(unit_label)
    else:
        im = ax.images[0]
        im.set_data(data)
        im.set_extent(extent)

    # Convert an arbitrary pixel to world coordinates to get frequency
    wcs = WCS(hdus[0])
    pix = np.array([[0, 0, 0, 0]])
    world = wcs.all_pix2world(pix, 0)
    freq = world[0][3] * units.Unit(hdus[0].header['CUNIT4'])
    freq = freq.to(units.MHz)
    if caption:
        ax.set_title(f'{caption} ({freq:.3f})')
    else:
        ax.set_title(f'{freq:.3f}')


def write_image(input_file, output_file, width=1024, height=768, dpi=DEFAULT_DPI, caption=None):
    """Write an image showing a single FITS file.

    This code is only designed to work with FITS files written by
    :func:`katsdpimager.io.write_fits_image` (it makes assumptions about axis
    order, units etc).

    Parameters
    ----------
    input_file : str
        Source FITS file
    output_file : str
        Output image file, including extension
    width, height : int
        Dimensions of the output image
    caption : Optional[str]
        Optional caption to include in the image
    """
    with fits.open(input_file) as hdus:
        data = hdus[0].data[0, 0]    # Stokes I, single channel
        vmin, vmax = zscale.zscale(zscale.sample_image(data))
        image_height, image_width = data.shape
        fig, ax = _prepare_axes(WCS(hdus[0]), width, height, image_width, image_height, dpi)
        _plot(hdus, caption, ax, None, vmin, vmax)
        fig.savefig(output_file)


def write_movie(files, output_file, width=1024, height=768, dpi=DEFAULT_DPI, fps=5.0):
    """Write a video with an animation of a set of FITS files.

    This code is only designed to work with FITS files written by
    :func:`katsdpimager.io.write_fits_image` (it makes assumptions about axis
    order, units etc).

    The field of view is determined by the last image. With increasing
    frequency, this is usually the one with the smallest field of view,
    ensuring that the frame is filled in all the images (with pixels
    clipped from the channels with a wider field of view).

    Note that matplotlib currently uses ffmpeg (the program) to write a video,
    so it needs to be installed.

    Parameters
    ----------
    files : Sequence[Tuple[Optional[str], str]]
        Pairs of caption and filename.
    output_file : str
        Output filename, including extension.
    width, height : int
        Nominal dimensions of the video. Due to limitations in matplotlib
        it might not be exact.
    fps : float
        Frames per second in the written video.
    """
    # Load the last image to get its WCS
    with fits.open(files[-1][1]) as hdus:
        common_wcs = WCS(hdus[0])
        image_width = hdus[0].header['NAXIS1']
        image_height = hdus[0].header['NAXIS2']
    # Sample all the images to choose data bounds
    samples = []
    n_samples = 1000000 // len(files) + 1
    for caption, filename in files:
        with fits.open(filename, memmap=True) as hdus:
            s = zscale.sample_image(hdus[0].data[0, 0], max_samples=n_samples, random_offsets=True)
            samples.append(s)
    samples = np.concatenate(samples)
    vmin, vmax = zscale.zscale(samples)
    fig, ax = _prepare_axes(common_wcs, width, height, image_width, image_height, dpi)

    def render_channel(caption_filename):
        caption, filename = caption_filename
        with fits.open(filename) as hdus:
            wcs = WCS(hdus[0])
            data = hdus[0].data[0, 0, :, :]
            # Convert corners of the image to world coordinates
            corners_pix = np.array([
                [-0.5, -0.5, 0, 0],
                [data.shape[1] - 0.5, data.shape[0] - 0.5, 0, 0]
            ])
            corners_world = wcs.all_pix2world(corners_pix, 0)
            # Convert back to pixel coordinates for the plotting WCS
            corners_data = common_wcs.all_world2pix(corners_world, 0)
            extent = [corners_data[0, 0], corners_data[1, 0],
                      corners_data[0, 1], corners_data[1, 1]]
            _plot(hdus, caption, ax, extent, vmin, vmax)

    ani = animation.FuncAnimation(fig, render_channel, files, cache_frame_data=False)
    ani.save(output_file, fps=fps)
