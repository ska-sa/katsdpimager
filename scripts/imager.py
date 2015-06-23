#!/usr/bin/env python
from __future__ import print_function, division
import math
import sys
import argparse
import astropy.units as units
import numpy as np
import katsdpsigproc.accel as accel
import katsdpimager.loader as loader
import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
import katsdpimager.grid as grid
import katsdpimager.io as io
import katsdpimager.fft as fft
import katsdpimager.clean as clean
import katsdpimager.progress as progress
from contextlib import closing, contextmanager

def parse_quantity(str_value):
    """Parse a string into an astropy Quantity. Rather than trying to guess
    where the split occurs, we try every position from the back until we
    succeed."""
    for i in range(len(str_value), -1, 0):
        try:
            value = float(str_value[:i])
            unit = units.Unit(str_value[i:])
            return units.Quantity(value, unit)
        except ValueError:
            pass
    raise ValueError('Could not parse {} as a quantity'.format(str_value))

def parse_stokes(str_value):
    ans = []
    for p in str_value:
        if p not in 'IQUV':
            raise ValueError('Invalid Stokes parameter {}'.format(p))
    if not str_value:
        raise ValueError('Empty Stokes parameter list')
    for p in 'IQUV':
        cnt = str_value.count(p)
        if cnt > 1:
            raise ValueError('Stokes parameter {} listed multiple times'.format(p))
        elif cnt > 0:
            ans.append(polarization.STOKES_NAMES.index(p))
    return sorted(ans)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    group = parser.add_argument_group('Input selection')
    group.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE', help='Backend-specific input parsing option')
    group.add_argument('--channel', '-c', type=int, default=0, help='Channel number [%(default)s]')
    group = parser.add_argument_group('Image options')
    group.add_argument('--q-fov', type=float, default=1.0, help='Field of view to image, relative to main lobe of beam [%(default)s]')
    group.add_argument('--image-oversample', type=float, default=5, help='Pixels per beam [%(default)s]')
    group.add_argument('--pixel-size', type=parse_quantity, help='Size of each image pixel [computed from array]')
    group.add_argument('--pixels', type=int, help='Number of pixels in image [computed from array]')
    group.add_argument('--stokes', type=parse_stokes, default='I', help='Stokes parameters to image e.g. IQUV for full-Stokes [%(default)s]')
    group.add_argument('--precision', choices=['single', 'double'], default='single', help='Internal floating-point precision [%(default)s]')
    group = parser.add_argument_group('Gridding options')
    group.add_argument('--grid-oversample', type=int, default=8, help='Oversampling factor for convolution kernels [%(default)s]')
    group.add_argument('--kernel-image-oversample', type=int, default=4, help='Oversampling factor for kernel generation [%(default)s]')
    group.add_argument('--w-planes', type=int, default=128, help='Number of W planes [%(default)s]'),
    group.add_argument('--max-w', type=parse_quantity, help='Largest w, as either distance or wavelengths [longest baseline]')
    group.add_argument('--aa-size', type=float, default=7, help='Support of anti-aliasing kernel [%(default)s]')
    group = parser.add_argument_group('Cleaning options')
    # TODO: compute from some heuristic if not specified, instead of a hard-coded default
    group.add_argument('--psf-patch', type=int, default=100, help='Pixels in beam patch for cleaning [%(default)s]')
    group.add_argument('--loop-gain', type=float, default=0.1, help='Loop gain for cleaning [%(default)s]')
    group.add_argument('--minor', type=int, default=1000, help='Minor cycles per major cycle [%(default)s]')
    group.add_argument('--clean-mode', choices=['I', 'IQUV'], default='IQUV', help='Stokes parameters to consider for peak-finding [%(default)s]')
    group = parser.add_argument_group('Performance tuning options')
    group.add_argument('--vis-block', type=int, default=1048576, help='Number of visibilities to load at a time [%(default)s]')
    group = parser.add_argument_group('Debugging options')
    group.add_argument('--host', action='store_true', help='Perform gridding on the CPU')
    group.add_argument('--write-psf', metavar='FILE', help='Write image of PSF to FITS file')
    group.add_argument('--write-grid', metavar='FILE', help='Write UV grid to FITS file')
    group.add_argument('--write-dirty', metavar='FILE', help='Write dirty image to FITS file')
    group.add_argument('--write-model', metavar='FILE', help='Write model image to FITS file')
    group.add_argument('--write-residuals', metavar='FILE', help='Write image residuals to FITS file')
    group.add_argument('--vis-limit', type=int, metavar='N', help='Use only the first N visibilities')
    return parser

def data_iter(dataset, args):
    """Wrapper around :py:meth:`katsdpimager.loader_core.LoaderBase.data_iter`
    that handles truncation to a number of visibilities specified on the
    command line.
    """
    N = args.vis_limit
    for chunk in dataset.data_iter(args.channel, args.vis_block):
        if N is not None:
            if N < len(chunk['uvw']):
                for key in ['uvw', 'weights', 'vis']:
                    if key in chunk:
                        chunk[key] = chunk[key][:N]
                chunk['progress'] = chunk['total']
        yield chunk
        if N is not None:
            N -= len(chunk['uvw'])
            if N == 0:
                return

def make_psf(queue, dataset, args, polarization_matrix, gridder, grid_to_image):
    bar = None
    gridder.clear()
    # TODO: pass a flag to avoid retrieving visibilities
    try:
        for chunk in data_iter(dataset, args):
            uvw = chunk['uvw']
            weights = chunk['weights']
            if bar is None:
                bar = progress.make_progressbar("Gridding PSF", max=chunk['total'])
            # Transform the visibilities to the desired polarization
            weights = polarization.apply_polarization_matrix_weights(weights, polarization_matrix)
            gridder.grid(uvw, weights)
            if queue:
                queue.finish()
            bar.goto(chunk['progress'])
    finally:
        bar.finish()

    with progress.step('FFT PSF'):
        grid_to_image()
        if queue:
            queue.finish()

def make_dirty(queue, dataset, args, polarization_matrix, gridder, grid_to_image):
    bar = None
    gridder.clear()
    try:
        for chunk in data_iter(dataset, args):
            uvw = chunk['uvw']
            weights = chunk['weights']
            vis = chunk['vis']
            if bar is None:
                bar = progress.make_progressbar("Gridding", max=chunk['total'])
            # Transform the visibilities to the desired polarization
            vis = polarization.apply_polarization_matrix(vis, polarization_matrix)
            weights = polarization.apply_polarization_matrix_weights(weights, polarization_matrix)
            # Pre-weight the visibilities
            vis *= weights
            gridder.grid(uvw, vis)
            if queue:
                queue.finish()
            bar.goto(chunk['progress'])
    finally:
        bar.finish()

    with progress.step('FFT'):
        grid_to_image()
        if queue:
            queue.finish()

def psf_shape(image_parameters, clean_parameters):
    psf_patch = min(image_parameters.pixels, clean_parameters.psf_patch)
    return (psf_patch, psf_patch, len(image_parameters.polarizations))

def extract_psf(image, psf):
    y0 = (image.shape[0] - psf.shape[0]) // 2
    y1 = y0 + psf.shape[0]
    x0 = (image.shape[1] - psf.shape[1]) // 2
    x1 = y0 + psf.shape[1]
    psf[...] = image[y0:y1, x0:x1, ...]

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]

    queue = None
    context = None
    if not args.host:
        context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
        queue = context.create_command_queue()

    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        #### Determine parameters ####
        input_polarizations = dataset.polarizations()
        output_polarizations = args.stokes
        polarization_matrix = polarization.polarization_matrix(output_polarizations, input_polarizations)
        array_p = dataset.array_parameters()
        image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample,
            dataset.frequency(args.channel), array_p, output_polarizations,
            (np.float32 if args.precision == 'single' else np.float64),
            args.pixel_size, args.pixels)
        if args.max_w is None:
            args.max_w = array_p.longest_baseline
        elif args.max_w.unit.physical_type == 'dimensionless':
            args.max_w = args.max_w * image_p.wavelength
        grid_p = parameters.GridParameters(
            args.aa_size, args.grid_oversample, args.kernel_image_oversample,
            args.w_planes, args.max_w)
        if args.clean_mode == 'I':
            clean_mode = clean.CLEAN_I
        elif args.clean_mode == 'IQUV':
            clean_mode = clean.CLEAN_SUMSQ
        else:
            raise ValueError('Unhandled --clean-mode {}'.format(args.clean_mode))
        clean_p = parameters.CleanParameters(
            args.minor, args.loop_gain, clean_mode,
            args.psf_patch)

        #### Create data and operation instances ####
        if args.host:
            gridder = grid.GridderHost(image_p, grid_p)
            grid_data = gridder.values
            layer = np.empty(grid_data.shape, image_p.complex_dtype)
            image = np.empty(grid_data.shape, image_p.real_dtype)
            model = np.empty(grid_data.shape, image_p.real_dtype)
            psf = np.empty(psf_shape(image_p, clean_p), image_p.real_dtype)
            grid_to_image = fft.GridToImageHost(grid_data, layer, image)
            cleaner = clean.CleanHost(image_p, clean_p, image, psf, model)
            image_scale = np.reciprocal(gridder.taper(image_p.pixels,
                np.empty((image_p.pixels, image_p.pixels), image_p.real_dtype)))
        else:
            allocator = accel.SVMAllocator(context)
            # Gridder
            gridder_template = grid.GridderTemplate(context, image_p, grid_p)
            gridder = gridder_template.instantiate(queue, array_p, args.vis_block, allocator)
            gridder.ensure_all_bound()
            grid_data = gridder.buffer('grid')
            # Grid to image
            image_scale = np.reciprocal(gridder_template.taper(image_p.pixels,
                accel.SVMArray(context, (image_p.pixels, image_p.pixels), image_p.real_dtype)))
            layer = accel.SVMArray(context, grid_data.shape, image_p.complex_dtype)
            image = accel.SVMArray(context, grid_data.shape, image_p.real_dtype)
            grid_to_image_template = fft.GridToImageTemplate(
                queue, grid_data.shape, grid_data.padded_shape, image.shape, image.dtype)
            grid_to_image = grid_to_image_template.instantiate(allocator)
            grid_to_image.bind(grid=grid_data, layer=layer, image=image)
            # CLEAN
            psf = accel.SVMArray(context, psf_shape(image_p, clean_p), image_p.real_dtype)
            model = accel.SVMArray(context, image.shape, image.dtype)
            model[:] = 0
            cleaner_template = clean.CleanTemplate(
                context, clean_p, image_p.real_dtype, len(output_polarizations))
            cleaner = cleaner_template.instantiate(queue, image_p, allocator)
            cleaner.bind(dirty=image, model=model, psf=psf)
            cleaner.ensure_all_bound()

        #### Create dirty image ####
        make_psf(queue, dataset, args, polarization_matrix, gridder, grid_to_image)
        # TODO: all this scaling is hacky. Move it into subroutines somewhere
        image *= image_scale[..., np.newaxis]
        scale = np.reciprocal(image[image.shape[0] // 2, image.shape[1] // 2, ...])
        image *= scale
        if args.write_psf is not None:
            with progress.step('Write PSF'):
                io.write_fits_image(dataset, image, image_p, args.write_psf)
        extract_psf(image, psf)
        make_dirty(queue, dataset, args, polarization_matrix, gridder, grid_to_image)
        image *= image_scale[..., np.newaxis]
        image *= scale
        if args.write_grid is not None:
            with progress.step('Write grid'):
                if args.host:
                    io.write_fits_grid(grid_data, image_p, args.write_grid)
                else:
                    io.write_fits_grid(np.fft.fftshift(grid_data), image_p, args.write_grid)
        if args.write_dirty is not None:
            with progress.step('Write dirty image'):
                io.write_fits_image(dataset, image, image_p, args.write_dirty)

        #### Deconvolution ####
        bar = progress.make_progressbar('CLEAN', max=clean_p.minor)
        cleaner.reset()
        with progress.finishing(bar):
            for i in bar.iter(range(clean_p.minor)):
                cleaner()
        if queue:
            queue.finish()
        # TODO: restoring beam
        if args.write_model is not None:
            with progress.step('Write model'):
                io.write_fits_image(dataset, model, image_p, args.write_model)
        if args.write_residuals is not None:
            with progress.step('Write residuals'):
                io.write_fits_image(dataset, image, image_p, args.write_residuals)
        # Add residuals back in
        model += image
        with progress.step('Write clean image'):
            io.write_fits_image(dataset, model, image_p, args.output_file)

if __name__ == '__main__':
    main()
