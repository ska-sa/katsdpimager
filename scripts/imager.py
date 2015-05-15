#!/usr/bin/env python
from __future__ import print_function, division
import math
import sys
import argparse
import astropy.units as units
import numpy as np
import progress.bar
import katsdpsigproc.accel as accel
import katsdpimager.loader as loader
import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
import katsdpimager.grid as grid
import katsdpimager.io as io
import katsdpimager.fft as fft
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
    group.add_argument('--aa-size', type=int, default=7, help='Support of anti-aliasing kernel [%(default)s]')
    group = parser.add_argument_group('Performance tuning options')
    group.add_argument('--vis-block', type=int, default=1048576, help='Number of visibilities to load at a time [%(default)s]')
    group = parser.add_argument_group('Debugging options')
    group.add_argument('--host', action='store_true', help='Perform gridding on the CPU')
    group.add_argument('--write-grid', metavar='FILE', help='Write UV grid to FITS file')
    return parser

def make_progressbar(name, *args, **kwargs):
    bar = progress.bar.Bar("{:16}".format(name), suffix='%(percent)3d%% [%(eta_td)s]', *args, **kwargs)
    bar.update()
    return bar

@contextmanager
def step(name):
    progress = make_progressbar(name, max=1)
    yield
    progress.next()
    progress.finish()

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]

    if not args.host:
        context = accel.create_some_context()
        if not context.device.is_cuda:
            print("Only CUDA is supported at present. Please select a CUDA device or use --host.", file=sys.stderr)
            sys.exit(1)
        queue = context.create_command_queue()

    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        input_polarizations = dataset.polarizations()
        output_polarizations = args.stokes
        polarization_matrix = polarization.polarization_matrix(output_polarizations, input_polarizations)
        array_p = dataset.array_parameters()
        image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample,
            dataset.frequency(args.channel), array_p, output_polarizations,
            (np.float32 if args.precision == 'single' else np.float64),
            args.pixel_size, args.pixels)
        grid_p = parameters.GridParameters(args.aa_size, args.grid_oversample)
        if args.host:
            gridder = grid.GridderHost(image_p, grid_p)
            grid_data = np.zeros((image_p.pixels, image_p.pixels, len(output_polarizations)), dtype=image_p.dtype_complex)
        else:
            gridder_template = grid.GridderTemplate(context, grid_p, len(output_polarizations), image_p.dtype_complex)
            gridder = gridder_template.instantiate(queue, image_p, array_p, args.vis_block, accel.SVMAllocator(context))
            gridder.ensure_all_bound()
            grid_data = gridder.buffer('grid')
            grid_data.fill(0)
        progress = None
        for chunk in dataset.data_iter(args.channel, args.vis_block):
            uvw = chunk['uvw']
            weights = chunk['weights']
            vis = chunk['vis']
            if progress is None:
                progress = make_progressbar("Gridding", max=chunk['total'])
            n = len(uvw)
            # Transform the visibilities to the desired polarization
            vis, weights = polarization.apply_polarization_matrix_weighted(
                vis, weights, polarization_matrix)
            # Pre-weight the visibilities
            vis *= weights
            if args.host:
                gridder.grid(grid_data, uvw, vis)
            else:
                gridder.buffer('uvw')[:n] = uvw.to(units.m).value
                gridder.buffer('vis')[:n] = vis
                gridder.set_num_vis(n)
                gridder()
                queue.finish()
            progress.goto(chunk['progress'])
        progress.finish()

        with step('FFT'):
            if args.host:
                image = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(grid_data), axes=(0, 1)).real)
            else:
                image = accel.SVMArray(context, grid_data.shape, image_p.dtype_complex)
                invert_template = fft.GridToImageTemplate(
                    queue, grid_data.shape, grid_data.padded_shape, image.shape, image.dtype)
                invert = invert_template.instantiate(accel.SVMAllocator(context))
                invert.bind(grid=grid_data, image=image)
                invert()
                queue.finish()
                image = image.real * np.reciprocal(image_p.dtype.type(image.shape[0] * image.shape[1]))
        if args.write_grid is not None:
            with step('Write grid'):
                io.write_fits_grid(grid_data, image_p, args.write_grid)
        with step('Write image'):
            io.write_fits_image(dataset, image, image_p, args.output_file)

if __name__ == '__main__':
    main()
