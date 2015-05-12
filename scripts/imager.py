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
import katsdpimager.grid as grid
import katsdpimager.io as io
from contextlib import closing

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
    group = parser.add_argument_group('Gridding options')
    group.add_argument('--grid-oversample', type=int, default=8, help='Oversampling factor for convolution kernels [%(default)s]')
    group.add_argument('--aa-size', type=int, default=7, help='Support of anti-aliasing kernel [%(default)s]')
    group = parser.add_argument_group('Performance tuning options')
    group.add_argument('--vis-block', type=int, default=1048576, help='Number of visibilities to load at a time [%(default)s]')
    group = parser.add_argument_group('Debugging options')
    group.add_argument('--host', action='store_true', help='Perform gridding on the CPU')
    group.add_argument('--write-grid', metavar='FILE', help='Write UV grid to FITS file')
    return parser

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

    print("Converting {} to {}".format(args.input_file, args.output_file))
    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        polarizations = dataset.polarizations()
        array_p = dataset.array_parameters()
        image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample,
            dataset.frequency(args.channel), array_p, polarizations,
            args.pixel_size, args.pixels)
        grid_p = parameters.GridParameters(args.aa_size, args.grid_oversample)
        if args.host:
            gridder = grid.GridderHost(image_p, grid_p)
            grid_data = np.zeros((image_p.pixels, image_p.pixels, len(polarizations)), dtype=np.complex64)
        else:
            gridder_template = grid.GridderTemplate(context, grid_p, len(polarizations))
            gridder = gridder_template.instantiate(queue, image_p, args.vis_block, accel.SVMAllocator(context))
            gridder.ensure_all_bound()
            grid_data = gridder.buffer('grid')
            grid_data.fill(0)
        for chunk in dataset.data_iter(args.channel, args.vis_block):
            n = len(chunk['uvw'])
            if args.host:
                gridder.grid(grid_data, chunk['uvw'], chunk['weights'], chunk['vis'])
            else:
                gridder.buffer('uvw')[:n] = chunk['uvw'].to(units.m).value
                gridder.buffer('weights')[:n] = chunk['weights']
                gridder.buffer('vis')[:n] = chunk['vis']
                gridder.set_num_vis(n)
                gridder()
                queue.finish()

        image = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(grid_data), axes=(0, 1)).real)
        if args.write_grid is not None:
            io.write_fits_grid(grid_data, image_p, args.write_grid)
        io.write_fits_image(dataset, image, image_p, args.output_file)

if __name__ == '__main__':
    main()
