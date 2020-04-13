#!/usr/bin/env python3

"""Convert a beam model from katholog format to that used by :class:`.TrivialBeamModel`."""

import argparse
import math

import numpy as np
import scipy.ndimage
import h5py
import astropy.units as units


def geometric_transform(input, mapping, output):
    scipy.ndimage.geometric_transform(input.real, mapping,
                                      output_shape=output.shape, output=output.real)
    scipy.ndimage.geometric_transform(input.imag, mapping,
                                      output_shape=output.shape, output=output.imag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--angles', type=int, default=64,
                        help='Number of angular samples to use for circularisation')
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    # Axis 1 is antenna, with last element being array average
    beam = data['beam'][:, -1]
    frequencies = data['freqMHz'] * units.MHz
    coords = data['margin']
    # Check that the samples are equally spaced, so that interpolation based
    # on scipy.ndimage can be used.
    coords_step = np.diff(coords)
    if abs(min(coords_step) / max(coords_step) - 1) > 1e-4:
        raise RuntimeError('Input coordinates are not equally spaced')
    step = np.average(coords_step)
    n = int(min(-coords[0], coords[-1]) / step) + 1

    def transform(output_coords):
        angle = 2 * math.pi * output_coords[1] / args.angles
        radius = output_coords[0] * step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return (x - coords[0]) / step, (y - coords[0]) / step

    # Interpolate each polarization and frequency separately.
    # Also have to treat real and imaginary separately because scipy.ndimage
    # doesn't support complex.
    interp = np.zeros(beam.shape[:2] + (n, args.angles), np.complex64)
    for i in range(beam.shape[0]):
        for j in range(beam.shape[1]):
            geometric_transform(beam[i, j], transform, interp[i, j])

    # Compute Stokes I power and average over angles
    power = np.sum(np.square(np.abs(interp)), axis=(0, 3)) * (0.5 / args.angles)
    # Convert back to an equivalent per-antenna real-valued voltage.
    voltage = np.sqrt(power)

    with h5py.File(args.output, 'w') as outf:
        outf.create_dataset('beam', data=voltage)
        outf.create_dataset('frequencies', data=frequencies.to_value(units.Hz))
        outf['beam'].attrs['step'] = step


if __name__ == '__main__':
    main()
