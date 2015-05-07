#!/usr/bin/env python
from __future__ import print_function, division
import math
import argparse
import casacore.tables
import astropy.io.fits as fits
import astropy.wcs as wcs
import numpy as np
import katsdpsigproc
import katsdpimager.loader
from contextlib import closing

def write_fits(dataset, image, cellsizes, filename):
    header = fits.Header()
    header['BUNIT'] = 'JY/BEAM'
    header['ORIGIN'] = 'katsdpimager'
    # Transformation from pixel coordinates to intermediate world
    # coordinates, which are taken to be l, m coordinates. The reference
    # point is current taken to be the centre of the image.
    # Note that astropy.io.fits reverses the axis order.
    header['CRPIX1'] = image.shape[1] * 0.5 + 0.5
    header['CRPIX2'] = image.shape[0] * 0.5 + 0.5
    # FITS uses degrees; and RA increases right-to-left
    header['CDELT1'] = -cellsizes[0] * 180.0 / math.pi
    header['CDELT2'] = cellsizes[1] * 180.0 / math.pi

    # Transformation from intermediate world coordinates to world
    # coordinates (celestial coordinates in this case).
    # TODO: get equinox from input
    header['EQUINOX'] = 2000.0
    header['RADESYS'] = 'FK5' # Julian equinox
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    phase_centre = dataset.phase_centre()
    header['CRVAL1'] = phase_centre[0] * 180.0 / math.pi
    header['CRVAL2'] = phase_centre[1] * 180.0 / math.pi

    hdu = fits.PrimaryHDU(image, header)
    hdu.writeto(filename, clobber=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    parser.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE', help='Backend-specific input parsing option')
    parser.add_argument('--channel', '-c', type=int, default=0, help='Channel number')

    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]

    print("Converting {} to {}".format(args.input_file, args.output_file))
    with closing(katsdpimager.loader.load(args.input_file, args.input_option)) as dataset:
        print(dataset.antenna_diameter())
        print(dataset.longest_baseline())
        for chunk in dataset.data_iter(args.channel, 65536):
            print(chunk['vis'].shape, chunk['weights'].shape, chunk['uvw'].shape)

        image = np.zeros((1024, 1024), dtype=np.float32)
        write_fits(dataset, image, (4.8481e-06, 4.8481e-06), args.output_file)

if __name__ == '__main__':
    main()
