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

def _key_values_to_dict(items):
    out = {}
    for item in items:
        kv = item.split('=', 1)
        if len(kv) != 2:
            raise ValueError('Missing equals sign in "{}"'.format(item))
        out[kv[0]] = kv[1]
    return out

def write_fits(data, cellsizes, filename):
    header = fits.Header()
    header['BUNIT'] = 'JY/BEAM'
    header['ORIGIN'] = 'katsdpimager'
    # Transformation from pixel coordinates to intermediate world
    # coordinates, which are taken to be l, m coordinates. The reference
    # point is current taken to be the centre of the image.
    # Note that astropy.io.fits reverses the axis order.
    header['CRPIX1'] = data.shape[1] * 0.5 + 0.5
    header['CRPIX2'] = data.shape[0] * 0.5 + 0.5
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
    header['CRVAL1'] = 52.5 # RA of field centre
    header['CRVAL2'] = -35.0 # DEC of field centre

    hdu = fits.PrimaryHDU(data, header)
    hdu.writeto(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    parser.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE', help='Backend-specific input parsing option')
    parser.add_argument('--channel', '-c', type=int, default=0, help='Channel number')

    args = parser.parse_args()
    args.input_option = _key_values_to_dict(args.input_option)

    print("Converting {} to {}".format(args.input_file, args.output_file))
    with closing(katsdpimager.loader.load(args.input_file, args.input_option)) as dataset:
        print(dataset.antenna_diameter())
        print(dataset.longest_baseline())
        for chunk in dataset.data_iter(args.channel, 65536):
            print(chunk['vis'].shape, chunk['weights'].shape, chunk['uvw'].shape)

    image = np.zeros((1024, 1024), dtype=np.float32)
    write_fits(image, (4.8481e-06, 4.8481e-06), args.output_file)

if __name__ == '__main__':
    main()
