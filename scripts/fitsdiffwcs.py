#!/usr/bin/env python

"""Compare two FITS images, applying WCS transformations so that comparisons
are at points with the same world coordinates rather than blindly matching
pixels.

At present, only single-channel images and only the primary HDU are supported.

The operation is asymmetric: for each pixel in the first file, we look up the
corresponding pixel in the second file, interpolating as necessary, or
reporting NaN if it falls outside the footprint.
"""

import argparse

import numpy as np
import scipy.interpolate
import astropy.io.fits as fits
import astropy.wcs as wcs


class FitsFile:
    def __init__(self, filename):
        self.hdulist = fits.open(filename)
        self.xform = wcs.WCS(self.hdulist[0].header, self.hdulist)
        self.data = self.hdulist[0].data
        self.axismap = [None, None, None]   # RA, DEC, Stokes
        for i, t in enumerate(self.xform.get_axis_types()):
            if t['coordinate_type'] == 'celestial' and t['number'] in [0, 1]:
                self.axismap[t['number']] = i
            elif t['coordinate_type'] == 'stokes':
                self.axismap[2] = i
        if None in self.axismap:
            raise ValueError('Not all required axes found')

    def to_world(self, coords):
        # WCS takes indices in Fortran order
        world = self.xform.all_pix2world(coords[..., ::-1], 0)
        return world[:, self.axismap]

    def from_world(self, world):
        mid = np.zeros((world.shape[0], len(self.xform.axis_type_names)), np.float_)
        mid[:, self.axismap] = world
        # WCS returns indices in Fortran order
        return self.xform.all_world2pix(mid, 0)[..., ::-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs=3)
    args = parser.parse_args()

    files = [FitsFile(filename) for filename in args.file[:2]]
    coords0 = np.indices(files[0].data.shape)
    # Rearrange structure of arrays of indices to array of multi-indices
    coords0 = np.rollaxis(coords0, 0, len(coords0.shape))
    # Flatten the multi-dimensional array created by np.indices
    coords0 = coords0.reshape((-1, coords0.shape[-1]))
    world = files[0].to_world(coords0)
    coords1 = files[1].from_world(world)
    # Strip out axes with size 1, since they cause interpn issues
    keep_axes = [x[0] for x in enumerate(files[1].data.shape) if x[1] > 1]
    squeezed = np.squeeze(files[1].data)
    values1 = scipy.interpolate.interpn(
        tuple(np.arange(x) for x in squeezed.shape),
        squeezed,
        coords1[..., keep_axes],
        bounds_error=False)
    # Unflatten the data again
    values1 = values1.reshape(files[0].data.shape)
    delta = values1 - files[0].data

    header = files[0].hdulist[0].header.copy()
    header['ORIGIN'] = 'fitsdiffwcs'
    hdu = fits.PrimaryHDU(delta, header)
    hdu.writeto(args.file[2], overwrite=True)


if __name__ == '__main__':
    main()
