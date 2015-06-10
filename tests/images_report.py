#!/usr/bin/env python

"""
Creates an HTML page showing the images written by the Jenkins script.
"""

import matplotlib
matplotlib.use('Agg')
import aplpy
from astropy.io import fits
from mako.template import Template
import argparse
import os
import os.path
import errno
from contextlib import closing

class Image(object):
    def __init__(self, name, filebase, fits_pattern):
        self.name = name
        self.filebase = filebase
        self.fits_pattern = fits_pattern

    def fits_filename(self, stokes):
        return self.fits_pattern.format(stokes=stokes)

    def output_filename(self, stokes):
        return '{}-{}.svg'.format(self.filebase, stokes)

    def build(self, output_dir, stokes):
        # Check whether the Stoke parameters are all bundled into one
        # image, and if so, slice it.
        filename = self.fits_filename(stokes)
        with closing(fits.open(filename)) as hdulist:
            naxis = int(hdulist[0].header['NAXIS'])
            slices = [0] * (naxis - 2)
            for i in range(3, naxis + 1):
                if hdulist[0].header['CTYPE{}'.format(i)] == 'STOKES':
                    # TODO: should use the WCS transformation
                    slices[i - 3] = 'IQUV'.find(stokes)
        f = aplpy.FITSFigure(filename, slices=slices, figsize=(4, 4))
        f.show_colorscale()
        f.add_colorbar()
        f.add_grid()
        f.colorbar.set_font(size='x-small')
        f.tick_labels.set_font(size='x-small')
        f.save(os.path.join(output_dir, self.output_filename(stokes)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--stokes', default='IQUV', help='Stokes parameters to show')
    args = parser.parse_args()
    images = [
        Image('WSClean', 'wsclean', 'wsclean-{stokes}-image.fits'),
        Image('katsdpimager (GPU)', 'katsdpimager-gpu', 'image-gpu.fits'),
        Image('katsdpimager (CPU)', 'katsdpimager-cpu', 'image-cpu.fits')
    ]
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if not os.path.isdir(args.output_dir):
            raise
    for image in images:
        for stokes in args.stokes:
            image.build(args.output_dir, stokes)

    template_filename = os.path.join(os.path.dirname(__file__), 'report.html.mako')
    template = Template(filename=template_filename)
    with open(os.path.join(args.output_dir, 'index.html'), 'w') as f:
        f.write(template.render(
            revision=os.environ.get('GIT_COMMIT', 'Unknown'),
            stokes=args.stokes,
            images=images))

if __name__ == '__main__':
    main()
