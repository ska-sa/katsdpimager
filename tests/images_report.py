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
import sys
import errno
import subprocess
import logging
import timeit
import io
import glob
from contextlib import closing

class BuildInfo(object):
    def __init__(self, cmd):
        self.cmd = cmd
        start = timeit.default_timer()
        try:
            self.output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            self.returncode = 0
        except subprocess.CalledProcessError as e:
            self.output = e.output
            self.returncode = e.returncode
            logging.warning(str(e) + '\n' + e.output)
        end = timeit.default_timer()
        self.elapsed = end - start
        self.output = self.output.decode('utf-8')

class Image(object):
    def __init__(self, name, filebase, fits_pattern, cmd_pattern, clean_globs=[]):
        self.name = name
        self.filebase = filebase
        self.fits_pattern = fits_pattern
        self.cmd_pattern = cmd_pattern
        self.clean_globs = list(clean_globs)

    def fits_filename(self, stokes):
        return self.fits_pattern.format(stokes=stokes)

    def svg_filename_thumb(self, stokes):
        return '{}-{}-thumb.svg'.format(self.filebase, stokes)

    def svg_filename_full(self, stokes):
        return '{}-{}.svg'.format(self.filebase, stokes)

    def build_info_filename(self):
        return '{}-log.html'.format(self.filebase)

    def build(self, ms, output_dir, stokes):
        cmd = [Template(x).render_unicode(ms=ms, output_dir=output_dir, stokes=stokes)
               for x in self.cmd_pattern]
        build_info = BuildInfo(cmd)
        # Clean up unwanted files, making sure not to delete any files we
        # actually wanted.
        keep = set(os.path.join(output_dir, self.fits_filename(s)) for s in stokes)
        for clean_glob in self.clean_globs:
            for filename in glob.iglob(os.path.join(output_dir, clean_glob)):
                if filename not in keep:
                    os.remove(filename)
        return build_info

    def _render_common(self, figure, colorscale_args={}):
        figure.show_colorscale(vmin=-0.01, vmax=2.0, vmid=-0.015, stretch='log', **colorscale_args)
        figure.add_colorbar()
        figure.add_grid()

    def _render_thumb(self, output_dir, input_filename, slices, stokes):
        figure = aplpy.FITSFigure(input_filename, slices=slices, figsize=(4, 4))
        self._render_common(figure)
        figure.colorbar.set_font(size='x-small')
        figure.tick_labels.set_font(size='xx-small')
        figure.save(os.path.join(output_dir, self.svg_filename_thumb(stokes)))

    def _render_full(self, output_dir, input_filename, slices, stokes):
        figure = aplpy.FITSFigure(input_filename, slices=slices, figsize=(19, 18))
        self._render_common(figure, colorscale_args=dict(interpolation=None))
        figure.colorbar.set_font(size='small')
        figure.tick_labels.set_font(size='small')
        figure.save(os.path.join(output_dir, self.svg_filename_full(stokes)))

    def render(self, output_dir, stokes):
        # Check whether the Stoke parameters are all bundled into one
        # image, and if so, slice it.
        filename = os.path.join(output_dir, self.fits_filename(stokes))
        with closing(fits.open(filename)) as hdulist:
            naxis = int(hdulist[0].header['NAXIS'])
            slices = [0] * (naxis - 2)
            for i in range(3, naxis + 1):
                if hdulist[0].header['CTYPE{}'.format(i)] == 'STOKES':
                    # TODO: should use the WCS transformation
                    slices[i - 3] = 'IQUV'.find(stokes)
        self._render_thumb(output_dir, filename, slices, stokes)
        self._render_full(output_dir, filename, slices, stokes)


def write_index(args, images, build_info):
    template_filename = os.path.join(os.path.dirname(__file__), 'report.html.mako')
    template = Template(filename=template_filename)
    with io.open(os.path.join(args.output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(template.render_unicode(
            revision=os.environ.get('GIT_COMMIT', 'Unknown'),
            stokes=args.stokes,
            images=images,
            build_info=build_info))

def write_build_log(args, image, build_info):
    template_filename = os.path.join(os.path.dirname(__file__), 'build_log.html.mako')
    template = Template(filename=template_filename)
    with io.open(os.path.join(args.output_dir, image.build_info_filename()), 'w', encoding='utf-8') as f:
        f.write(template.render_unicode(image=image, build_info=build_info, stokes=args.stokes))

def run(args, images):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if not os.path.isdir(args.output_dir):
            raise
    build_info = {}
    for image in images:
        build_info[image] = image.build(args.ms, args.output_dir, args.stokes)
    write_index(args, images, build_info)
    for image in images:
        write_build_log(args, image, build_info[image])

    for image in images:
        if build_info[image].returncode == 0:
            for stokes in args.stokes:
                image.render(args.output_dir, stokes)

    for info in build_info.values():
        if info.returncode != 0:
            return 1
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ms', help='Measurement set to image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--stokes', default='IQUV', help='Stokes parameters to show')
    args = parser.parse_args()
    katsdpimager_common = ['imager.py', '--stokes=${stokes}', '--input-option', 'data=CORRECTED_DATA', '${ms}']
    images = [
        Image('WSClean', 'wsclean', 'wsclean-{stokes}-image.fits',
              ['wsclean', '-mgain', '0.85', '-niter', '1000', '-threshold', '0.01',
               '-size', '4608', '4608', '-scale', '1.747asec', '-pol', '${",".join(stokes.lower())}',
               '-name', '${output_dir}/wsclean', '${ms}'],
              clean_globs=['wsclean-*.fits']),
        Image('katsdpimager (GPU)', 'katsdpimager-gpu', 'image-gpu.fits',
              katsdpimager_common + ['${output_dir}/image-gpu.fits']),
        Image('katsdpimager (CPU)', 'katsdpimager-cpu', 'image-cpu.fits',
              katsdpimager_common + ['--host', '${output_dir}/image-cpu.fits'])
    ]
    return run(args, images)

if __name__ == '__main__':
    sys.exit(main())
