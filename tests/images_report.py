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
import shutil
import sys
import errno
import subprocess
import logging
import timeit
import io
import glob
from contextlib import closing

MODE_CLEAN = 'Clean'
MODE_DIRTY = 'Dirty'
MODES = [MODE_CLEAN, MODE_DIRTY]

class BuildInfo(object):
    """Runs a command and stores the output and exit code. The output is also
    echoed to stdout.
    """
    def __init__(self, cmds):
        self.cmds = cmds
        start = timeit.default_timer()
        output = io.BytesIO()
        for cmd in cmds:
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # The child.stdout seems to do full reads, even though it is supposedly unbuffered
            stdout = io.open(child.stdout.fileno(), mode='rb', buffering=0, closefd=False)
            while True:
                data = stdout.read(256)
                if not data:
                    break
                output.write(data)
                sys.stdout.write(data)
                sys.stdout.flush()
            self.returncode = child.wait()
            if self.returncode != 0:
                logging.warning("Process failed with exit code %s", self.returncode)
                break
        self.output = output.getvalue().decode('utf-8')
        end = timeit.default_timer()
        self.elapsed = end - start
        logging.info("Elapsed time %.3f", self.elapsed)

class Image(object):
    def __init__(self, name, filebase, clean_fits_pattern, dirty_fits_pattern, cmd_patterns, clean_globs=[]):
        self.name = name
        self.filebase = filebase
        self.fits_patterns = {MODE_CLEAN: clean_fits_pattern, MODE_DIRTY: dirty_fits_pattern}
        self.cmd_patterns = cmd_patterns
        self.clean_globs = list(clean_globs)

    def fits_filename(self, mode, stokes):
        return self.fits_patterns[mode].format(stokes=stokes)

    def svg_filename_thumb(self, mode, stokes):
        return '{}-{}-{}-thumb.svg'.format(self.filebase, mode, stokes)

    def svg_filename_full(self, mode, stokes):
        return '{}-{}-{}.svg'.format(self.filebase, mode, stokes)

    def build_info_filename(self):
        return '{}-log.html'.format(self.filebase)

    def build(self, ms, output_dir, modes, stokes):
        cmds = [[
                Template(x).render_unicode(ms=ms, output_dir=output_dir, stokes=stokes)
                for x in pattern
            ] for pattern in self.cmd_patterns]
        logging.info("Running %s...", cmds)
        build_info = BuildInfo(cmds)
        # Clean up unwanted files, making sure not to delete any files we
        # actually wanted.
        keep = set(os.path.join(output_dir, self.fits_filename(mode, s)) for mode in modes for s in stokes)
        for clean_glob in self.clean_globs:
            for filename in glob.iglob(os.path.join(output_dir, clean_glob)):
                if filename not in keep:
                    if os.path.isdir(filename):
                        shutil.rmtree(filename)
                    else:
                        os.remove(filename)
        return build_info

    def _render_common(self, figure, colorscale_args={}):
        figure.show_colorscale(vmin=-0.01, vmax=2.0, vmid=-0.015, stretch='log', **colorscale_args)
        figure.add_colorbar()
        figure.add_grid()

    def _render_thumb(self, output_dir, input_filename, slices, mode, stokes):
        figure = aplpy.FITSFigure(input_filename, slices=slices, figsize=(4, 4))
        self._render_common(figure)
        figure.colorbar.set_font(size='x-small')
        figure.tick_labels.set_font(size='xx-small')
        figure.save(os.path.join(output_dir, self.svg_filename_thumb(mode, stokes)))
        figure.close()

    def _render_full(self, output_dir, input_filename, slices, mode, stokes):
        figure = aplpy.FITSFigure(input_filename, slices=slices, figsize=(19, 18))
        self._render_common(figure, colorscale_args=dict(interpolation=None))
        figure.colorbar.set_font(size='small')
        figure.tick_labels.set_font(size='small')
        figure.save(os.path.join(output_dir, self.svg_filename_full(mode, stokes)))
        figure.close()

    def render(self, output_dir, mode, stokes):
        # Check whether the Stoke parameters are all bundled into one
        # image, and if so, slice it.
        filename = os.path.join(output_dir, self.fits_filename(mode, stokes))
        with closing(fits.open(filename)) as hdulist:
            naxis = int(hdulist[0].header['NAXIS'])
            slices = [0] * (naxis - 2)
            for i in range(3, naxis + 1):
                if hdulist[0].header['CTYPE{}'.format(i)] == 'STOKES':
                    # TODO: should use the WCS transformation
                    slices[i - 3] = 'IQUV'.find(stokes)
        self._render_thumb(output_dir, filename, slices, mode, stokes)
        self._render_full(output_dir, filename, slices, mode, stokes)


def write_index(args, images, build_info, modes):
    template_filename = os.path.join(os.path.dirname(__file__), 'report.html.mako')
    template = Template(filename=template_filename)
    try:
        commit = subprocess.check_output('git rev-parse HEAD')
    except subprocess.CalledProcessError:
        commit = 'Unknown'
    with io.open(os.path.join(args.output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(template.render_unicode(
            revision=commit,
            modes=modes,
            stokes=args.stokes,
            images=images,
            build_info=build_info))

def write_build_log(args, image, build_info, modes):
    template_filename = os.path.join(os.path.dirname(__file__), 'build_log.html.mako')
    template = Template(filename=template_filename)
    with io.open(os.path.join(args.output_dir, image.build_info_filename()), 'w', encoding='utf-8') as f:
        f.write(template.render_unicode(
            image=image, build_info=build_info, modes=modes, stokes=args.stokes))

def run(args, images, modes):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if not os.path.isdir(args.output_dir):
            raise
    build_info = {}
    for image in images:
        build_info[image] = image.build(args.ms, args.output_dir, modes, args.stokes)
    write_index(args, images, build_info, modes)
    for image in images:
        write_build_log(args, image, build_info[image], modes)

    for image in images:
        if build_info[image].returncode == 0:
            for mode in modes:
                for stokes in args.stokes:
                    image.render(args.output_dir, mode, stokes)

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

    pixel_size = 3.49328831150462    # in arcsec: to match the value computed by katsdpimager
    katsdpimager_common = [
        'imager.py',
        '--stokes=${stokes}',
        '--input-option', 'data=CORRECTED_DATA',
        '--eps-w=0.001', '--major=5',
        '${ms}']
    lwimager_common = [
        'lwimager', 'ms=${ms}', 'npix=4608', 'cellsize={}arcsec'.format(pixel_size), 'wprojplanes=128', 'threshold=0.01Jy',
        'weight=natural', 'stokes=${stokes}', 'data=CORRECTED_DATA']
    images = [
        Image('WSClean', 'wsclean', 'wsclean-{stokes}-image.fits', 'wsclean-{stokes}-dirty.fits',
              [['wsclean', '-mgain', '0.85', '-niter', '1000', '-threshold', '0.01',
                '-weight', 'natural',
                '-size', '4608', '4608', '-scale', '{}asec'.format(pixel_size), '-pol', '${",".join(stokes.lower())}',
                '-name', '${output_dir}/wsclean', '${ms}']],
              clean_globs=['wsclean-*.fits']),
        Image('lwimager', 'lwimager', 'lwimager.fits', 'lwimager-dirty.fits',
              [
                  lwimager_common + ['operation=image', 'fits=${output_dir}/lwimager-dirty.fits'],
                  lwimager_common + ['operation=csclean', 'fits=${output_dir}/lwimager.fits']
              ],
              clean_globs=['*-mfs1.img*']),
        Image('katsdpimager (GPU)', 'katsdpimager-gpu', 'image-gpu.fits', 'dirty-gpu.fits',
              [katsdpimager_common + ['--write-dirty=${output_dir}/dirty-gpu.fits', '${output_dir}/image-gpu.fits']]),
        Image('katsdpimager (CPU)', 'katsdpimager-cpu', 'image-cpu.fits', 'dirty-cpu.fits',
              [katsdpimager_common + ['--host', '--write-dirty=${output_dir}/dirty-cpu.fits', '${output_dir}/image-cpu.fits']])
    ]
    return run(args, images, MODES)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
