#!/usr/bin/env python

"""
Creates an HTML page showing the images written by the Jenkins script.
"""

import argparse
import os
import os.path
import shutil
import sys
import subprocess
import logging
import timeit
import io
import glob
from contextlib import closing

from mako.template import Template
import matplotlib
matplotlib.use('Agg')   # noqa: E402
import aplpy
from astropy.io import fits
from astropy.wcs import WCS


MODE_CLEAN = 'Clean'
MODE_DIRTY = 'Dirty'
MODES = [MODE_CLEAN, MODE_DIRTY]


class BuildInfo:
    """Runs a command and stores the output and exit code. The output is also
    echoed to stdout.
    """
    def __init__(self, cmds):
        self.cmds = cmds
        start = timeit.default_timer()
        output = io.BytesIO()
        self.returncode = 0
        for cmd in cmds:
            sys.stdout.flush()
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # The child.stdout seems to do full reads, even though it is supposedly unbuffered
            stdout = io.open(child.stdout.fileno(), mode='rb', buffering=0, closefd=False)
            sys_stdout = io.open(sys.stdout.fileno(), mode='wb', buffering=0, closefd=False)
            while True:
                data = stdout.read(256)
                if not data:
                    break
                output.write(data)
                sys_stdout.write(data)
                sys_stdout.flush()
            self.returncode = child.wait()
            if self.returncode != 0:
                logging.warning("Process failed with exit code %s", self.returncode)
                break
        self.output = output.getvalue().decode('utf-8')
        end = timeit.default_timer()
        self.elapsed = end - start
        logging.info("Elapsed time %.3f", self.elapsed)


class Image:
    def __init__(self, name, filebase,
                 clean_fits_pattern, dirty_fits_pattern, cmd_patterns,
                 clean_globs=[]):
        self.name = name
        self.filebase = filebase
        self.fits_patterns = {MODE_CLEAN: clean_fits_pattern, MODE_DIRTY: dirty_fits_pattern}
        self.cmd_patterns = cmd_patterns
        self.clean_globs = list(clean_globs)

    def fits_filename(self, mode, stokes, channel, rel_channel):
        return self.fits_patterns[mode].format(stokes=stokes, channel=channel,
                                               rel_channel=rel_channel)

    def svg_filename_thumb(self, mode, stokes, channel):
        return '{}-{}-{}-{}-thumb.svg'.format(self.filebase, mode, stokes, channel)

    def svg_filename_full(self, mode, stokes, channel):
        return '{}-{}-{}-{}.svg'.format(self.filebase, mode, stokes, channel)

    def build_info_filename(self):
        return '{}-log.html'.format(self.filebase)

    def build(self, ms, output_dir, modes, stokes, channels):
        cmds = [[
                Template(x).render_unicode(
                    ms=ms, output_dir=output_dir, stokes=stokes, channels=channels)
                for x in pattern
                ] for pattern in self.cmd_patterns]
        logging.info("Running %s...", cmds)
        build_info = BuildInfo(cmds)
        # Clean up unwanted files, making sure not to delete any files we
        # actually wanted.
        keep = set(os.path.join(output_dir,
                                self.fits_filename(mode, s, channel, channel - channels[0]))
                   for mode in modes
                   for s in stokes
                   for channel in channels)
        for clean_glob in self.clean_globs:
            for filename in glob.iglob(os.path.join(output_dir, clean_glob)):
                if filename not in keep:
                    logging.info('Removing intermediate file %s', filename)
                    if os.path.isdir(filename):
                        shutil.rmtree(filename)
                    else:
                        os.remove(filename)
        return build_info

    def _render_common(self, figure, colorscale_args={}):
        figure.show_colorscale(vmin=-0.01, vmax=2.0, vmid=-0.015, stretch='log', **colorscale_args)
        figure.add_colorbar()
        figure.add_grid()

    def _render_thumb(self, output_dir, input_filename, slices, mode, stokes, channel):
        figure = aplpy.FITSFigure(input_filename, slices=slices, figsize=(4, 4))
        self._render_common(figure)
        figure.colorbar.set_font(size='x-small')
        figure.tick_labels.set_font(size='xx-small')
        figure.save(os.path.join(output_dir, self.svg_filename_thumb(mode, stokes, channel)))
        figure.close()

    def _render_full(self, output_dir, input_filename, slices, mode, stokes, channel):
        figure = aplpy.FITSFigure(input_filename, slices=slices, figsize=(19, 18))
        self._render_common(figure, colorscale_args=dict(interpolation=None))
        figure.colorbar.set_font(size='small')
        figure.tick_labels.set_font(size='small')
        figure.save(os.path.join(output_dir, self.svg_filename_full(mode, stokes, channel)))
        figure.close()

    def render(self, output_dir, mode, stokes, channel, rel_channel):
        # Check whether the Stokes parameters and/or channel are all bundled
        # into one image, and if so, slice it.
        filename = os.path.join(output_dir, self.fits_filename(mode, stokes, channel, rel_channel))
        with closing(fits.open(filename)) as hdulist:
            wcs = WCS(hdulist[0])
            wcsaxes = wcs.naxis
            # Reverse shape to put back into same order as axis_types
            shape = list(reversed(hdulist[0].shape))
            if wcsaxes < 2:
                raise ValueError('At least two dimensions expected')
            axis_types = wcs.get_axis_types()
            if axis_types[0]['coordinate_type'] != 'celestial' or axis_types[0]['number'] != 0:
                raise ValueError('First axis is not longitudinal')
            if axis_types[1]['coordinate_type'] != 'celestial' or axis_types[1]['number'] != 1:
                raise ValueError('Second axis is not latitudinal')
            slices = [0] * (len(shape) - 2)
            for i in range(2, len(shape)):
                ctype = axis_types[i]['coordinate_type']
                if ctype == 'stokes':
                    # TODO: should use the WCS transformation
                    slices[i - 2] = 'IQUV'.find(stokes)
                elif ctype == 'spectral' and shape[i] > 1:
                    slices[i - 2] = rel_channel
        self._render_thumb(output_dir, filename, slices, mode, stokes, channel)
        self._render_full(output_dir, filename, slices, mode, stokes, channel)


def write_index(args, images, build_info, modes):
    template_filename = os.path.join(os.path.dirname(__file__), 'report.html.mako')
    template = Template(filename=template_filename)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except subprocess.CalledProcessError:
        commit = 'Unknown'
    with io.open(os.path.join(args.output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(template.render_unicode(
            revision=commit,
            modes=modes,
            stokes=args.stokes,
            channels=list(range(args.start_channel, args.stop_channel)),
            images=images,
            build_info=build_info))


def write_build_log(args, image, build_info, modes):
    template_filename = os.path.join(os.path.dirname(__file__), 'build_log.html.mako')
    template = Template(filename=template_filename)
    path = os.path.join(args.output_dir, image.build_info_filename())
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(template.render_unicode(
            image=image, build_info=build_info, modes=modes, stokes=args.stokes,
            channels=list(range(args.start_channel, args.stop_channel))))


def run(args, images, modes):
    try:
        os.makedirs(args.output_dir)
    except OSError:
        if not os.path.isdir(args.output_dir):
            raise
    build_info = {}
    channels = list(range(args.start_channel, args.stop_channel))
    for image in images:
        if not args.skip_build:
            build_info[image] = image.build(args.ms, args.output_dir, modes, args.stokes, channels)
        else:
            build_info[image] = BuildInfo([])

    write_index(args, images, build_info, modes)
    for image in images:
        write_build_log(args, image, build_info[image], modes)

    for image in images:
        if build_info[image].returncode == 0:
            for mode in modes:
                for stokes in args.stokes:
                    for channel in channels:
                        image.render(args.output_dir, mode, stokes, channel, channel - channels[0])

    for info in build_info.values():
        if info.returncode != 0:
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ms', help='Measurement set to image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--skip-build', action='store_true', help='Use existing image files')
    parser.add_argument('--stokes', default='IQUV', help='Stokes parameters to show')
    parser.add_argument('--start-channel', type=int, default=0, help='First channel to image')
    parser.add_argument('--stop-channel', type=int, default=1,
                        help='One past last channel to image')
    args = parser.parse_args()

    pixel_size = 1.7    # in arcsec
    pixels = 5760
    katsdpimager_common = [
        'imager.py',
        '--stokes=${stokes}',
        '--input-option', 'data=CORRECTED_DATA',
        '--eps-w=0.001', '--major=5',
        '--start-channel=${channels[0]}',
        '--stop-channel=${channels[-1] + 1}',
        '--pixel-size={}arcsec'.format(pixel_size),
        '--pixels={}'.format(pixels),
        '${ms}']
    lwimager_common = [
        'lwimager', 'ms=${ms}', 'npix={}'.format(pixels), 'mode=channel',
        'chanstart=${channels[0]}', 'nchan=${len(channels)}',
        'img_chanstart=${channels[0]}', 'img_nchan=${len(channels)}',
        'cellsize={}arcsec'.format(pixel_size), 'wprojplanes=128', 'threshold=0.01Jy',
        'weight=natural', 'stokes=${stokes}', 'data=CORRECTED_DATA']
    images = [
        Image('WSClean', 'wsclean',
              'wsclean-{rel_channel:04}-{stokes}-image.fits',
              'wsclean-{rel_channel:04}-{stokes}-dirty.fits',
              [['wsclean', '-mgain', '0.85', '-niter', '1000', '-threshold', '0.01',
                '-weight', 'natural',
                '-size', '{}'.format(pixels), '{}'.format(pixels),
                '-scale', '{}asec'.format(pixel_size), '-pol', '${",".join(stokes.lower())}',
                '-channels-out', '${len(channels)}',
                '-channel-range', '${channels[0]}', '${channels[-1] + 1}',
                '-name', '${output_dir}/wsclean', '${ms}']],
              clean_globs=['wsclean-*.fits']),
        Image('lwimager', 'lwimager', 'lwimager.fits', 'lwimager-dirty.fits',
              [
                  lwimager_common + ['operation=image', 'fits=${output_dir}/lwimager-dirty.fits'],
                  lwimager_common + ['operation=csclean', 'fits=${output_dir}/lwimager.fits']
              ],
              clean_globs=['*-channel*.img*']),
        Image('katsdpimager (GPU, direct predict)', 'katsdpimager-gpu',
              'image-gpu-{channel}.fits', 'dirty-gpu-{channel}.fits',
              [katsdpimager_common + ['--write-dirty=${output_dir}/dirty-gpu-%d.fits',
                                      '${output_dir}/image-gpu-%d.fits']]),
        Image('katsdpimager (GPU, degridding)', 'katsdpimager-gpu-degrid',
              'image-gpu-degrid-{channel}.fits', 'dirty-gpu-degrid-{channel}.fits',
              [katsdpimager_common + ['--degrid',
                                      '--write-dirty=${output_dir}/dirty-gpu-degrid-%d.fits',
                                      '${output_dir}/image-gpu-degrid-%d.fits']]),
        Image('katsdpimager (CPU)', 'katsdpimager-cpu',
              'image-cpu-{channel}.fits', 'dirty-cpu-{channel}.fits',
              [katsdpimager_common + ['--host', '--write-dirty=${output_dir}/dirty-cpu-%d.fits',
                                      '${output_dir}/image-cpu-%d.fits']])
    ]
    return run(args, images, MODES)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
