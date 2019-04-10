#!/usr/bin/env python

"""Benchmark suite for various steps in imaging"""

import argparse
import timeit
import json

import katpoint
import pkg_resources
import numpy as np
from astropy import units
from katsdpsigproc import accel

from katsdpimager import grid, preprocess, parameters, polarization, fft


def load_antennas():
    """Return an array of katpoint antenna objects"""
    antennas = []
    with pkg_resources.resource_stream(__name__, 'meerkat_antennas.txt') as f:
        for line in f:
            antennas.append(katpoint.Antenna(line))
    return antennas


def add_parameters(args):
    """Augment args with extra fields for parameter objects"""
    grid_oversample = 8
    antialias_width = 7
    # This is used to pick the number of w slices. That allows the --kernel-width
    # argument to vary ONLY the kernel width used in gridding, without
    # simultaneously affecting the set of visibilities used.
    nominal_kernel_width = 60
    if hasattr(args, 'kernel_width'):
        kernel_width = args.kernel_width
    else:
        kernel_width = nominal_kernel_width
    eps_w = 0.001
    antennas = load_antennas()
    longest = 0.0
    for i in range(len(antennas)):
        for j in range(i):
            longest = max(longest, np.linalg.norm(antennas[i].baseline_toward(antennas[j])))
    longest = longest * units.m
    array_parameters = parameters.ArrayParameters(antennas[0].diameter * units.m, longest)
    image_parameters = parameters.ImageParameters(
        q_fov=1.0,
        image_oversample=5,
        frequency=args.frequency * units.Hz,
        array=array_parameters,
        polarizations=polarization.STOKES_IQUV[:args.polarizations],
        dtype=np.float32)

    grid_oversample = 8
    w_step = image_parameters.cell_size / grid_oversample
    w_slices = parameters.w_slices(
        image_parameters,
        longest,
        eps_w,
        nominal_kernel_width,
        antialias_width)
    w_planes = int(np.ceil(longest / w_step / w_slices))
    grid_parameters = parameters.GridParameters(
        antialias_width=antialias_width,
        oversample=grid_oversample,
        image_oversample=4,
        w_slices=w_slices,
        w_planes=w_planes,
        max_w=longest,
        kernel_width=kernel_width)
    args.antennas = antennas
    args.array_parameters = array_parameters
    args.grid_parameters = grid_parameters
    args.image_parameters = image_parameters


def make_uvw(args, n_time):
    start = 946728000.0  # J2000, in UNIX time
    dec = -np.pi / 4
    target = katpoint.construct_radec_target(0, dec)
    timestamps = np.arange(n_time) * args.int_time + start
    ref = katpoint.Antenna('', *args.antennas[0].ref_position_wgs84)
    basis = target.uvw_basis(timestamp=timestamps, antenna=ref)
    antenna_uvw = []
    rows = [[], [], []]
    for antenna in args.antennas:
        enu = np.array(ref.baseline_toward(antenna))
        antenna_uvw.append(np.tensordot(basis, enu, ([1], [0])))
    for i in range(len(args.antennas)):
        for j in range(i):
            u, v, w = antenna_uvw[j] - antenna_uvw[i]
            rows[0].append(u)
            rows[1].append(v)
            rows[2].append(w)
    uvw = np.array(rows, dtype=np.float32)
    return uvw.reshape(3, uvw.shape[1] * n_time).T.copy()


def make_vis(args, n_time):
    """Generate uncompressed visibilities

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    n_time : int
        Number of time samples
    """
    n_baselines = len(args.antennas) * (len(args.antennas) - 1) // 2
    N = n_time * n_baselines
    uvw = make_uvw(args, n_time)
    weights = np.ones((1, N, args.polarizations), np.float32)
    vis = np.ones((1, N, args.polarizations), np.complex64)
    baselines = np.repeat(np.arange(n_baselines, dtype=np.int32), n_time)
    return uvw, weights, baselines, vis


def make_compressed_vis(args, n_time):
    """Generate compressed visibilities

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    n_time : int
        Number of time samples

    Returns
    -------
    reader : :py:class:`katsdpimager.preprocess.VisibilityReader`
        Reader that iterates the visibilities
    """
    uvw, weights, baselines, vis = make_vis(args, n_time)
    if args.write:
        collector = preprocess.VisibilityCollectorHDF5(
            args.write,
            [args.image_parameters], [args.grid_parameters], vis.shape[1])
    else:
        collector = preprocess.VisibilityCollectorMem(
            [args.image_parameters], [args.grid_parameters], vis.shape[1])
    mueller = np.identity(args.polarizations, np.complex64)
    collector.add(uvw, weights, baselines, vis, None, None, mueller, None)
    collector.close()
    reader = collector.reader()
    return reader


def benchmark_preprocess(args):
    add_parameters(args)
    n_time = 900
    uvw, weights, baselines, vis = make_vis(args, n_time)
    n_vis = vis.shape[0] * vis.shape[1]
    start = timeit.default_timer()
    collector = preprocess.VisibilityCollectorMem(
        [args.image_parameters], [args.grid_parameters], n_vis)
    mueller = np.identity(args.polarizations, np.complex64)
    collector.add(uvw, weights, baselines, vis, None, None, mueller, None)
    collector.close()
    end = timeit.default_timer()
    elapsed = end - start
    print("preprocessed {} visibilities in {} seconds".format(n_vis, elapsed))
    print("{:.2f} Mvis/s".format(n_vis / elapsed / 1e6))


def benchmark_grid_degrid(args):
    n_time = 3600
    add_parameters(args)
    N = n_time * len(args.antennas) * (len(args.antennas) - 1) // 2
    reader = make_compressed_vis(args, n_time)

    context = accel.create_some_context()
    queue = context.create_tuning_command_queue()
    gridder_template = args.template_class(context, args.image_parameters,
                                           args.grid_parameters, tuning=args.tuning)
    gridder = gridder_template.instantiate(queue, args.array_parameters, N)
    gridder.ensure_all_bound()
    elapsed = 0.0
    N_compressed = 0
    uv = gridder.buffer('uv').empty_like()
    w_plane = gridder.buffer('w_plane').empty_like()
    vis = gridder.buffer('vis').empty_like()
    for w_slice in range(reader.num_w_slices(0)):
        gridder.num_vis = reader.len(0, w_slice)
        N_compressed += gridder.num_vis
        if gridder.num_vis > 0:
            gridder.buffer('grid').zero(queue)
            start = 0
            for chunk in reader.iter_slice(0, w_slice):
                rng = slice(start, start + len(chunk))
                uv[rng, 0:2] = chunk['uv']
                uv[rng, 2:4] = chunk['sub_uv']
                w_plane[rng] = chunk['w_plane']
                vis[rng] = chunk['vis']
                start += len(chunk)
            gridder.buffer('uv').set_async(queue, uv)
            gridder.buffer('w_plane').set_async(queue, w_plane)
            gridder.buffer('vis').set_async(queue, vis)
            queue.finish()
            queue.start_tuning()
            gridder()
            elapsed += queue.stop_tuning()
            queue.finish()
    gaps = N_compressed * args.grid_parameters.kernel_width**2 * args.polarizations / elapsed
    print('Processed {} ({}) visibilities in {:.6f}s with kernel size {} and {} polarizations'
          .format(N_compressed, N, elapsed, args.grid_parameters.kernel_width, args.polarizations))
    print('{:.3f} GGAPS uncompressed'.format(gaps * N / N_compressed / 1e9))
    print('{:.3f} GGAPS compressed'.format(gaps / 1e9))


def benchmark_fft(args):
    context = accel.create_some_context()
    queue = context.create_tuning_command_queue()
    allocator = accel.SVMAllocator(context)
    shape = (args.pixels, args.pixels)
    template = fft.FftTemplate(queue, 2, shape, np.complex64, np.complex64, shape, shape)
    fn = template.instantiate(args.mode, allocator=allocator)
    fn.ensure_all_bound()
    # Zero-fill, just to ensure no NaNs etc
    fn.buffer('src').fill(0)
    fn.buffer('dest').fill(0)
    fn()  # Warm-up and forces data transfer
    queue.start_tuning()
    fn()
    elapsed = queue.stop_tuning()
    print('{pixels}x{pixels} in {elapsed:.6f} seconds'.format(pixels=args.pixels, elapsed=elapsed))
    # 8 bytes for complex64, 4 accesses (from source, to/from scratch, to dest)
    mem_rate = args.pixels * args.pixels * 8 * 4 / elapsed
    print('{:.3f} GiB/s'.format(mem_rate / 1024**3))


def add_arguments(subparser, arguments):
    arg_map = {
        '--polarizations': lambda: subparser.add_argument(
            '--polarizations', type=int, default=4, choices=[1, 2, 3, 4],
            help='Number of polarizations'),
        '--frequency': lambda: subparser.add_argument(
            '--frequency', type=float, default=1412000000.0,
            help='Observation frequency (Hz)'),
        '--int-time': lambda: subparser.add_argument(
            '--int-time', type=float, default=2.0, help='Integration time (seconds)'),
        '--kernel-width': lambda: subparser.add_argument(
            '--kernel-width', type=int, default=60, help='Convolutional kernel size in pixels'),
        '--pixels': lambda: subparser.add_argument(
            '--pixels', type=int, default=4608, help='Grid/image dimensions'),
        '--tuning': lambda: subparser.add_argument(
            '--tuning', type=json.loads, help='Tuning arguments (JSON)'),
        '--write': lambda: subparser.add_argument(
            '--write', type=str, help='Write compressed visibilities to HDF5 file')
    }
    for arg_name in arguments:
        arg_map[arg_name]()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-commands')

    parser_preprocess = subparsers.add_parser('preprocess', help='preprocessing benchmark')
    parser_preprocess.set_defaults(func=benchmark_preprocess)
    add_arguments(parser_preprocess, ['--polarizations', '--frequency', '--int-time'])

    grid_degrid_arguments = ['--polarizations', '--frequency', '--int-time', '--kernel-width',
                             '--tuning', '--write']
    parser_grid = subparsers.add_parser('grid', help='gridding benchmark')
    parser_grid.set_defaults(func=benchmark_grid_degrid, template_class=grid.GridderTemplate)
    add_arguments(parser_grid, grid_degrid_arguments)

    parser_degrid = subparsers.add_parser('degrid', help='degridding benchmark')
    parser_degrid.set_defaults(func=benchmark_grid_degrid, template_class=grid.DegridderTemplate)
    add_arguments(parser_degrid, grid_degrid_arguments)

    parser_ifft = subparsers.add_parser('ifft', help='grid-to-image FFT benchmark')
    add_arguments(parser_ifft, ['--pixels'])
    parser_ifft.set_defaults(func=benchmark_fft, mode=fft.FFT_INVERSE)

    parser_fft = subparsers.add_parser('fft', help='image-to-grid FFT benchmark')
    add_arguments(parser_fft, ['--pixels'])
    parser_fft.set_defaults(func=benchmark_fft, mode=fft.FFT_FORWARD)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
