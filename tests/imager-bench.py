#!/usr/bin/env python

"""Benchmark suite for various steps in imaging"""

from __future__ import print_function, absolute_import, division
import argparse
import katpoint
import ephem
import pkg_resources
import numpy as np
from astropy import units
from katsdpimager import grid, preprocess, parameters, polarization
from katsdpsigproc import accel


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
    kernel_width = 60
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
        kernel_width,
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


def benchmark_grid(args):
    add_parameters(args)
    N = 8 * 1024**2
    collector = preprocess.VisibilityCollectorMem(
            [args.image_parameters], args.grid_parameters, N)
    n_baselines = len(args.antennas) * (len(args.antennas) - 1) // 2
    n_time = N // n_baselines
    N = n_time * n_baselines
    uvw = make_uvw(args, n_time)
    weights = np.ones((N, args.polarizations), np.float32)
    vis = np.ones((N, args.polarizations), np.complex64)
    baselines = np.repeat(np.arange(n_baselines, dtype=np.int32), n_time)
    collector.add(0, uvw, weights, baselines, vis)
    collector.close()
    reader = collector.reader()

    context = accel.create_some_context()
    queue = context.create_tuning_command_queue()
    allocator = accel.SVMAllocator(context)
    gridder_template = grid.GridderTemplate(context, args.image_parameters, args.grid_parameters)
    gridder = gridder_template.instantiate(queue, args.array_parameters, N, allocator=allocator)
    gridder.ensure_all_bound()
    elapsed = 0.0
    N_compressed = 0
    for w_slice in range(reader.num_w_slices):
        gridder.num_vis = reader.len(0, w_slice)
        N_compressed += gridder.num_vis
        if gridder.num_vis > 0:
            # TODO: zero the grid before each addition
            start = 0
            for chunk in reader.iter_slice(0, w_slice):
                rng = slice(start, start + len(chunk))
                gridder.buffer('uv')[rng, 0:2] = chunk['uv']
                gridder.buffer('uv')[rng, 2:4] = chunk['sub_uv']
                gridder.buffer('w_plane')[rng] = chunk['w_plane']
                gridder.buffer('vis')[rng] = chunk['vis']
                start += len(chunk)
            gridder()  # Forces data transfer
            queue.start_tuning()
            gridder()
            elapsed += queue.stop_tuning()
            queue.finish()
    gaps = N_compressed * args.grid_parameters.kernel_width**2 * args.polarizations / elapsed
    print('Gridded {} ({}) points in {:.6f}s with kernel size {} and {} polarizations'.format(
        N_compressed, N, elapsed, args.grid_parameters.kernel_width, args.polarizations))
    print('{:.3f} GGAPS uncompressed'.format(gaps * N / N_compressed / 1e9))
    print('{:.3f} GGAPS compressed'.format(gaps / 1e9))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-commands')
    parser_grid = subparsers.add_parser('grid', help='gridding benchmark')
    parser_grid.add_argument('--polarizations', type=int, default=4, choices=[1, 2, 3, 4], help='Number of polarizations')
    parser_grid.add_argument('--frequency', type=float, default=1412000000.0, help='Observation frequency (Hz)')
    parser_grid.add_argument('--int-time', type=float, default=2.0, help='Integration time (seconds)')
    parser_grid.set_defaults(func=benchmark_grid)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
