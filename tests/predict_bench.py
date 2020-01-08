#!/usr/bin/env python3

import argparse
import uuid

import numpy as np
from katsdpsigproc import accel
import katpoint
from astropy import units
from katsdpimager import parameters, polarization, sky_model, predict
from katsdpimager.test.utils import RandomState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', type=int, default=10**6)
    parser.add_argument('--sources', type=int, default=10**4)
    args = parser.parse_args()

    image_parameters = parameters.ImageParameters(
        q_fov=1.0,
        image_oversample=None,
        frequency=0.2 * units.m,
        array=None,
        polarizations=polarization.STOKES_IQUV,
        dtype=np.float64,
        pixel_size=0.00001,
        pixels=4096)
    oversample = 8
    w_planes = 100
    grid_parameters = parameters.GridParameters(
        antialias_width=7.0,
        oversample=oversample,
        image_oversample=4,
        w_slices=10,
        w_planes=w_planes,
        max_w=5 * units.m,
        kernel_width=7)
    base_sources = [
        "dummy0, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231 0 0 0 1 0.1 0 0)",       # noqa: E501
        "dummy1, radec, 19:39:20.38, -63:42:09.1, (800.0 8400.0 -3.708 3.807 -0.7202 0 0 0 1 0.2 0.2 0.2)",   # noqa: E501
        "dummy2, radec, 19:39:08.29, -63:42:33.0, (800.0 43200.0 0.956 0.584 -0.1644 0 0 0 1 0.1 0 1)"        # noqa: E501
    ]
    sources = []
    for i in range(args.sources):
        sources.append(str(uuid.uuid4()) + base_sources[i % len(base_sources)])
    model = sky_model.KatpointSkyModel(katpoint.Catalogue(sources))
    phase_centre = katpoint.construct_radec_target(
        '19:39:30', '-63:42:30').astrometric_radec() * units.rad

    rs = RandomState(seed=1)
    uv = rs.random_integers(-2048, 2048, size=(args.vis, 2)).astype(np.int16)
    sub_uv = rs.random_integers(0, grid_parameters.oversample - 1,
                                size=(args.vis, 2)).astype(np.int16)
    w_plane = rs.random_integers(0, grid_parameters.w_planes - 1, size=args.vis).astype(np.int16)
    weights = rs.uniform(size=(args.vis, len(image_parameters.polarizations))).astype(np.float32)
    vis = rs.complex_normal(size=(args.vis, len(image_parameters.polarizations)))

    context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = context.create_command_queue()
    allocator = accel.SVMAllocator(context)
    template = predict.PredictTemplate(context, np.float32, len(image_parameters.polarizations))
    fn = template.instantiate(queue, image_parameters, grid_parameters,
                              args.vis, len(model), allocator=allocator)
    fn.ensure_all_bound()
    fn.num_vis = args.vis
    fn.set_coordinates(uv, sub_uv, w_plane)
    fn.set_vis(vis)
    fn.set_weights(weights)
    fn.set_sky_model(model, phase_centre)
    fn.set_w(1.2)
    fn()
    fn()
    queue.finish()


if __name__ == '__main__':
    main()
