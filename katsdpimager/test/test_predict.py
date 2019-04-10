"""Tests for :mod:`katsdpimager.predict`"""

import numpy as np
import katpoint
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test, force_autotune
from astropy import units

from .. import predict, parameters, polarization, sky_model
from .utils import RandomState


class TestPredict:
    """Tests for :class:`.Predict`"""
    def _test_random(self, context, queue, n_vis):
        # Deliberately use a strange subset of polarizations, to test that
        # the indexing is correct.
        pols = [polarization.STOKES_I,
                polarization.STOKES_Q,
                polarization.STOKES_V]
        image_parameters = parameters.ImageParameters(
            q_fov=1.0,
            image_oversample=None,
            frequency=0.2 * units.m,
            array=None,
            polarizations=pols,
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
        catalogue = katpoint.Catalogue([
            "dummy0, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231 0 0 0 1 0.1 0 0)",       # noqa: E501
            "dummy1, radec, 19:39:20.38, -63:42:09.1, (800.0 8400.0 -3.708 3.807 -0.7202 0 0 0 1 0.2 0.2 0.2)",   # noqa: E501
            "dummy2, radec, 19:39:08.29, -63:42:33.0, (800.0 43200.0 0.956 0.584 -0.1644 0 0 0 1 0.1 0 1)"        # noqa: E501
        ])
        model = sky_model.KatpointSkyModel(catalogue)
        phase_centre = katpoint.construct_radec_target('19:39:30', '-63:42:30').astrometric_radec()
        phase_centre = phase_centre * units.rad
        rs = RandomState(seed=1)
        uv = rs.random_integers(-2048, 2048, size=(n_vis, 2)).astype(np.int16)
        sub_uv = rs.random_integers(0, oversample - 1, size=(n_vis, 2)).astype(np.int16)
        w_plane = rs.random_integers(0, w_planes - 1, size=n_vis).astype(np.int16)
        weights = rs.uniform(size=(n_vis, len(pols))).astype(np.float32)
        vis = rs.complex_normal(size=(n_vis, len(pols)))

        allocator = accel.SVMAllocator(context)
        template = predict.PredictTemplate(context, np.float32, len(pols))
        fn = template.instantiate(queue, image_parameters, grid_parameters,
                                  n_vis, len(model), allocator=allocator)
        fn.ensure_all_bound()
        fn.num_vis = n_vis
        fn.set_coordinates(uv, sub_uv, w_plane)
        fn.set_vis(vis)
        fn.set_weights(weights)
        fn.set_sky_model(model, phase_centre)
        fn.set_w(1.2)
        fn()

        host = predict.PredictHost(image_parameters, grid_parameters)
        host.num_vis = n_vis
        host.set_coordinates(uv, sub_uv, w_plane)
        host.set_vis(vis)
        host.set_weights(weights)
        host.set_sky_model(model, phase_centre)
        host.set_w(1.2)
        host()

        queue.finish()
        device_vis = fn.buffer('vis')[:n_vis]
        # Accuracy of individual visibilities is low, because the delay can have
        # a large whole number of wavelengths which degrades the precision of the
        # phase. Differences in where FMAs are inserted by compilers causes
        # differences.
        np.testing.assert_allclose(device_vis, vis, rtol=5e-4)

    @device_test
    def test_random(self, context, queue):
        """Compare the host and device versions with random data"""
        self._test_random(context, queue, 1001)

    @device_test
    def test_few_vis(self, context, queue):
        """Test with fewer visibilities than sources.

        This is a regression test: the first version had a bug in this case.
        """
        self._test_random(context, queue, 2)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that autotuner runs successfully"""
        predict.PredictTemplate(context, np.float32, 4)
        predict.PredictTemplate(context, np.float64, 4)
