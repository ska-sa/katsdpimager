"""Tests for :py:mod:`katsdpimager.clean`"""

from __future__ import division
import numpy as np
import scipy.signal
import katsdpimager.clean as clean
import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
from katsdpsigproc.test.test_accel import device_test, cuda_test
import mock
from nose.tools import *

class TestUpdateTiles(object):
    def setup(self):
        self.clean_parameters = parameters.CleanParameters(
            100, 0.25, clean.CLEAN_I, 45, 0)

    @classmethod
    def _make_psf(cls, size):
        """Creates a dummy PSF as a Gaussian plus random noise"""
        rs = np.random.RandomState(seed=1)
        gaussian1d = scipy.signal.gaussian(size, size / 8)
        psf = np.outer(gaussian1d, gaussian1d)
        psf[:] += rs.standard_normal(psf.shape)
        return psf.astype(np.float32)

    @classmethod
    def _zero_buffer(cls, command_queue, buf):
        buf.set(command_queue, np.zeros(buf.shape, buf.dtype))

    @device_test
    def test_update_tiles(self, context, command_queue):
        pixels = 567
        rs = np.random.RandomState(seed=1)
        template = clean._UpdateTilesTemplate(context, self.clean_parameters, np.float32, 4)
        fn = template.instantiate(command_queue, pixels)
        fn.ensure_all_bound()
        # TODO: this could lead to ties, which will fail the test
        dirty = rs.standard_normal((pixels, pixels, 4)).astype(np.float32)
        fn.buffer('dirty').set(command_queue, dirty)
        self._zero_buffer(command_queue, fn.buffer('tile_max'))
        self._zero_buffer(command_queue, fn.buffer('tile_pos'))
        fn(2, 3, 10, 13)
        tile_max = fn.buffer('tile_max').get(command_queue)
        tile_pos = fn.buffer('tile_pos').get(command_queue)

        num_tiles_y, num_tiles_x = tile_max.shape
        assert_equal(18, num_tiles_y)
        assert_equal(18, num_tiles_x)
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Make sure updates don't happen where they are not meant to
                if x < 2 or x >= 10 or y < 3 or y >= 13:
                    assert_equal(0.0, tile_max[y, x])
                    assert_equal(0, tile_pos[y, x, 0])
                    assert_equal(0, tile_pos[y, x, 1])
                else:
                    y0 = y * template.tiley
                    x0 = x * template.tilex
                    y1 = min(y0 + template.tiley, dirty.shape[0])
                    x1 = min(x0 + template.tilex, dirty.shape[1])
                    tile = np.abs(dirty[y0:y1, x0:x1, 0])
                    pos = np.unravel_index(np.argmax(tile), tile.shape)
                    assert_equal(tile[pos], tile_max[y, x])
                    assert_equal(pos[0] + y0, tile_pos[y, x, 0])
                    assert_equal(pos[1] + x0, tile_pos[y, x, 1])
