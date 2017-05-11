"""Tests for :py:mod:`katsdpimager.clean`"""

from __future__ import division, print_function
import numpy as np
import scipy.signal
from .. import clean
from katsdpsigproc.test.test_accel import device_test
from nose.tools import *


class TestPsfPatch(object):
    """Tests for :class:`~katsdpimager.clean.PsfPatch`."""
    @device_test
    def setup(self, context, command_queue):
        self.template = clean.PsfPatchTemplate(context, np.float32, 4)
        self.fn = self.template.instantiate(command_queue, (4, 206, 304))
        self.fn.ensure_all_bound()
        self.psf = self.fn.buffer('psf')
        self.psf_host = self.psf.empty_like()
        self.psf_host.fill(0.0)
        self.psf_host[:, 103, 152] = 1.0    # Set central peak

    def _test(self, threshold=0.01):
        """Run the function using psf_host, returning the resulting patch size."""
        self.psf.set(self.fn.command_queue, self.psf_host)
        return self.fn(threshold)

    def test_peak_only(self):
        assert_equal((4, 2, 2), self._test())

    def test_low_corner(self):
        self.psf_host[0, 0, 0] = 0.1
        assert_equal((4, 206, 304), self._test())

    def test_high_corner(self):
        self.psf_host[3, 205, 303] = 1.0
        assert_equal((4, 206, 304), self._test())

    def test_1d(self):
        target = self.psf_host[1, 0, :152]
        target[:] = np.arange(152)
        threshold = 50.5
        box = self._test(threshold=threshold)
        hw = box[2] // 2
        assert_equal(0, sum(target[:-hw] >= threshold))
        assert_less(0, sum(target[:-hw + 2] >= threshold))


class TestClean(object):
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
        image_shape = (4, 567, 456)
        border = 65
        rs = np.random.RandomState(seed=1)
        template = clean._UpdateTilesTemplate(context, np.float32, 4, clean.CLEAN_I)
        fn = template.instantiate(command_queue, image_shape, border)
        fn.ensure_all_bound()
        # TODO: this could lead to ties, which will fail the test
        dirty = rs.standard_normal(image_shape).astype(np.float32)
        fn.buffer('dirty').set(command_queue, dirty)
        self._zero_buffer(command_queue, fn.buffer('tile_max'))
        self._zero_buffer(command_queue, fn.buffer('tile_pos'))
        fn(135, 161, 385, 450)
        tile_max = fn.buffer('tile_max').get(command_queue)
        tile_pos = fn.buffer('tile_pos').get(command_queue)

        num_tiles_y, num_tiles_x = tile_max.shape
        assert_equal(14, num_tiles_y)
        assert_equal(11, num_tiles_x)
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Make sure updates don't happen where they are not meant to
                if x < 2 or x >= 10 or y < 3 or y >= 13:
                    assert_equal(0.0, tile_max[y, x])
                    assert_equal(0, tile_pos[y, x, 0])
                    assert_equal(0, tile_pos[y, x, 1])
                else:
                    y0 = y * template.tiley + border
                    x0 = x * template.tilex + border
                    y1 = min(y0 + template.tiley, dirty.shape[1] - border)
                    x1 = min(x0 + template.tilex, dirty.shape[2] - border)
                    tile = np.abs(dirty[0, y0:y1, x0:x1])
                    pos = np.unravel_index(np.argmax(tile), tile.shape)
                    assert_equal(tile[pos], tile_max[y, x])
                    assert_equal(pos[0] + y0, tile_pos[y, x, 0])
                    assert_equal(pos[1] + x0, tile_pos[y, x, 1])

    @device_test
    def test_find_peak(self, context, command_queue):
        image_shape = (4, 256, 256)
        tile_shape = (72, 67)
        template = clean._FindPeakTemplate(context, np.float32, 4)
        fn = template.instantiate(command_queue, image_shape, tile_shape)
        fn.ensure_all_bound()
        # TODO: this could lead to ties, which will fail the test
        rs = np.random.RandomState(seed=1)
        dirty = rs.uniform(1.0, 2.0, image_shape).astype(np.float32)
        tile_max = rs.uniform(1.0, 2.0, tile_shape).astype(np.float32)
        # Set tile_pos so that each pos indexes the corresponding position in
        # the dirty image, to make testing easier.
        tile_pos = np.array(
            [[[y, x] for x in range(tile_shape[1])] for y in range(tile_shape[0])], np.int32)
        fn.buffer('tile_max').set(command_queue, tile_max)
        fn.buffer('tile_pos').set(command_queue, tile_pos)
        fn.buffer('dirty').set(command_queue, dirty)
        fn()
        peak_value = fn.buffer('peak_value').get(command_queue)
        peak_pos = fn.buffer('peak_pos').get(command_queue)
        peak_pixel = fn.buffer('peak_pixel').get(command_queue)
        best = np.unravel_index(np.argmax(tile_max), tile_max.shape)
        assert_equal(tile_max[best], peak_value[0])
        np.testing.assert_array_equal(tile_pos[best], peak_pos)
        np.testing.assert_array_equal(dirty[:, peak_pos[0], peak_pos[1]], peak_pixel)

    @device_test
    def test_subtract_psf(self, context, command_queue):
        loop_gain = 0.25
        image_shape = (4, 200, 345)
        psf_shape = (4, 72, 131)
        pos = (170, 59)    # chosen so that the PSF will be clipped to the image
        rs = np.random.RandomState(seed=1)
        dirty = rs.standard_normal(image_shape).astype(np.float32)
        psf = rs.standard_normal(psf_shape).astype(np.float32)
        expected = dirty.copy()
        peak_pixel = dirty[:, pos[0], pos[1]]
        expected[:, 134:200, 0:125] -= loop_gain * peak_pixel[:, np.newaxis, np.newaxis] * psf[:, :66, 6:]

        template = clean._SubtractPsfTemplate(context, np.float32, 4)
        fn = template.instantiate(command_queue, loop_gain, image_shape, psf_shape)
        fn.ensure_all_bound()
        fn.buffer('dirty').set(command_queue, dirty)
        fn.buffer('psf').set(command_queue, psf)
        fn.buffer('peak_pixel').set(command_queue, peak_pixel)
        self._zero_buffer(command_queue, fn.buffer('model'))
        fn(pos)
        dirty = fn.buffer('dirty').get(command_queue)
        model = fn.buffer('model').get(command_queue)
        np.testing.assert_allclose(expected, dirty, atol=1e-4)
        np.testing.assert_allclose(loop_gain * peak_pixel, model[:, pos[0], pos[1]])
