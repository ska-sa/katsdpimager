"""Tests for :mod:`katsdpimager.weight`."""

import numpy as np
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test

from katsdpimager import weight


class TestGridWeights:
    def setup(self):
        self.grid_shape = (4, 100, 200)
        self.uv = np.array([
            [-10, 5, 0, 0],
            [23, 17, 0, 0],
            [-10, 5, 0, 0],
            [-10, 5, 0, 0],
            [-10, 6, 0, 0],
            [-11, 5, 0, 0]], np.int16)
        self.weights = np.array([
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [4.0, 40.0, 400.0, 4000.0],
            [8.0, 80.0, 800.0, 8000.0],
            [16.0, 160.0, 1600.0, 16000.0],
            [32.0, 320.0, 3200.0, 32000.0]], np.float32)

    def _set_partial(self, command_queue, buffer, data):
        """Load a device array from a smaller host array. The rest of the
        array is filled with random values to ensure that the algorithm
        doesn't depend on them being zero.
        """
        rs = np.random.RandomState(1)
        host = buffer.empty_like()
        host[:] = rs.uniform(size=host.shape)
        sub_rect = tuple(np.s_[0 : x] for x in data.shape)
        host[sub_rect] = data
        buffer.set(command_queue, host)

    @device_test
    def test(self, context, command_queue):
        template = weight.GridWeightsTemplate(context, 4)
        fn = template.instantiate(command_queue, self.grid_shape, 1000)
        fn.ensure_all_bound()
        self._set_partial(command_queue, fn.buffer('uv'), self.uv)
        self._set_partial(command_queue, fn.buffer('weights'), self.weights)
        fn.buffer('grid').zero(command_queue)
        fn.num_vis = len(self.uv)
        fn()
        actual = fn.buffer('grid').get(command_queue)
        expected = np.zeros(self.grid_shape, np.float32)
        for i in range(4):
            expected[i, 55, 90] = 13 * 10**i
            expected[i, 67, 123] = 2 * 10**i
            expected[i, 56, 90] = 16 * 10**i
            expected[i, 55, 89] = 32 * 10**i
        np.testing.assert_equal(expected, actual)


class TestDensityWeights:
    def setup(self):
        rs = np.random.RandomState(1)
        self.grid_shape = (4, 50, 107)
        self.data = np.zeros(self.grid_shape, np.float32)
        self.expected = np.zeros(self.grid_shape, np.float32)
        n_indices = 100
        flat_indices = rs.choice(self.data.size, n_indices, replace=False)
        sum_w = 0.0
        sum_dw = 0.0
        sum_d2w = 0.0
        for index in flat_indices:
            w = rs.uniform(low=0.1, high=2.0)
            d = 1.0 / (2.5 * w + 1.75)
            self.data.flat[index] = w
            self.expected.flat[index] = d
            if index < self.data[0].size:
                sum_w += w
                sum_dw += d * w
                sum_d2w += d**2 * w
        self.normalized_rms = np.sqrt(sum_d2w * sum_w) / sum_dw

    @device_test
    def test(self, context, command_queue):
        template = weight.DensityWeightsTemplate(context, 4)
        fn = template.instantiate(command_queue, self.grid_shape)
        fn.ensure_all_bound()
        fn.a = 2.5
        fn.b = 1.75
        # The 'grid' is set in a roundabout way to fill the padding with
        # non-zero values
        tmp_data = fn.buffer('grid').empty_like()
        accel.HostArray.padded_view(tmp_data).fill(3)
        tmp_data[:] = self.data
        fn.buffer('grid').set(command_queue, tmp_data)
        normalized_rms = fn()
        actual = fn.buffer('grid').get(command_queue)
        np.testing.assert_allclose(self.expected, actual, 1e-5, 1e-5)
        np.testing.assert_allclose(self.normalized_rms, normalized_rms, 1e-6)


class TestMeanWeight:
    def setup(self):
        rs = np.random.RandomState(1)
        self.grid_shape = (4, 50, 107)
        self.data = rs.uniform(size=self.grid_shape).astype(np.float32)

    @device_test
    def test(self, context, command_queue):
        template = weight.MeanWeightTemplate(context)
        fn = template.instantiate(command_queue, self.grid_shape)
        fn.ensure_all_bound()
        fn.buffer('grid').set(command_queue, self.data)
        actual = fn()
        pol0 = self.data[0]
        expected = np.sum(pol0 * pol0) / np.sum(pol0)
        np.testing.assert_allclose(expected, actual, rtol=1e-5)
