# -*- coding: utf-8 -*-

r"""
Computation of density weights, for uniform and robust weighting. This is done
in stages:

1. The statistical weights are gridded, *without* any convolution. The subgrid
coordinates are still passed so that the same data layouts can be shared with
convolutional gridding, but they can be ignored.

2. The total statistical weight per cell is converted into a density weight.
If :math:`W_i` is the total statistical weight for a cell, then a uniform
weight is computed as :math:`1 / W` and a robust weight as :math:`1 / (WS^2 +
1)`, where

.. math:: S^2 = \frac{(5\cdot 10^{-R})^2 \sum W_i}{\sum W_i^2}

and :math:`R` is the robustness parameter.

3. During gridding, as visibilities are loaded the density weights are looked
up and multiplied in. The visibilities are already pre-weighted by the
statistical weights.

Note that all these steps use compressed visibilities. This works because the
density weights are constant for a grid cell, and thus constant across the
original visibilities that contribute to a compressed visibility.

Weights are processed separately per polarization. However, the
robustness parameter :math:`S` is computed for the first polarization
(generally Stokes I) and used for all polarizations, to avoid qualitatively
different beam shapes for the different polarizations.
"""

import pkg_resources
import numpy as np
from katsdpsigproc import accel, fill


NATURAL = 0
UNIFORM = 1
ROBUST = 2


class GridWeightsTemplate(object):
    def __init__(self, context, num_polarizations, tuning=None):
        self.context = context
        self.num_polarizations = num_polarizations
        # TODO: autotuning
        self.wgs = 256
        parameters = {
            'num_polarizations': num_polarizations,
            'wgs': self.wgs
        }
        self.program = accel.build(
            context, "imager_kernels/grid_weights.mako", parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return GridWeights(self, *args, **kwargs)


class GridWeights(accel.Operation):
    def __init__(self, template, command_queue, grid_shape, max_vis, allocator=None):
        super(GridWeights, self).__init__(command_queue, allocator)
        self.template = template
        if grid_shape[0] != self.template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        if grid_shape[1] % 2 or grid_shape[2] % 2:
            raise ValueError('Odd-sized grid not currently supported')
        self.max_vis = max_vis
        self.slots['grid'] = accel.IOSlot(grid_shape, np.float32)
        self.slots['uv'] = accel.IOSlot(
            (max_vis, accel.Dimension(4, exact=True)), np.int16)
        self.slots['weights'] = accel.IOSlot(
            (max_vis, accel.Dimension(self.template.num_polarizations, exact=True)), np.float32)
        self._num_vis = 0
        self._kernel = self.template.program.get_kernel('grid_weights')

    @property
    def num_vis(self):
        return self._num_vis

    @num_vis.setter
    def num_vis(self, n):
        """Change the number of actual visibilities stored in the buffers."""
        if n < 0 or n > self.max_vis:
            raise ValueError('Number of visibilities {} is out of range 0..{}'.format(
                n, self.max_vis))
        self._num_vis = n

    def _run(self):
        grid = self.buffer('grid')
        uv = self.buffer('uv')
        weights = self.buffer('weights')
        half_v = grid.shape[1] // 2
        half_u = grid.shape[2] // 2
        address_bias = half_v * grid.padded_shape[2] + half_u
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                uv.buffer,
                weights.buffer,
                np.int32(address_bias),
                np.int32(self._num_vis)
            ],
            global_size=(accel.roundup(self._num_vis, self.template.wgs),),
            local_size=(self.template.wgs,)
        )

    def parameters(self):
        return {
            'wgs': self.template.wgs,
            'num_polarizations': self.template.num_polarizations,
            'max_vis': self.max_vis
        }


class DensityWeightsTemplate(object):
    def __init__(self, context, num_polarizations, tuning=None):
        self.context = context
        self.num_polarizations = num_polarizations
        # TODO: autotuning
        self.wgs_x = 16
        self.wgs_y = 16
        parameters = {
            'num_polarizations': num_polarizations,
            'wgs_x': self.wgs_x,
            'wgs_y': self.wgs_y
        }
        self.program = accel.build(
            context, "imager_kernels/density_weights.mako", parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return DensityWeights(self, *args, **kwargs)


class DensityWeights(accel.Operation):
    def __init__(self, template, command_queue, grid_shape, allocator=None):
        super(DensityWeights, self).__init__(command_queue, allocator)
        self.template = template
        if grid_shape[0] != self.template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        # Default to uniform
        self.a = 1.0
        self.b = 0.0
        self.slots['grid'] = accel.IOSlot(
            (grid_shape[0],
             accel.Dimension(grid_shape[1], self.template.wgs_y),
             accel.Dimension(grid_shape[2], self.template.wgs_x)),
            np.float32)
        self._kernel = self.template.program.get_kernel('density_weights')

    def _run(self):
        grid = self.buffer('grid')
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                np.float32(self.a),
                np.float32(self.b)
            ],
            global_size=(accel.roundup(grid.shape[2], self.template.wgs_x),
                         accel.roundup(grid.shape[1], self.template.wgs_y)),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )

    def parameters(self):
        return {
            'wgs_x': self.template.wgs_x,
            'wgs_y': self.template.wgs_y,
            'num_polarizations': self.template.num_polarizations,
        }


class MeanWeightTemplate(object):
    def __init__(self, context, tuning=None):
        self.context = context
        # TODO: autotuning
        self.wgs_x = 16
        self.wgs_y = 16
        parameters = {
            'wgs_x': self.wgs_x,
            'wgs_y': self.wgs_y
        }
        self.program = accel.build(
            context, "imager_kernels/mean_weight.mako", parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return MeanWeight(self, *args, **kwargs)


class MeanWeight(accel.Operation):
    def __init__(self, template, command_queue, grid_shape, allocator=None):
        super(MeanWeight, self).__init__(command_queue, allocator)
        self.template = template
        self.slots['grid'] = accel.IOSlot(
            (grid_shape[0],
             accel.Dimension(grid_shape[1], self.template.wgs_y),
             accel.Dimension(grid_shape[2], self.template.wgs_x)),
            np.float32)
        self.slots['sums'] = accel.IOSlot((2,), np.float32)
        self._kernel = template.program.get_kernel('mean_weight')
        self._sums_host = accel.HostArray((2,), np.float32, context=self.template.context)

    def _run(self):
        grid = self.buffer('grid')
        sums = self.buffer('sums')
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                sums.buffer,
                grid.buffer,
                np.int32(grid.padded_shape[-1])
            ],
            global_size=(accel.roundup(grid.shape[2], self.template.wgs_x),
                         accel.roundup(grid.shape[1], self.template.wgs_y)),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )
        if isinstance(sums, accel.SVMArray):
            self.command_queue.finish()
        sums.get(self.command_queue, self._sums_host)
        return self._sums_host[1] / self._sums_host[0]


class WeightsTemplate(object):
    def __init__(self, context, weight_type, num_polarizations, grid_weights_tuning=None, mean_weight_tuning=None, density_weights_tuning=None):
        self.context = context
        self.weight_type = weight_type
        if weight_type == NATURAL:
            self.grid_weights = None
            self.mean_weight = None
            self.density_weights = None
            self.fill = fill.FillTemplate(context, np.float32, 'float')
        else:
            self.grid_weights = GridWeightsTemplate(context, num_polarizations, tuning=grid_weights_tuning)
            if weight_type == ROBUST:
                self.mean_weight = MeanWeightTemplate(context, tuning=mean_weight_tuning)
            else:
                self.mean_weight = None
            self.density_weights = DensityWeightsTemplate(context, num_polarizations, tuning=density_weights_tuning)
            self.fill = None

    def instantiate(self, *args, **kwargs):
        return Weights(self, *args, **kwargs)


class Weights(accel.OperationSequence):
    def __init__(self, template, command_queue, grid_shape, max_vis, allocator=None):
        self.template = template
        operations = []
        compounds = {'grid': []}

        if template.grid_weights is not None:
            self._grid_weights = template.grid_weights.instantiate(
                    command_queue, grid_shape, max_vis, allocator)
            operations.append(('grid_weights', self._grid_weights))
            compounds['grid'].append('grid_weights:grid')
            compounds['uv'] = ['grid_weights:uv']
            compounds['weights'] = ['grid_weights:weights']
        else:
            self._grid_weights = None

        if template.fill is not None:
            self._fill = template.fill.instantiate(
                    command_queue, grid_shape, allocator)
            self._fill.set_value(1)
            operations.append(('fill', self._fill))
            compounds['grid'].append('fill:data')
        else:
            self._fill = None

        if template.mean_weight is not None:
            self._mean_weight = template.mean_weight.instantiate(
                    command_queue, grid_shape, allocator)
            operations.append(('mean_weight', self._mean_weight))
            compounds['grid'].append(('mean_weight:grid'))
            self.robustness = 0.0
        else:
            self._mean_weight = None
            self.robustness = None

        if template.density_weights is not None:
            self._density_weights = template.density_weights.instantiate(
                    command_queue, grid_shape, allocator)
            operations.append(('density_weights', self._density_weights))
            compounds['grid'].append(('density_weights:grid'))
        else:
            self._density_weights = None

        super(Weights, self).__init__(command_queue, operations, compounds, allocator=allocator)

    def _run(self):
        raise NotImplementedError('Weights should not be used as a callable')

    def clear(self):
        self.ensure_all_bound()
        # If self._fill is set, it is not necessary to clear first because
        # finalize will set all relevant values.
        if self._fill is not None:
            self.buffer('grid').zero(self.command_queue)

    def grid(self, num_vis):
        self.ensure_all_bound()
        if self._grid_weights is not None:
            self._grid_weights.num_vis = num_vis
            return self._grid_weights()

    def finalize(self):
        self.ensure_all_bound()
        if self._mean_weight is not None:
            # It must be robust weighting
            mean_weight = self._mean_weight()
            S2 = (5 * 10**(-self.robustness))**2 / mean_weight
            self._density_weights.a = S2
            self._density_weights.b = 1.0
        if self._density_weights is not None:
            self._density_weights()
        if self._fill is not None:
            self._fill()
