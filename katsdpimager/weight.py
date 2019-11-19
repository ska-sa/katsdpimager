# -*- coding: utf-8 -*-

r"""
Computation of density weights, for uniform and robust weighting. Refer to
[Bri95]_ for definitions. Computation of weights is done in up to four
stages:

1. The statistical weights are gridded, *without* any convolution. The subgrid
   coordinates are still passed so that the same data layouts can be shared with
   convolutional gridding, but they are be ignored.

2. For robust weighting, an "average" statistical weight is computed as

   .. math:: \overline{W} = \frac{\sum W_i^2}{\sum W_i},

   where the sums are over grid cells.

3. The total statistical weight per cell is converted into a density weight.
   If :math:`W_i` is the total statistical weight for a cell, then a uniform
   weight is computed as :math:`1 / W_i` and a robust weight as
   :math:`1 / (W_i S^2 + 1)`, where

   .. math:: S^2 = \overline{W}(5\cdot 10^{-R})^2

   and :math:`R` is the robustness parameter. The conversion formula for robust
   weighting is taken from wsclean.

4. During gridding, as visibilities are loaded the density weights are looked
   up and multiplied in. The visibilities are already pre-weighted by the
   statistical weights.

Note that all these steps use compressed visibilities. This works because the
density weights are constant for a grid cell, and thus constant across the
original visibilities that contribute to a compressed visibility.

Weights are processed separately per polarization. However, the
robustness parameter :math:`S` is computed for the first polarization
(generally Stokes I) and used for all polarizations, to avoid qualitatively
different beam shapes for the different polarizations.

.. [Bri95] Briggs, D. S. 1995. High fidelity deconvolution of moderately
   resolved sources. PhD Thesis, The New Mexico Institute of Mining and Technology.
   http://www.aoc.nrao.edu/dissertations/dbriggs/

.. include:: macros.rst
"""

import enum

import pkg_resources
import numpy as np
from katsdpsigproc import accel, fill


class WeightType(enum.Enum):
    NATURAL = 0
    UNIFORM = 1
    ROBUST = 2


class GridWeightsTemplate:
    """Template for accumulating weights onto a grid.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    num_polarizations : int
        Number of polarizations to grid
    tuning : dict, optional
        Tuning parameters (unused)
    """
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
    """Instantiation of :class:`GridWeightsTemplate`.

    The user should set some prefix of the `uv` and `weights` slots, set
    :attr:`num_vis` to indicate how many weights are provided, and then call
    the operation to grid them.

    .. rubric:: Slots

    **uv** : array of shape `max_vis` × 4, int16
        UV coordinates of the grid points. The coordinates are biased by half
        the grid size, so that (0, 0) refers to the centre of the grid.
    **weights** : array of shape `max_vis` × `num_polarizations`, float32
        Weights to accumulate
    **grid** : array of shape `num_polarizations` × height × width, float32
        Accumulated weights

    Parameters
    ----------
    template : :class:`GridWeightsTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    grid_shape : tuple of ints
        Shape for the grid, (polarizations, height, width)
    max_vis : int
        Maximum number of weights that can be gridded in one pass
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        if the grid width or height is odd, or if the number of polarizations
        does not match `template`.
    """
    def __init__(self, template, command_queue, grid_shape, max_vis, allocator=None):
        super().__init__(command_queue, allocator)
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
        """The actual number of weights to grid"""
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


class DensityWeightsTemplate:
    """Template for converting cell sum of statistical weights to density weights.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    num_polarizations : int
        Number of polarizations to grid
    tuning : dict, optional
        Tuning parameters (unused)
    """
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
    """Instantiation of :class:`DensityWeightsTemplate`. This operation
    modifies the grid in-place. The type of weights is controlled by two
    attributes, :attr:`a` and :attr:`b`. For an input :math:`W`, the result
    is :math:`1 / (aW + b)`. This allows for natural, uniform or robust
    weights. If not altered, the default gives uniform weights.

    It also computes and returns the normalized thermal RMS, as described in
    equation 3.5 of [Bri95]_.

    .. rubric:: Slots

    **grid** : array of shape `num_polarizations` × height × width, float32
        On input, the sum of statistical weights per cell; on output, the density
        weight for each cell.

    Parameters
    ----------
    template : :class:`DensityWeightsTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    grid_shape : tuple of ints
        Shape for the grid, (polarizations, height, width)
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, grid_shape, allocator=None):
        super().__init__(command_queue, allocator)
        self.template = template
        if grid_shape[0] != self.template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        self.a = 1.0
        self.b = 0.0
        self.slots['grid'] = accel.IOSlot(
            (grid_shape[0],
             accel.Dimension(grid_shape[1], self.template.wgs_y),
             accel.Dimension(grid_shape[2], self.template.wgs_x)),
            np.float32)
        self.slots['sums'] = accel.IOSlot((3,), np.float32)
        self._kernel = self.template.program.get_kernel('density_weights')
        self._sums_host = accel.HostArray((3,), np.float32, context=self.template.context)

    def _run(self):
        grid = self.buffer('grid')
        sums = self.buffer('sums')
        sums.zero(self.command_queue)
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                sums.buffer,
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                np.int32(grid.shape[2]),
                np.int32(grid.shape[1]),
                np.float32(self.a),
                np.float32(self.b)
            ],
            global_size=(accel.roundup(grid.shape[2], self.template.wgs_x),
                         accel.roundup(grid.shape[1], self.template.wgs_y)),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )
        if isinstance(sums, accel.SVMArray):
            self.command_queue.finish()
        sums.get(self.command_queue, self._sums_host)
        return np.sqrt(self._sums_host[2] * self._sums_host[0]) / self._sums_host[1]

    def parameters(self):
        return {
            'a': self.a,
            'b': self.b,
            'wgs_x': self.template.wgs_x,
            'wgs_y': self.template.wgs_y,
            'num_polarizations': self.template.num_polarizations,
        }


class MeanWeightTemplate:
    """Template for computing the "mean weight", as defined by equation 3.17 in
    [Bri95]_. The input is the gridded statistical weights. The kernel outputs the
    sum of squared cell weights and the sum of weights, and the Python code
    computes the mean weight from these. Only the first polarization is used.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    tuning : dict, optional
        Tuning parameters (unused)
    """
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
    """Instantiation of :class:`MeanWeightTemplate`.

    .. rubric:: Slots

    **grid** : array of shape `num_polarizations` × height × width, float32
        The sum of statistical weights per cell

    Parameters
    ----------
    template : :class:`MeanWeightTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    grid_shape : tuple of ints
        Shape for the grid, (polarizations, height, width)
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, grid_shape, allocator=None):
        super().__init__(command_queue, allocator)
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
        self.command_queue.enqueue_zero_buffer(sums.buffer)
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                sums.buffer,
                grid.buffer,
                np.int32(grid.padded_shape[-1]),
                np.int32(grid.shape[2]),
                np.int32(grid.shape[1])
            ],
            global_size=(accel.roundup(grid.shape[2], self.template.wgs_x),
                         accel.roundup(grid.shape[1], self.template.wgs_y)),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )
        if isinstance(sums, accel.SVMArray):
            self.command_queue.finish()
        sums.get(self.command_queue, self._sums_host)
        return self._sums_host[1] / self._sums_host[0]


class WeightsTemplate:
    """Compound template for computing imaging weights.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    weight_type : :class:`WeightType`
        Weighting method
    num_polarizations : int
        Number of polarizations
    grid_weights_tuning, mean_weights_tuning, density_weights_tuning : dict, optional
        Tuning parameters passed to :class:`GridWeightsTemplate`,
        :class:`MeanWeightsTemplate` and :class:`DensityWeightsTemplate` respectively
    """
    def __init__(self, context, weight_type, num_polarizations,
                 grid_weights_tuning=None, mean_weight_tuning=None,
                 density_weights_tuning=None):
        self.context = context
        self.weight_type = weight_type
        if weight_type == WeightType.NATURAL:
            self.grid_weights = None
            self.mean_weight = None
            self.density_weights = None
            self.fill = fill.FillTemplate(context, np.float32, 'float')
        else:
            self.grid_weights = GridWeightsTemplate(context, num_polarizations,
                                                    tuning=grid_weights_tuning)
            if weight_type == WeightType.ROBUST:
                self.mean_weight = MeanWeightTemplate(context, tuning=mean_weight_tuning)
            else:
                self.mean_weight = None
            self.density_weights = DensityWeightsTemplate(context, num_polarizations,
                                                          tuning=density_weights_tuning)
            self.fill = None

    def instantiate(self, *args, **kwargs):
        return Weights(self, *args, **kwargs)


class Weights(accel.OperationSequence):
    """Instantiation of :class:`WeightsTemplate`. The steps to use it are:

     1. If using :const:`WeightType.ROBUST` weighting, set :attr:`robustness`.
     2. Call :meth:`clear`.
     3. For each batch of weights, call :meth:`grid`.
     4. Call :meth:`finalize`, which returns the normalized thermal RMS.

    .. rubric:: Slots

    **uv** : array of shape `max_vis` × 4, int16
        UV coordinates of the grid points. The coordinates are biased by half
        the grid size, so that (0, 0) refers to the centre of the grid.
    **weights** : array of shape `max_vis` × `num_polarizations`, float32
        Weights to accumulate
    **grid** : array of shape `num_polarizations` × height × width, float32
        Accumulated weights

    .. note::

        The **uv** and **weights** slots will be absent if the template was
        created with :const:`WeightType.NATURAL` weighting.

    Parameters
    ----------
    command_queue : |CommandQueue|
        Command queue for the operation
    grid_shape : tuple of ints
        Shape for the grid, (polarizations, height, width)
    max_vis : int
        Maximum number of weights that can be gridded in one pass
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Attributes
    ----------
    robustness : float
        Parameter for robust weighting
    """
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

        super().__init__(command_queue, operations, compounds, allocator=allocator)

    def _run(self):
        raise NotImplementedError('Weights should not be used as a callable')

    def clear(self):
        self.ensure_all_bound()
        # If self._fill is set, it is not necessary to clear first because
        # finalize will set all relevant values.
        if self._fill is None:
            self.buffer('grid').zero(self.command_queue)

    def grid(self, uv, weights):
        self.ensure_all_bound()
        if self._grid_weights is not None:
            N = len(uv)
            self._grid_weights.num_vis = N
            self.buffer('uv')[:N, 0:2] = uv
            self.buffer('weights')[:N] = weights
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
            normalized_rms = self._density_weights()
        else:
            normalized_rms = 1.0
        if self._fill is not None:
            self._fill()
        return normalized_rms


class WeightsHost:
    """Equivalent to :class:`Weights` that runs on the host.

    Parameters
    ----------
    weight_type : :class:`WeightType`
        Weighting method
    weights_grid : array-like, float
        Grid for weights

    Attributes
    ----------
    robustness : float
        Parameter for robust weighting
    """
    def __init__(self, weight_type, weights_grid):
        self.weight_type = weight_type
        self.robustness = 0.0
        self.weights_grid = weights_grid
        assert weights_grid.shape[1] % 2 == 0 and weights_grid.shape[2] % 2 == 0, \
            "Only even-sized grids are currently supported"

    def clear(self):
        if self.weight_type != WeightType.NATURAL:
            self.weights_grid.fill(0)

    def grid(self, uv, weights):
        shape = self.weights_grid.shape
        # Bias uv coordinates to grid center
        uv += np.array([[shape[2] // 2, shape[1] // 2]], np.int16)
        for i in range(len(uv)):
            self.weights_grid[:, uv[i, 1], uv[i, 0]] += weights[i, :]

    def finalize(self):
        if self.weight_type == WeightType.NATURAL:
            self.weights_grid.fill(1)
            normalized_rms = 1.0
        elif self.weight_type == WeightType.UNIFORM:
            sum_w = np.sum(self.weights_grid[0])
            sum_dw = np.count_nonzero(self.weights_grid[0])
            # Force density weights to be zero for cells with no visibilities
            self.weights_grid[self.weights_grid == 0] = np.inf
            np.reciprocal(self.weights_grid, out=self.weights_grid)
            # d^2w == d, because d is 1/w
            sum_d2w = np.sum(self.weights_grid[0])
            normalized_rms = np.sqrt(sum_d2w * sum_w) / sum_dw
        elif self.weight_type == WeightType.ROBUST:
            sum_sq = np.dot(self.weights_grid[0].flat, self.weights_grid[0].flat)
            sum = np.sum(self.weights_grid[0])
            mean_weight = sum_sq / sum
            S2 = (5 * 10**(-self.robustness))**2 / mean_weight
            old_weights0 = self.weights_grid[0].copy()
            # Force density weights to be zero for cells with no visibilities
            self.weights_grid[self.weights_grid == 0] = np.inf
            np.reciprocal(self.weights_grid * S2 + 1, out=self.weights_grid)
            sum_w = np.sum(old_weights0)
            sum_dw = np.sum(self.weights_grid[0] * old_weights0)
            sum_d2w = np.sum(self.weights_grid[0]**2 * old_weights0)
            normalized_rms = np.sqrt(sum_d2w * sum_w) / sum_dw
        else:
            raise ValueError('Unknown weight_type {}'.format(self.weight_type))
        return normalized_rms
