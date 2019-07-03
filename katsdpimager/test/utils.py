"""Utilities for test code"""

import numpy as np


class RandomState(np.random.RandomState):
    """Extension of :class:`random.RandomState` with extra distributions"""
    def complex_normal(self, loc=0.0j, scale=1.0, size=None):
        """Circularly symmetric Gaussian in the Argand plane"""
        return self.normal(np.real(loc), scale, size) + 1j * self.normal(np.imag(loc), scale, size)

    def complex_uniform(self, low=0.0, high=1.0, size=None):
        """Uniform distribution over a rectangle in the complex plane.

        If low or high is purely real (by dtype, not by value), it is taken to
        be the boundary on both the real and imaginary extent.
        """
        if not np.iscomplexobj(low):
            low = np.asarray(low) * (1 + 1j)
        if not np.iscomplexobj(high):
            high = np.asarray(high) * (1 + 1j)
        return self.uniform(np.real(low), np.real(high), size) \
            + 1j * self.uniform(np.imag(low), np.imag(high), size)
