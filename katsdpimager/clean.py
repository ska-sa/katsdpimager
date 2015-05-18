"""Deconvolution routines based on CLEAN"""

from __future__ import division, print_function
import numpy as np
import katsdpsigproc.accel as accel

#: Use only Stokes I to find peaks
CLEAN_I = 0
#: Use the sum of squares of available Stokes components to find peaks
CLEAN_SUMSQ = 1

class CleanHost(object):
    def __init__(self, image_parameters, clean_parameters, image, psf, model):
        self.clean_parameters = clean_parameters
        self.image_parameters = image_parameters
        self.image = image
        self.model = model
        self.psf = psf

    def _update_tile(self, tile_max, tile_pos, y, x):
        tile_size = self.clean_parameters.tile_size
        x0 = x * tile_size
        y0 = y * tile_size
        x1 = min(x0 + tile_size, self.image.shape[1])
        y1 = min(y0 + tile_size, self.image.shape[0])
        tile = self.image[y0:y1, x0:x1, ...]
        if self.clean_parameters.mode == CLEAN_I:
            tile_fn = np.abs(tile[..., 0])
        else:
            tile_fn = np.sum(tile * tile, axis=2)
        pos = np.unravel_index(np.argmax(tile_fn), tile_fn.shape)
        tile_max[y, x] = tile_fn[pos]
        tile_pos[y, x] = (pos[0] + y0, pos[1] + x0)

    def _subtract_psf(self, y, x):
        psf_x = self.psf.shape[1] // 2
        psf_y = self.psf.shape[0] // 2
        x0 = x - psf_x
        x1 = x0 + self.psf.shape[1]
        y0 = y - psf_y
        y1 = y0 + self.psf.shape[0]
        psf_x0 = 0
        psf_y0 = 0
        psf_x1 = self.psf.shape[1]
        psf_y1 = self.psf.shape[0]
        if x0 < 0:
            psf_x0 -= x0
            x0 = 0
        if y0 < 0:
            psf_y0 -= y0
            y0 = 0
        if x1 > self.image.shape[1]:
            psf_x1 -= (x1 - self.image.shape[1])
            x1 = self.image.shape[1]
        if y1 > self.image.shape[0]:
            psf_y1 -= (y1 - self.image.shape[0])
            y1 = self.image.shape[0]
        scale = self.clean_parameters.loop_gain * self.image[y, x]
        self.image[y0:y1, x0:x1, ...] -= scale * self.psf[psf_y0:psf_y1, psf_x0:psf_x1, ...]
        self.model[y, x] += scale
        return (y0, x0, y1, x1)

    def __call__(self, progress=None):
        tile_size = self.clean_parameters.tile_size
        ntiles = accel.divup(self.image_parameters.pixels, tile_size)
        tile_max = np.zeros((ntiles, ntiles), self.image_parameters.dtype)
        tile_pos = np.empty((ntiles, ntiles, 2), np.int32)
        for y in range(ntiles):
            for x in range(ntiles):
                self._update_tile(tile_max, tile_pos, y, x)
        for i in range(self.clean_parameters.minor):
            peak_tile = np.unravel_index(np.argmax(tile_max), tile_max.shape)
            peak = tile_max[peak_tile]
            peak_pos = tile_pos[peak_tile]
            (y0, x0, y1, x1) = self._subtract_psf(peak_pos[0], peak_pos[1])
            tile_y0 = y0 // tile_size
            tile_x0 = x0 // tile_size
            tile_y1 = accel.divup(y1, tile_size)
            tile_x1 = accel.divup(x1, tile_size)
            for y in range(tile_y0, tile_y1):
                for x in range(tile_x0, tile_x1):
                    self._update_tile(tile_max, tile_pos, y, x)
            if progress:
                progress.next()
        if progress:
            progress.finish()
