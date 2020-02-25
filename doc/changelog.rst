Changelog
=========

.. rubric:: 1.1

- Use direct prediction for the measurement equation, rather than FFT and
  degridding. This could potentially be slightly more accurate, although the
  images are not noticeably different. The old behaviour can be enabled with
  ``--degrid``, which could be faster if there are very large numbers of CLEAN
  components.
- Fix some bugs that could cause CUDA errors.
- Fix a bug that would cause a crash for some image sizes.
- A very small correction to parallactic angle calculations.
- More efficient loading of katdal datasets.

.. rubric:: 1.0

This is the first versioned release.
