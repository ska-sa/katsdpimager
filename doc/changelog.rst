Changelog
=========

.. rubric:: 1.5

This version mainly focuses on performance. Improvements of 5× have been
observed in the MeerKAT pipeline.

Other changes:

- MVF (katdal datasets) older that v4 (such as KAT-7) data are no longer
  supported. Use the katdal :program:`mvftoms` tool to convert them to Measurement
  Sets instead (this is necessary anyway since they are uncalibrated).
- Add more internal profiling.
- Disable compression of the temporary file, as it substantially reduced
  performance. This does mean that temporary space requirements will increase,
  and that performance could be harmed if the temporary filesystem is slow.
- When using an RFI mask with MeerKAT data, exclude channels that are only
  partially covered by the mask.
- Suppress an unwanted `astropy warning`_.

.. _astropy warning: https://github.com/astropy/astropy/issues/10365

.. rubric:: 1.4

- Fix estimation of image noise (estimates were about 2.7× too high).
- Add command-line arguments as HISTORY keywords in the output FITS files.
- New command-line option ``-i rfi-mask`` for katdal data sets, to skip
  channels that the observation indicates are known to be affected by RFI.
- Workaround for a small number of MeerKAT observations that had the
  incorrect stream type in the katdal data set.
- Fix help text for :option:`--stokes` parameter.
- Work around a bug in h5py 3.0+ that causes segmentation faults.
- There are a number of changes that are only visible when run as part of the
  MeerKAT imaging pipeline, and not when used as a standalone imager:

  - Various metrics are stored in the telescope state, and can be extracted to
    generate a report.
  - Profiling information can be stored in the telescope state.
  - If the imager crashes after processing a subset of channels, they will be
    skipped the next time.

.. rubric:: 1.3

- Add support for primary beam correction (MeerKAT only).

.. rubric:: 1.2

- Add extra FITS headers to the output; particularly those needed to compute
  Doppler corrections.
- Improve performance when loading data from katdal datasets.
- Add script to generate a video from FITS files.

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
