User guide
----------
katsdpimager is a GPU-accelerated spectral line imager for radio astronomy.

Installation
============

Only Python 3.5+ is supported. You will also need an NVIDIA GPU with the
CUDA toolkit and drivers. At least version 6.0 is required, but testing is only
done with 10.0 and later (but see :ref:`cpu`).

To install with support for Measurement Sets, run

.. code-block:: sh

   pip install 'katsdpimager[ms]'

To install with support for katdal_ data sets, run

.. code-block:: sh

   pip install 'katsdpimager[katdal]'

and to support both, use

.. code-block:: sh

   pip install 'katsdpimager[katdal,ms]'

After these steps, you should be able to run ``imager.py``, but see following
sections for details of how to run it.

Binary installation
^^^^^^^^^^^^^^^^^^^
If you have a sufficiently recent version of :program:`pip` and a Linux x86-64
system the above will install a binary wheel, and all requirements will be
handled by :program:`pip`.

From source
^^^^^^^^^^^
If there is no binary wheel available for your platform, then there are
additional requirements as some C code will need to be compiled:

 - libhdf5, including development headers (``libhdf5-dev`` in Debian/Ubuntu);
 - `Eigen3`_ (``libeigen3-dev`` in Debian/Ubuntu);
 - A C++ compiler such as GCC;
 - Boost headers.

.. _Eigen3: http://eigen.tuxfamily.org

File formats
============
Two input formats are supported: `Measurement Sets`_ and KAT/MeerKAT
data sets read by `katdal`_. Output is to `FITS`_ files. The input can contain
multiple channels, but a separate FITS file is written for each channel.

.. _Measurement sets: http://casa.nrao.edu/Memos/229.html
.. _katdal: https://katdal.readthedocs.io/
.. _FITS: http://fits.gsfc.nasa.gov/fits_documentation.html

The input file format is detected by extension, so a Measurement Set *must*
have the suffix ``.ms`` and a katdal data set must have the suffix ``.h5`` or
``.rdb``.

Command-line options
====================
The simplest possible incantation is

.. code-block:: sh

   imager.py input.ms output%05d.fits

Imaging is not a one-size-fits-all process, and so this is unlikely
to do what you need. Nevertheless, it tries to make reasonable guesses about
resolution and image size based on the dish sizes and array layout encoded in
the input.

Where command-line options represent physical quantities, they should be
specified with the unit e.g. ``0.21m`` or ``18 arcsec``. Angles specified
without a unit are assumed to be in radians. Wavelengths can also be specified
using units of frequency.

Run ``imager.py -h`` to see a summary of command-line options, as well as
their default values. There are also a number of options that are not
documented here, as they require a more detailed understanding of the
implementation details and won't be needed for common usage.

Input selection options
^^^^^^^^^^^^^^^^^^^^^^^

.. option:: --start-channel <CHANNEL>, --stop-channel <CHANNEL>

   Selects a range of channels to image. The channels are numbered from 0, and
   the stop channel is *excluded*.

.. option:: --subtract <URL>

   Specifies a local sky model to subtract from the visibilities
   (typically for continuum subtraction). There are three options:

   auto
       Specifying the value ``auto`` will use a sky model found in the input
       data set. This only works with katdal data sets.
   `katpoint`_ catalogue
       A ``file://`` URL containing a catalogue of sources. Sources whose flux
       model frequency range do not cover the channel being imaged will be
       ignored.
   `katdal`_ dataset
       For more flexibility than the ``auto`` option, one can specify a katdal
       URL explicitly. There are a few extra query parameters to specify:

       format
           Must be ``katdal``
       target
           The katpoint description of the target that was imaged (required).
       continuum
           Optional, specifies the name of the continuum image stream. This is
           only needed if there were multiple continuum imager configurations
           run on this data set.

   The fluxes must be *apparent* fluxes i.e., modulated by the
   primary beam. That may change in future versions.

.. option:: -i <KEY>=<VALUE>, --input-option <KEY>=<VALUE>

   Passes an option to an input backend. The MS backend supports the following
   key-value pairs:

   data=<COLUMN>
     Specifies the column in the measurement set containing the data to image
     (e.g. ``DATA`` or ``CORRECTED_DATA``). The default is ``DATA``.
   data-desc=<INDEX>
     Data description in the measurement set to image, starting from 0
   field=<INDEX>
     Field in the measurement set to image, starting from 0
   pol-frame=sky | feed
     Reference frame for polarization. Use ``feed`` if the visibilities
     correspond to the feeds on altitude-azimuth mount dishes. The default
     assumes that X is towards the north celestial pole (IAU/IEEE
     definition). When using this option, the input must have a full four
     polarizations.
   uvw=casa | strict
     Sign convention for UVW coordinates. Use ``strict`` if the UVW
     coordinates follow the Measurement Set definition. The default
     (``casa``) uses the opposite convention, which is implemented by CASA
     and other imagers.

   The katdal backend supports the following:

   subarray=<INDEX>
     Subarray index within the file, starting from 0 (defaults to first in
     file).
   spw=<INDEX>
     Spectral window index within the file, starting from 0 (defaults to first
     in file).
   target=<TARGET>
     Target to image. This can be either an index into the catalogue stored in
     the file (starting from 0) or a name. If not specified, it defaults to the
     first target with the ``target`` tag. If there isn't one, it defaults to
     the first without a ``bpcal`` or ``gaincal`` tag.
   ref-ant=<NAME>
     Name of antenna to use as the reference for identifying scans. Refer to
     the katdal documentation for details. If not specified, the virtual
     "array" antenna is used.
   apply-cal=<TYPES>
     1GC calibration solutions to apply. This does not do any calibration
     itself, but uses solutions stored in the dataset. This can be a
     comma-separated list or ``all`` (the default) to apply all available
     calibration solutions. Refer to the katdal documentation for more
     information.

   To provide multiple key-value pairs, specify :option:`-i` multiple times.

.. _katpoint: https://pypi.org/project/katpoint/

Output image options
^^^^^^^^^^^^^^^^^^^^
By default, katsdpimager uses the dish size and wavelength to estimate the
field of view, and the longest baseline and wavelength to estimate the
resolution. You can either keep these heuristics but adjust the scaling
factors using :option:`--q-fov` and :option:`--image-oversample`, or you can
disable the heuristics and specify your own sizes using :option:`--pixel-size`
and :option:`--pixels`.

.. option:: --q-fov <RATIO>

   Specifies a scaling factor for the field-of-view estimation. Since there is
   no information in the measurement set about aperture efficiency or beam
   shape, the heuristics assume a uniformly illuminated dish and chooses a
   field of view that encompasses the first null of this ideal beam. For a
   tapered illumination or to image beyond the first null, one will need to
   specify a value larger than 1.

.. option:: --image-oversample <RATIO>

   Specify the number of pixels per synthesized beam. The beam size used here
   is computed using only the longest baseline and the wavelength, rather than
   the full point spread function.

.. option:: --pixel-size <ANGLE>

   Specify the size of pixels at the centre of the image (pixels do not all
   subtend exactly the same angle due to the projection).

.. option:: --pixels <N>

   The number of pixels in each direction. For implementation reasons, not all
   sizes are supported. If an unsupported size is specified, the closest
   supported size will be reported in the error message.

.. option:: --stokes <PARAMETERS>

   A list of Stokes parameters to image, with no spaces and in upper case e.g.
   :kbd:`IQUV`.

Imaging control options
^^^^^^^^^^^^^^^^^^^^^^^

.. option:: --weight-type {natural,uniform,robust}

   Method used to compute imaging density weights.

.. option:: --robustness <N>

   Robustness parameter for robust weighting.

.. option:: --primary-beam {meerkat,meerkat:1,none}

   Specify a primary beam model. At present only a built-in MeerKAT model is
   available, and it is a simple circularly-symmetric, amplitude-only,
   dish-independent, polarization-independent model. Note that this is too
   simplistic to properly model the MeerKAT primary beam: it can introduce
   flux errors of up to 20% towards the edges of the main lobe (particularly in
   short observations which span a narrow range of parallactic angles), and it
   only accounts for Stokes I response to unpolarized emission.

   The name ``meerkat:1`` will continue to refer to this specific model in
   future versions, so can be used in scripts that need to have reproducible
   results.

.. option:: --primary-beam-cutoff <VALUE>

   In the final image, pixels corresponding to points in the primary beam with
   less than this amount of power are discarded when using
   :opt:`--primary-beam`. This avoids polluting the image with high levels of
   noise from the null of the primary beam. Note that this only affects the
   output; sufficiently bright sources in the null will still be cleaned.

   At present this will only remove the nulls and may leave side-lobes (if
   they have more power than the cutoff), but in future that may change so
   that only the main lobe is preserved.

Quality options
^^^^^^^^^^^^^^^

.. option:: --precision {single,double}

   Specify the floating-point precision of the output image. This precision is
   also used in the gridding and Fourier transforms. Note that most NVIDIA
   GPUs other than Tesla have extremely poor double-precision performance.

.. option:: --psf-cutoff <VALUE>

   Fraction of PSF peak at which to truncate the PSF for CLEAN. Using a larger
   value will reduce the cost of each CLEAN cycle, but too large a value may
   prevent CLEAN from converging.

.. option:: --psf-limit <VALUE>

   Maximum fraction of image to use for PSF. This restricts the size of the
   PSF to a certain fraction of the image, if not already further constrained
   by :option:`--psf-cutoff`.

.. option:: --major-gain <VALUE>

   Fraction of the peak at the start of a major cycle that will be cleaned in
   that cycle.

.. option:: --threshold <SIGMAS>

   Threshold at which CLEAN should stop, as a multiple of the estimated RMS
   noise. CLEAN is stopped when any of the conditions specified by
   :option:`--major-gain`, :option:`--threshold` or :option:`--minor` is
   reached.

.. option:: --major <N>, --minor <M>

   Maximum number of major cycles and maximum number of minor cycles per major
   cycle for CLEAN.

.. option:: --eps-w <VALUE>

   Level at which W-correction kernel is truncated.

Output options
^^^^^^^^^^^^^^
Normally only the output image is written, but it is also possible to write
various intermediate products:

.. option:: --write-weights <FILE>, --write-psf <FILE>, --write-grid <FILE>,
   --write-dirty <FILE>, --write-model <FILE>, --write-residuals <FILE>

   Write a FITS file with the corresponding intermediate results.

When imaging multiple channels, both these intermediate filenames and the
output filename should be a printf-style format string which will be populated
with the channel index.

.. _cpu:

Running on the CPU
==================
It is also possible to run katsdpimager without a GPU, although it is not well
optimised and thus not recommended. When installing katsdpimager, use the
command

.. code-block:: sh

   pip install '.[cpu]'

to install the necessary support packages. Note that this will still install GPU packages like
pycuda; if you're unable to install them, you'll need to modify katsdpimager
yourself to remove the dependencies.

.. option:: --host

   Perform all computations on the CPU.

Generating a video
==================
While serious investigation of the outputs should be done with dedicated FITS
viewing tools, a script called :program:`fits-video.py` is provided that can
combine a number of FITS files produced by katsdpimager into a video file (in
``.mp4`` format). Run it with :option:`-h` for usage instructions.

In addition to the normal requirements of katsdpimager, this script requires
:mod:`matplotib`.

.. note::

   It is only designed to work with FITS files produced by katsdpimager,
   ideally with the same parameters and on the same field, and makes
   assumptions about units, axis ordering etc. It might or might not work with
   other FITS files.
