User guide for katsdpimager
---------------------------
katsdpimager is a GPU-accelerated spectral line imager for radio astronomy. It
is still in development, so not all features necessarily work.

Requirements
============
katsdpimager is implemented in Python, and is installed with standard Python
packaging tools. The pure-Python dependencies are automatically handled by
these packaging tools, with one exception: for now a special fork of pybind11
is required, which can be installed with

.. code-block:: sh

    pip install git+https://github.com/bmerry/pybind11@factory-constructors#egg=pybind11

At some point this functionality is expected to be merged into pybind11.

There are some additional requirements:

 - A Github account with access to the SKA South Africa private repositories;
 - An NVIDIA GPU with the CUDA toolkit. At least version 6.0 is required, but
   testing is only done with 8.0 and later (but see :ref:`cpu`);
 - `Casacore`_ 2.x, compiled with Python support (optional, only needed for
   reading Measurement Sets);
 - libhdf5, including development headers (``libhdf5-dev`` in Debian/Ubuntu)
 - `Eigen3`_ (``libeigen3-dev`` in Debian/Ubuntu);
 - A C++ compiler such as GCC.
 - Boost headers, if the C++ compiler is too old to provide either
   ``std::experimental::optional`` or ``std::optional``.

.. _Casacore: https://github.com/casacore/casacore

.. _Eigen3: http://eigen.tuxfamily.org

Installation
============
Once the pre-requisites from the previous section are present, install
katsdpsigproc:

.. code-block:: sh

   git clone git@github.com:ska-sa/katsdpsigproc
   cd katsdpsigproc
   pip install .

Then install katsdpimager from the katsdppipelines repository. The cutting
edge is the ``imager-dev`` branch of katsdppipelines; the ``master`` branch
contains code that has been reviewed.

.. code-block:: sh

   git clone -b imager-dev git@github.com:ska-sa/katsdppipelines
   cd katsdppipelines/katsdpimager
   pip install .

After these steps, you should be able to run ``imager.py``, but see the next
section for information on file format-specific dependencies.

File formats
============
Two input formats are supported: `Measurement Sets`_ and KAT-style HDF5 files
read by `katdal`_. Output is to `FITS`_ files. The input can contain multiple
channels, but a separate FITS file is written for each channel.

.. _Measurement sets: http://casa.nrao.edu/Memos/229.html
.. _katdal: https://github.com/ska-sa/katdal/
.. _FITS: http://fits.gsfc.nasa.gov/fits_documentation.html

The input file format is detected by extension, so a Measurement Set *must*
have the suffix ``.ms`` and a katdal file must have the suffix ``.h5``.

Each file format has additional Python package dependencies. Use ``pip install
.[ms]`` to ensure support for Measurement Sets and ``pip install .[katdal]`` to
ensure support for katdal.

Command-line options
====================
The simplest possible incantation is

.. code-block:: sh

   imager.py input.ms output.fits

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
     the katdal documentation for details.
   apply-cal=<TYPES>
     1GC calibration solutions to apply. This does not do any calibration
     itself, but uses solutions stored in the dataset. The argument can contain
     any subset of K, B and G to select delay, bandpass and gain solutions. It
     can also be ``all`` or ``none``.

   To provide multiple key-value pairs, specify :option:`-i` multiple times.

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

.. option:: --write-weights <FILE>, --write-psf <FILE>, --write-grid <FILE>, --write-dirty <FILE>, --write-model <FILE>, --write-residuals <FILE>

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
