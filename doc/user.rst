User guide for katsdpimager
---------------------------
katsdpimager is a GPU-accelerated spectral line imager for radio astronomy. It
is still in development, so not all features necessarily work.

Requirements
============
katsdpimager is implemented in Python, and is installed with standard Python
packaging tools. The pure-Python dependencies are automatically handled by
these packaging tools, but there are some additional requirements:

 - An SSH key that is authorised for Github access to the SKA South Africa
   private repositories.
 - An NVIDIA GPU with the CUDA toolkit. At least version 6.0 is required, but
   testing is only done with 7.0 and later (but see :ref:`cpu`).
 - `Casacore`_ 2.x, compiled with Python support
 - libhdf5, including development headers (``libhdf5-dev`` in Debian/Ubuntu)
 - Boost.Python
 - A C++ compiler such as GCC

.. _Casacore: https://github.com/casacore/casacore

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

After these steps, you should be able to run ``imager.py``.

File formats
============
The input is taken from a `Measurement Set`_ containing visibilities and the
output image is a `FITS`_ file. The input can contain multiple channels, but
only one channel can be imaged.

.. _Measurement set: http://casa.nrao.edu/Memos/229.html
.. _FITS: http://fits.gsfc.nasa.gov/fits_documentation.html

The input file *must* have the suffix ``.ms``. This is to support future
expansion where multiple file formats will be supported and detected by their
extension.

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

.. option:: --channel <CHANNEL>

   Selects the channel to image from a multi-channel input, counting from 0.

.. option:: --input-option <KEY>=<VALUE>

   Passes an option to an input backend. At the moment the only backend is for
   measurement sets, which supports the following key-value pairs:

   data=<COLUMN>
     Specifies the column in the measurement set containing the data to image
     (e.g. ``DATA`` or ``CORRECTED_DATA``). The default is ``DATA``.
   data-desc=<INDEX>
     Data description in the measurement set to image, starting from 0
   field=<INDEX>
     Field in the measurement set to image, starting from 0

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

.. option:: --psf-patch

   Pixels in beam patch used for CLEAN.

.. option:: --major <N>, --minor <M>

   Number of major cycles and number of minor cycles per major cycle for
   CLEAN.

.. option:: --eps-w <VALUE>

   Level at which W-correction kernel is truncated.

Output options
^^^^^^^^^^^^^^
Normally only the output image is written, but it is also possible to write
various intermediate products:

.. option:: --write-weights <FILE>, --write-psf <FILE>, --write-grid <FILE>, --write-dirty <FILE>, --write-model <FILE>, --write-residuals <FILE>

   Write a FITS file with the corresponding intermediate results.

.. _cpu:

Running on the CPU
==================
It is also possible to run katsdpimager without a GPU, although it is not well
optimised and thus not recommended. When installing katsdpimager, use the
command

.. code-block:: sh

   pip install '.[cpu]'

to install the necessary support packages. Then pass :option:`--host` when
running.  Note that this will still install GPU packages like
pycuda; if you're unable to install them, you'll need to modify katsdpimager
yourself to remove the dependencies.