===========================
Cosmic Ray Heating For 21cm
===========================

Overview
--------

:Repository: Cosmic Ray Heating For 21cm
:Author: Thomas Gessey-Jones
:Homepage:  https://github.com/ThomasGesseyJones/CosmicRayHeatingFor21cm
:Paper: https://arxiv.org/abs/2304.07201


This repository contains the code used to produce the cosmic ray heating
window functions and heating coefficients for the paper
`Signatures of Cosmic Ray Heating in 21-cm Observables <https://ui.adsabs.harvard.edu/abs/2023arXiv230407201G/abstract>`__.

The code was developed to be integrated into the semi-numerical 21-cm signal simulation code described in
`Visbal et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012Natur.487...70V/abstract>`__,
`Fialkov et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014Natur.506..197F/abstract>`__,
and `Reis et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5993R/abstract>`__
(for access to this simulation code please contact Anastasia Fialkov at
`anastasia.fialkov@gmail.com <mailto:anastasia.fialkov@gmail.com>`__).
However, this cosmic ray heating code can operate standalone, and can be used to produce the heating window functions
for any semi-numerical 21-cm simulation code.
Hence, it is made available here for the community to use.
For access to this simulation code please contact Anastasia Fialkov at
`anastasia.fialkov@gmail.com <mailto:anastasia.fialkov@gmail.com>`__.


Code Structure
--------------

The code is split into four files:

- `cosmic_ray_general.py` which contains the general functions used to calculate the heating window functions and
  heating coefficients. Most of these functions are either approximate analytic solutions to the cosmic ray energy
  equation (paper equation 25) or the comoving path length equation (paper equation 26).
- `cosmic_ray_coeffs.py` which contains the functions used to calculate the heating coefficients which describe the
  rate of cosmic ray heating around a point source. This coefficients are calculated on an adaptive redshift grid
  which will refine to ensure that the heating coefficients are calculated to a specified accuracy.
- `cosmic_ray_window.py` which contains the functions used to calculate the heating window functions which describe the
  distribution of cosmic ray heating around a point source. Near the point source these window functions are calculated
  on a higher resolution grid, that is then convolved with the possible point source positions to produce the heating
  window functions.
- `generate_cosmic_ray.py` acts as the main interface to the code, taking command line arguments to specify the
  cosmic ray spectrum to use and the type of window functions to produce. It is described in more detail below.

It is intended to be used through the command line interface of `generate_cosmic_ray.py` which takes the following
arguments:

- path_to_storage: a string. The path of folder where window functions and coefficients are to be stored
- window_function_type: a string. ID of the type of window function to be generated. Options are 'global',
  'locally_confined' and 'free_streaming'. Note for 'global' and 'locally_confined' the window functions are
  trivial and so are not stored (coefficients are still stored).
- alpha: a float. Exponent of cosmic ray power-law spectrum (see paper equation 22)
- ke_min: a float. Minimum energy of cosmic ray power-law spectrum in MeV (see paper equation 22)
- ke_max: a float. Maximum energy of cosmic ray power-law spectrum in MeV (see paper equation 22)

Other aspects of the code can be changed by editing the constants at the top of the code. For example, the
number of pixels in the semi-numerical signal simulation cube or the pixels side length can be changed by editing
the constants at the top of `generate_cosmic_ray.py`.

The code is parallelised using `joblib <https://pypi.org/project/joblib/>`__ and so can be run on multiple cores
for faster execution. The number of jobs is currently 35 so no benefit will be seen from running on more than 35 cores.
This value is set by the number of emission redshifts coefficients/window functions are calculated at.


Definitions
-----------

The heating coefficients are defined as the normalized rate of cosmic ray energy deposition into the IGM (not heating)
at :math:`z` due to a source at :math:`z'`:

.. math::
    C(z, z') =  \int_{0}^{\infty} dT \left(\left.\frac{dE}{dz}\right|_{E\&I}(T) \frac{dN_{E}(T'[z', z, T])}{dT} \frac{dT'}{dT} \right)

where :math:`T` is the kinetic energy of cosmic ray protons at :math:`z`, :math:`T'` is the kinetic energy the cosmic
ray proton would have needed to have had at :math:`z'` to have a kinetic energy of :math:`T` at :math:`z`,
:math: `\frac{dN_{E}(T'[z', z, T])}{dT}` is the cosmic ray spectrum at :math:`z'` (normalized to
one unit of energy emission) and,
:math:`\left.\frac{dE}{dz}\right|_{E\&I}(T)` is the energy deposition rate per redshift of a cosmic ray proton of
kinetic energy :math:`T` at :math:`z` (see paper equation 4). :math:`\frac{dT'}{dT}` is a Jacobian factor which
transforms from the cosmic ray energy at :math:`z` to the cosmic ray energy at :math:`z'`.

The heating window functions :math:`W(\vec{x}, z, z')` are simply the distribution this energy deposition rate around a point source at
:math:`z'`. Hence, vary with cosmic ray propagation mode (see paper section 3.3.2). The heating window functions are
therefore probability distributions and so are normalized to one.

Therefore the cosmic ray energy deposition rate into the IGM (not heating rate) is given by:

.. math::
    \left.\frac{dU}{dV dz}\right|_{cr}{(\vec{x}) =  \eta_{cr} \int C(z, z') (W(\vec{x}, z, z') \ast SFRD_{cr})(\vec{x}, z') dz'}

where :math:`\eta_{cr}` is the efficiency of cosmic ray emission and :math:`SFRD_{cr}` is the star
formation (redshift) rate density of cosmic ray emitting sources (see paper section 3.3.1). This can then be converted to a heating
rate by multiplying by the local heating efficiency (see paper section 3.3.3) and the local heat capacity of the IGM.



Licence and Citation
--------------------

The software is available on the MIT licence.

If you use the code for academic purposes we request that you cite the following
`paper <https://ui.adsabs.harvard.edu/abs/2023arXiv230407201G/abstract>`__.

.. code:: bibtex

    @ARTICLE{2023arXiv230407201G,
           author = {{Gessey-Jones}, T. and {Fialkov}, A. and {de Lera Acedo}, E. and {Handley}, W.~J. and {Barkana}, R.},
            title = "{Signatures of Cosmic Ray Heating in 21-cm Observables}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - High Energy Astrophysical Phenomena},
             year = 2023,
            month = apr,
              eid = {arXiv:2304.07201},
            pages = {arXiv:2304.07201},
              doi = {10.48550/arXiv.2304.07201},
    archivePrefix = {arXiv},
           eprint = {2304.07201},
     primaryClass = {astro-ph.CO},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230407201G},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


Requirements
------------

The code requires the following packages to run:

- `numpy <https://pypi.org/project/numpy/>`__
- `scipy <https://pypi.org/project/scipy/>`__
- `numba <https://pypi.org/project/numba/>`__
- `joblib <https://pypi.org/project/joblib/>`__

and was developed using python 3.8. It has not been tested on other versions
of python.


Questions
---------

If you have any questions about the code please contact Thomas Gessey-Jones
at `tg400@cam.ac.uk <mailto:tg400@cam.ac.uk'>`__. Or alternatively open an
issue on the github page.
