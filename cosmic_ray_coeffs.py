"""
Cosmic Ray Coefficients
=======================
Compute the cosmic ray C coefficients that give the normalized heating rate of cosmic rays in the simulation
"""

# Required modules
from collections.abc import Callable
from typing import Union, Tuple
import cosmic_ray_general as crg
import numpy as np
from scipy.integrate import quad, trapz
from numba import njit


# Accuracy parameters
C_UPPER_Z_CUTOFF = 40  # Redshift above which no star formation is expected
Z_LOW_ACC_FACTOR = 0.01  # Fraction of volume of a cell that photons could escape from in Delta z low
C_INT_ACC_FACTOR = 1e-3  # Accuracy required of z' sampling, estimated using int_z^(z_upper) C(z, z') dz'


# Functions to compute the coefficients
@njit(cache=True)
def cosmic_ray_c_integrand(ke_now: float, z_now: float, z_ini: float, alpha: float, ke_min: float,
                            ke_max: float) -> float:
    """ Utility function to compute the integrand of the C coefficient integral

    Parameters
    ----------
    ke_now : Float
        Kinetic energy of proton now in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted
    alpha : Float
        Exponent of the power-law
    ke_min : Float
        Minimum KE cutoff of the power-law in MeV
    ke_max : Float
        Maximum KE cutoff of the power-law in MeV

    Returns
    -------
    Float
        Integrand of the C coefficient integral for the given model evaluated at ke_now
    """
    return crg.normalized_heating_rate(ke_now, z_now) * \
        crg.power_law_spectrum(crg.initial_ke(ke_now, z_now, z_ini), alpha, ke_min, ke_max) * \
        crg.jacobian_factor(ke_now, z_now, z_ini)


def cosmic_ray_c(z_now: float, z_ini: float,  alpha: float, ke_min: float, ke_max: float) -> float:
    """ Compute the cosmic ray C coefficient between z_now and z_ini

    Parameters
    ----------
    z_now : Float
        Redshifts of observation
    z_ini : Float
        Redshift cosmic rays are emitted at
    alpha : Float
        Exponent of the power-law
    ke_min : Float
        Minimum KE cutoff of the power-law in MeV
    ke_max : Float
        Maximum KE cutoff of the power-law in MeV

    Returns
    -------
    Float
        C coefficient, encapsulating heating rate from cosmic rays at z_now, emitted at z_ini
    """
    return quad(cosmic_ray_c_integrand, 0, np.inf, args=(z_now, z_ini, alpha, ke_min, ke_max),
                epsabs=0, epsrel=C_INT_ACC_FACTOR/10)[0]


# Functions to sample Cs to sufficient accuracy
def delta_z_low(z: Union[float, np.array]) -> Union[float, np.array]:
    """ Compute Delta z low, the first Delta z C(z, z') will be sampled at after the 0th term C(z, z)

    Parameters
    ----------
    z : Float
        z redshift of C(z, z') you are evaluating for
    Returns
    -------
    Float
        Delta z low for that redshift
    """
    scale_prefactor = (crg.LPIX*crg.H0*np.sqrt(crg.OMEGA_M0))/(2 * crg.C_IN_MPC)
    return scale_prefactor * (1 - (1 - Z_LOW_ACC_FACTOR)**(1/3)) * ((1 + z)**(3/2))


def _sufficient_c_sampling_check(cs: np.array, delta_zs: np.array) -> bool:
    """ Determine if C coefficients are sufficiently sampled

    Parameters
    ----------
    cs : Array
        Current sample of C coefficients
    delta_zs : Array
        Current sample of Delta z values they are evaluated at

    Returns
    -------
    Bool
        True if sufficient sampling has been reached so that int C(z, z+Delta z) dDeltaZ has converged
    """

    # Compute integral with full and half sample
    full_sample_int = trapz(cs*delta_zs, np.log(delta_zs))
    half_sample_int = trapz(cs[::2]*delta_zs[::2], np.log(delta_zs[::2]))

    # Compare fractional difference to desired accuracy
    return (np.abs(half_sample_int - full_sample_int)/full_sample_int) < C_INT_ACC_FACTOR


def autosample_cosmic_ray_c(z: float, coeff_func: Callable) -> Tuple[np.array, np.array]:
    """ Automatically sample the cosmic ray coefficients to a pre-specified accuracy

    Parameters
    ----------
    z : Float
        z Redshift of C(z, z')
    coeff_func : Callable
        Function that returns C(z, z') with signature float, float -> float

    Returns
    -------
    Array
        C(z, z') coefficients
    Array
        z' redshift values these coefficients were sampled at
    """
    # Function internal parameters
    if z < 30:
        initial_nof_samples = 65  # Higher number of samples at low redshift as SFR evolves more rapidly 
    else:
        initial_nof_samples = 33

    # Handle edge cases of z = C_UPPER_Z_CUTOFF and the invalid case of z greater than said cutoff
    if z == C_UPPER_Z_CUTOFF:
        return np.array([0.0]), np.array([C_UPPER_Z_CUTOFF])
    elif z >= C_UPPER_Z_CUTOFF:
        raise ValueError(f'Redshift given z={z} exceeds maximum that coefficients are to be calculated '
                         f'for z={C_UPPER_Z_CUTOFF}')

    # Compute Delta z low, and the 0th coefficient
    dz_low = delta_z_low(z)
    c0 = quad(lambda x: coeff_func(z, z + x), 1e-15, dz_low, epsabs=0, epsrel=C_INT_ACC_FACTOR)[0]/dz_low

    # Setup initial sample
    delta_z_samples = np.logspace(np.log10(dz_low), np.log10(C_UPPER_Z_CUTOFF - z), initial_nof_samples)
    cs = np.array([coeff_func(z, z+dz) for dz in delta_z_samples])

    # Loop while not at sufficient accuracy
    while not(_sufficient_c_sampling_check(cs, delta_z_samples)):
        # Increase nof samples to 2N - 1
        delta_z_samples = np.logspace(np.log10(dz_low), np.log10(C_UPPER_Z_CUTOFF - z), 2*len(delta_z_samples) - 1)
        new_cs = np.empty_like(delta_z_samples)

        # Escape condition in case there is an issue and the integral based sufficiency condition never converges
        if len(delta_z_samples) > 1000:
            raise RuntimeError(f'Cosmic ray coefficient sampling not converging, stopping as {1000} coefficients '
                               f'reached and yet to satisfy sufficient sample check.')

        # Copy over existing results to avoid wasted evaluation time
        new_cs[::2] = cs

        # Compute the new samples that are needed
        new_cs[1::2] = np.array([coeff_func(z, z+dz) for dz in delta_z_samples[1::2]])
        cs = new_cs

    # Return the final results, the output being samples in z' not delta z
    return np.insert(cs, 0, c0), np.insert(z + delta_z_samples, 0, z)
