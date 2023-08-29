"""
Cosmic Ray General
==================
Code to model the evolution of the energy and position of cosmic ray protons emitted into the early universe IGM.
Essentially a set of common utility functions needed to compute the cosmic ray C coefficients and window functions.
"""

# Required modules
from typing import Union
import numpy as np
import scipy.constants
import os
from scipy.io import loadmat
from numba import njit
from scipy.special import gamma, hyp2f1

# Cosmology constants
H0 = 100*0.6704/3.086e19  # s^{-1}
OMEGA_M0 = 0.31687
OMEGA_B0 = 0.04902

fHe = 0.0732
H_FRAC = 1 - 4*fHe/(4*fHe + (1-fHe))

# Atomic Physics Constants
PROTON_M = 938.28  # In MeV/c^2
C_IN_MPC = scipy.constants.c / 3.086e22  # In Mpc s^{-1}
HI_ION_TIME_SCALE = 3e3 * 3.15e7  # From Sazonov & Sunyaev 2015, in s

# Derived Parameters
RHO_CRIT = 3 * H0**2 / (8 * np.pi * scipy.constants.G * 1e6)  # In kg cm^{-3}
TAU_INTERM = HI_ION_TIME_SCALE * scipy.constants.m_p / (RHO_CRIT * H_FRAC * OMEGA_B0)  # In s
KE_DECAY_CONST = 1**(3/2)/(H0*np.sqrt(OMEGA_M0)*TAU_INTERM)  # In MeV^{3/2}

# Model Parameters
KE_REL_TRANS_MEV = 100  # Kinetic energy of the relativistic to non-relativistic transition in MeV

# Simulation Parameters
LPIX = 3  # comoving Mpc


# General Utility Functions
@njit(cache=True)
def power_law_spectrum(ke: float, alpha: float = -2, ke_min: float = 1e-3, ke_max: float = 1e8) -> float:
    """ Probability density of a truncated power-law distribution normalized to a total of 1 MeV of energy

    Parameters
    ----------
    ke : Float
        Kinetic energy to evaluate at in MeV
    alpha : Float
        Exponent of the power-law, defaults to -2
    ke_min : Float
        Minimum KE cutoff of the power-law in MeV, defaults to 1e-3
    ke_max : Float
        Maximum KE cutoff of the power-law in MeV, defaults to 1e8

    Returns
    -------
    Float
        Normalized probability density in MeV^-2
    """
    if alpha == -2:
        nf = 1/(np.log(ke_max) - np.log(ke_min))
    else:
        nf = (alpha + 2)/(ke_max**(alpha + 2) - ke_min**(alpha + 2))
    return nf * (ke**alpha) * (ke <= ke_max) * (ke >= ke_min)


# Heating Functions
@njit(cache=True)
def normalized_heating_rate(ke: float, zs: float) -> float:
    """ Compute heating rate (normalized to remove fheat factor) of the IGM by CR protons by excitation and ionization

    Parameters
    ----------
    ke : Float
        Kinetic energies in MeV of the proton
    zs : Float
        Redshift heating rate is being evaluated at

    Returns
    -------
    Float
        Heating rate in MeV per redshift of the given proton
    """
    return KE_DECAY_CONST * ((1 + zs) ** (1/2)) * (ke ** (-1/2))


# CR Energy Evolution Functions
@njit(cache=True)
def initial_ke_rel(ke_now: float, z_now: float, z_ini: float) -> float:
    """ Initial kinetic energy of a CR proton if always in the relativistic regime

    Parameters
    ----------
    ke_now : Float
        Kinetic energy of protons now in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Kinetic energy of the proton when emitted in MeV
    """
    p_sq_factor = (ke_now**2 + 2*ke_now*PROTON_M) * (((1 + z_ini)/(1 + z_now))**2)
    return -PROTON_M + np.sqrt(PROTON_M**2 + p_sq_factor)


@njit(cache=True)
def current_ke_rel(ke_ini: float, z_now: float, z_ini: float) -> float:
    """ Current kinetic energy of a CR proton if always in the relativistic regime

    Parameters
    ----------
    ke_ini : Float
        Initial kinetic energy of the proton in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Kinetic energy of the proton at z_now in MeV
    """
    p_sq_factor = (ke_ini**2 + 2*ke_ini*PROTON_M) * (((1 + z_now)/(1 + z_ini))**2)
    return -PROTON_M + np.sqrt(PROTON_M**2 + p_sq_factor)


@njit(cache=True)
def initial_ke_non_rel(ke_now: float, z_now: float, z_ini: float) -> float:
    """ Initial kinetic energy of a CR proton if always in the non-relativistic regime

    Parameters
    ----------
    ke_now : Float
        Kinetic energy of protons now in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Kinetic energy of the proton when emitted in MeV
    """
    adb_factor = (1 + z_ini)**2 / (1 + z_now)**2
    return adb_factor * (ke_now**(3/2) + KE_DECAY_CONST*((1+z_now)**(3/2) -
                                                         ((1 + z_now)**3)*((1 + z_ini)**(-3/2))))**(2/3)


@njit(cache=True)
def current_ke_non_rel(ke_ini: float, z_now: float, z_ini: float) -> float:
    """ Current kinetic energy of a CR proton if always in the non-relativistic regime

    Parameters
    ----------
    ke_ini : Float
        Initial kinetic energy of the proton in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Kinetic energy of the proton at z_now in MeV
    """
    adb_factor = (1 + z_now)**2 / (1 + z_ini)**2
    return adb_factor * (ke_ini**(3/2) - KE_DECAY_CONST*(((1+z_now)**(-3/2))*((1 + z_ini)**3) -
                                                         (1 + z_ini)**(3/2)))**(2/3)


@njit(cache=True)
def transition_redshifts(ke_now: float, z_now: float) -> float:
    """ Compute redshift a proton would have transitioned between the relativistic and non-relativistic regime

    Parameters
    ----------
    ke_now : Float
        Kinetic energy of proton now in MeV
    z_now : Float
        Redshift now

    Returns
    -------
    Float
        Redshift of relativistic to non-relativistic transition
    """
    ke_parameter = KE_DECAY_CONST*((1 + z_now)**(-3/2)) + (ke_now**(3/2))/((1 + z_now)**3)
    return ((-KE_DECAY_CONST + np.sqrt(KE_DECAY_CONST**2 + 4*ke_parameter*(KE_REL_TRANS_MEV**(3/2)))) /
            (2*(KE_REL_TRANS_MEV**(3/2))))**(-2/3) - 1


@njit(cache=True)
def initial_ke(ke_now: float, z_now: float, z_ini: float) -> float:
    """ Initial kinetic energy of an arbitrary CR proton

    Parameters
    ----------
    ke_now : Float
        Kinetic energy of proton now in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Kinetic energy of the proton when emitted in MeV
    """

    # Assume initially no transitions occur
    if ke_now > KE_REL_TRANS_MEV:
        ini_ke = initial_ke_rel(ke_now, z_now, z_ini)
    else:
        ini_ke = initial_ke_non_rel(ke_now, z_now, z_ini)
        if ini_ke > KE_REL_TRANS_MEV:  # Check if transition occurs
            ini_ke = initial_ke_rel(KE_REL_TRANS_MEV, transition_redshifts(ke_now, z_now), z_ini)
    return ini_ke


@njit(cache=True)
def current_ke(ke_ini: float, z_now: float, z_ini: float) -> float:
    """ Current kinetic energy of an arbitrary CR proton

    Parameters
    ----------
    ke_ini : Float
        Initial kinetic energy of the proton in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Kinetic energy of the proton at z_now in MeV
    """

    # If at a non-relativistic initial energy then it stays non-relativistic
    if ke_ini <= KE_REL_TRANS_MEV:
        ke_now = current_ke_non_rel(ke_ini, z_now, z_ini)

    # Otherwise, starts relativistic and might transition to non-relativistic later
    else:
        ke_now = current_ke_rel(ke_ini, z_now, z_ini)
        if ke_now < KE_REL_TRANS_MEV:  # Check if transition occurs
            z_trans = (1 + z_ini) * (((KE_REL_TRANS_MEV**2 + 2*KE_REL_TRANS_MEV*PROTON_M) /
                                      (ke_ini**2 + 2*ke_ini*PROTON_M))**(1/2)) - 1
            ke_now = current_ke_non_rel(KE_REL_TRANS_MEV, z_now, z_trans)

    return ke_now


@njit(cache=True)
def jacobian_factor(ke_now: float, z_now: float, z_ini: float) -> float:
    """ Jacobian factor that transforms between density of KE now and KE at emission

    Branch-less implementation for numpy arrays

    Parameters
    ----------
    ke_now : Float
        Kinetic energy of proton now in MeV
    z_now : Float
        Redshift now
    z_ini : Float
        Redshift when proton was emitted

    Returns
    -------
    Float
        Jacobian factor (dimensionless) dT'/dT
    """

    # Get initial KE of proton
    ini_ke = initial_ke(ke_now, z_now, z_ini)

    # Assume initially no transitions occur
    if ke_now > KE_REL_TRANS_MEV:
        jacob = ((ke_now + PROTON_M)/(ini_ke + PROTON_M))*(((1 + z_ini) / (1 + z_now))**2)
    else:
        if ini_ke < KE_REL_TRANS_MEV:
            jacob = np.sqrt(ke_now / ini_ke)*(((1 + z_ini) / (1 + z_now))**3)
        else:  # Transitioning case
            jacob = ((KE_REL_TRANS_MEV + PROTON_M) / (ini_ke + PROTON_M)) * np.sqrt(ke_now / KE_REL_TRANS_MEV) * \
                    (((1 + z_ini) ** 2) * (1 + transition_redshifts(ke_now, z_now)) / ((1 + z_now) ** 3))
    return jacob


# CR Path Length Functions
def g_1_func(x: np.ndarray) -> np.ndarray:
    """ Special function required to compute path length in the non-relativistic regime

    G_1 == 2F1(-1/3, -1/3; 2/3; x)

    Parameters
    ----------
    x : Array
        Arguments to evaluate G_1 at

    Returns
    -------
    Array
        G_1(x)
    """

    # Input sanitization
    if np.any(x < 0):
        raise ValueError(f'One or more of the given arguments is outside the range 0 <= x <= 1 for which '
                         f'this function is defined.')

    # Initialize data structure for the output
    output = np.empty(np.shape(x), dtype=float)

    # Consider the two special boundary cases
    output[x == 0] = 1
    output[x >= 1] = gamma(2/3)*gamma(4/3)

    # General case
    output[np.logical_and(x > 0, x < 1)] = hyp2f1(-1/3, -1/3, 2/3, x[np.logical_and(x > 0, x < 1)])
    return output


def g_2_func(x: np.ndarray) -> np.ndarray:
    """ Special function required to compute path length in the relativistic regime

        G_2 == 2F1(1/4, 1/2; 5/4; -x^2)      x in R

        Parameters
        ----------
        x : Array
            Arguments to evaluate G_2 at, x must be real

        Returns
        -------
        Array
            G_2(x)
        """
    return hyp2f1(1/4, 1/2, 5/4, -np.power(x, 2))


def alpha_1_parameter(ke_0: np.ndarray, z_0: float) -> np.ndarray:
    """ Dimensionless parameter required to compute path length in the non-relativistic regime

    alpha_1 == ((H_0 sqrt{Omega_m} tau_{inter} ke_0^{3/2})/((1 MeV)^{3/2}(1+z_0)^3)  + (1+z_0)^{-3/2})^{-2/3}

    Equivalently alpha_1 can be interpreted as 1 + z_{abs}, the inverse scale factor when a cosmic ray proton would
    be thermalized with the IGM.

    Parameters
    ----------
    ke_0 : Array
        Initial KE of the protons in MeV
    z_0: Float
        Emission redshift of the proton

    Returns
    -------
    Array
        alpha_1 parameter evaluated for each of these protons
    """
    return (((ke_0**(3/2)) * ((1 + z_0)**(-3)))/KE_DECAY_CONST + (1 + z_0)**(-3/2))**(-2/3)


def alpha_2_parameter(ke_0: np.ndarray, z_0: float) -> np.ndarray:
    """ Dimensionless parameter required to compute path length in the relativistic regime

    alpha_2 == sqrt(T_0^2/(m_p^2 c^4) + 2 T_0 / (m_pc^2)) / (1 + z_0)

    Parameters
    ----------
    ke_0 : Array
        Initial KE of the protons in MeV
    z_0: Float
        Emission redshift of the proton

    Returns
    -------
    Array
        alpha_2 parameter evaluated for each of these protons
    """
    norm_ke = ke_0 / PROTON_M
    return np.sqrt(norm_ke**2 + 2*norm_ke) / (1 + z_0)


def comoving_path_length_non_rel(z: float, ke_0: Union[float, np.ndarray], z_0: Union[float, np.ndarray]) -> np.ndarray:
    """ Comoving path length a non-relativistic proton would travel between z_0 and z

    Note for performance it is assumed that none of the protons have been absorbed, any such protons should be filtered
    out before calling this function

    Parameters
    ----------
    z : Float
        Redshift proton is at now
    ke_0 : Array
        Initial kinetic energies of the protons in MeV
    z_0 : Float
        Source redshift of protons

    Returns
    -------
    Array
        Comoving path lengths travelled in comoving Megaparsec
    """
    # Pre-computations
    alpha_1 = alpha_1_parameter(ke_0, z_0)
    prefactor = (C_IN_MPC/H0) * ((8/(OMEGA_M0 * PROTON_M))**(1/2)) * (KE_DECAY_CONST**(1/3)) * (alpha_1**(-1/2))

    # Primary computation
    output = ((1 + z_0)**(1/2))*g_1_func((alpha_1/(1 + z_0))**(3/2)) - \
             ((1 + z)**(1/2))*g_1_func((alpha_1/(1 + z))**(3/2))
    return prefactor*output


def comoving_path_length_rel(z: float, ke_0: np.ndarray, z_0: float) -> np.ndarray:
    """ Comoving path length a relativistic proton would travel between z_0 and z

    Parameters
    ----------
    z : Float
        Redshift proton is at now
    ke_0 : Array
        Initial kinetic energies of the protons in MeV
    z_0 : Float
        Source redshift of protons

    Returns
    -------
    Array
        Comoving path lengths travelled in comoving Megaparsec
    """
    # Pre-computations
    alpha_2 = alpha_2_parameter(ke_0, z_0)
    prefactor = 2 * C_IN_MPC * alpha_2 / (H0 * np.sqrt(OMEGA_M0))

    # Primary computation
    output = np.sqrt(1 + z_0)*g_2_func(alpha_2*(1 + z_0)) - np.sqrt(1 + z)*g_2_func(alpha_2*(1 + z))
    return prefactor*output


def comoving_path_proton(z: float, ke_0: np.ndarray, z_0: float, rel_only: bool = False) -> np.ndarray:
    """ Comoving path length a CR proton would travel between z_0 and z


    Note for performance it is assumed that none of the protons have been absorbed, any such protons should be filtered
    out before calling this function

    Parameters
    ----------
    z : Float
        Redshift proton is at now
    ke_0 : Array
        Initial kinetic energies of the protons in MeV
    z_0 : Float
        Source redshift of protons
    rel_only : Boolean
        Flag to only use the relativistic regime solution rather than the full solution

    Returns
    -------
    Array
        Comoving path lengths travelled in comoving Megaparsec
    """
    # Datastructure for output
    output = np.empty(np.shape(ke_0), dtype=float)

    # Handle relativistic only case
    if rel_only:
        output = comoving_path_length_rel(z, ke_0, z_0)
        return output

    # Otherwise, the simplest case is if starts non-relativistic, then as they only lose energy it stays
    # non-relativistic
    output[ke_0 <= KE_REL_TRANS_MEV] = comoving_path_length_non_rel(z, ke_0[ke_0 <= KE_REL_TRANS_MEV], z_0)

    # For relativistic case we have the added complication that the proton can change into the non-relativistic regime
    # if sufficient time has passed, this transition would occur at a redshift
    z_rel_trans = (1 + z_0) * np.sqrt((KE_REL_TRANS_MEV ** 2 + 2 * KE_REL_TRANS_MEV * PROTON_M)
                                      / (ke_0 ** 2 + 2 * ke_0 * PROTON_M)) - 1

    # If z >= z_rel_trans then this transition has not occurred yet, and so we do not have to deal with it
    output[np.logical_and(ke_0 > KE_REL_TRANS_MEV, z >= z_rel_trans)] = \
        comoving_path_length_rel(z, ke_0[np.logical_and(ke_0 > KE_REL_TRANS_MEV, z >= z_rel_trans)], z_0)

    # If z < z_rel_trans then we do need to handle the transition by summing the seperated Delta Rs for each part of the
    # protons journey
    slice_trans_p = np.logical_and(ke_0 > KE_REL_TRANS_MEV, z < z_rel_trans)
    output[slice_trans_p] = comoving_path_length_rel(z_rel_trans[slice_trans_p], ke_0[slice_trans_p], z_0) + \
        comoving_path_length_non_rel(z, KE_REL_TRANS_MEV, z_rel_trans[slice_trans_p])
    return output
