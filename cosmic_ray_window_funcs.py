"""
Cosmic Ray Window Functions
===========================
Compute the cosmic ray W window functions that give the heating distributions of cosmic rays in the simulation
"""

# Required modules and packages
from typing import Tuple, Callable
from scipy.integrate import quad
import cosmic_ray_general as crg
import cosmic_ray_coeffs as crc
import numpy as np
import math as mt
from numba import njit
from scipy.interpolate import interp1d
from scipy.io import savemat


# Accuracy parameters
INTERP_NUM_SAMPLES = 1_000  # When computing Ke_now(r) and W(r) how many samples should be used in interpolation
KE_R_LOW_KE_CUTOFF = 1e-6  # Lower limit on KE used in Ke_now(r) computation, in MeV
REFINE_REGION = 4  # Factor by which refined region is smaller than entire window function
REFINE_FACTOR = 2  # Factor by which refined region resolution is increased


# Geometric Functions
def distance_from_emitting_cell(length_in_cells: int, cell_side_length: float) -> np.ndarray:
    """ Compute the centre to centre distance of cells from the emission cell.

    Note this is not in PBC

    Parameters
    ----------
    length_in_cells : Int
        Number of cells along each direction the cube has, e.g. the cube has a total of length_in_cells**3 cells
    cell_side_length : Float
        Side length of an individual cell

    Returns
    -------
    Array
        length_in_cells x length_in_cells x length_in_cells array of distance each cells if from the emission cell,
        the floor(length_in_cells/2), floor(length_in_cells/2), floor(length_in_cells/2) th cell
    """
    # Setup datastructure for final result
    build_up_distances = np.zeros((length_in_cells, length_in_cells, length_in_cells))

    # Get 1D periodic distance squared
    one_d_idx = np.arange(length_in_cells)
    ems_c_idx = mt.floor(length_in_cells/2)
    one_d_p_sq_distance = (one_d_idx - ems_c_idx)**2

    # Sum over hyper-planes building up distance
    build_up_distances += one_d_p_sq_distance[:, np.newaxis, np.newaxis]
    build_up_distances += one_d_p_sq_distance[np.newaxis, :, np.newaxis]
    build_up_distances += one_d_p_sq_distance[np.newaxis, np.newaxis, :]

    # Convert to distance in cell_side_length units
    return cell_side_length*np.sqrt(build_up_distances)


# Proton evolution functions
@njit(cache=True)
def range_of_surviving_ke(z_now: float, z_ini: float, ke_min: float, ke_max: float) -> Tuple[float, float]:
    """ Compute the minimum and maximum kinetic energy of CR protons that have not been absorbed by z_now emitted
    at z_ini

    Parameters
    ----------
    z_now : float
        Current redshift
    z_ini : float
        Redshift of emission
    ke_min : float
        Minimum kinetic energy of CR protons injected into the IGM
    ke_max : float
        Maximum kinetic energy of CR protons injected into the IGM

    Returns
    -------
    Float
        Minimum kinetic energy of surviving CR protons
    Float
        Maximum kinetic energy of surviving CR protons (if 0 all CR protons are absorbed)
    """

    # Check for case that all CR protons were absorbed
    ke_ems_for_protons_being_absorbed = crg.initial_ke(0, z_now, z_ini)
    if ke_ems_for_protons_being_absorbed >= ke_max:
        return 0.0, 0.0

    # Determine if any CR protons have been absorbed and hence the minimum surviving ke
    if ke_ems_for_protons_being_absorbed >= ke_min:
        min_surviving_ke = 0
    else:
        min_surviving_ke = crg.current_ke(ke_min, z_now, z_ini)

    # Maximum surviving ke can be determined similarly
    return min_surviving_ke, crg.current_ke(ke_max, z_now, z_ini)


def comoving_path_proton_for_ke_now(z_now: float, ke_now: np.ndarray, z_ini: float, smoothing: bool = False)\
        -> np.ndarray:
    """ Comoving path length a CR proton would travel between z_0 and z

    Note this is a wrapper of comoving_path_proton but uses ke_now rather than ke_ini as an input

    Parameters
    ----------
    z_now : Float
        Redshift proton is at now
    ke_now : Array
        Kinetic energy of the protons at z_now in MeV
    z_ini : Float
        Source redshift of protons
    smoothing : Boolean
        Whether to smooth Delta R around the transition kinetic energy

    Returns
    -------
    Array
        Comoving path lengths travelled in comoving Megaparsec
    """

    # Function parameters
    smoothing_fraction = 0.5  # Fraction below KE_REL_TRANS_MEV to which smoothing is applied

    # Compute initial KE and path length without any smoothing
    v_initial_ke = np.vectorize(crg.initial_ke)
    ke_ini = v_initial_ke(ke_now, z_now, z_ini)
    path_lengths = crg.comoving_path_proton(z_now, ke_ini, z_ini)

    # If not smoothing then simply return the non-smoothed path lengths
    if not smoothing:
        return path_lengths

    # Add smoothing to value below the relativistic kinetic energy transition
    affected_idxs = np.logical_and(ke_now < crg.KE_REL_TRANS_MEV, ke_now > smoothing_fraction*crg.KE_REL_TRANS_MEV)

    frac_rel_only = (np.log(ke_now[affected_idxs]) - np.log(smoothing_fraction*crg.KE_REL_TRANS_MEV))\
        / (-np.log(smoothing_fraction))  # Smoothing performed linearly in log(KE_now)
    frac_transition = 1 - frac_rel_only

    path_lengths[affected_idxs] = frac_transition*path_lengths[affected_idxs] + \
        frac_rel_only*crg.comoving_path_proton(z_now, ke_ini[affected_idxs], z_ini, True)

    return path_lengths


def ke_now_given_r_func(z_now: float, z_ini: float, ke_now_min: float, ke_now_max: float) -> Callable:
    """ Return a function to compute ke_now given the path-length (r), i.e. T_r from Notes

    Parameters
    ----------
    z_now : Float
        Redshift of heating
    z_ini : Float
        Redshift of CR emission
    ke_now_min : Float
        Minimum kinetic energy of surviving protons
    ke_now_max
        Maximum kinetic energy of surviving protons

    Returns
    -------
    Callable
        Function to compute ke_now given r
    """

    # Compute R(ke_now) first
    if ke_now_max < KE_R_LOW_KE_CUTOFF:
        ke_nows = np.linspace(ke_now_min, ke_now_max, INTERP_NUM_SAMPLES)
    else:
        ke_nows = np.logspace(np.log10(max(ke_now_min, KE_R_LOW_KE_CUTOFF)),
                              np.log10(ke_now_max), INTERP_NUM_SAMPLES)
    r_for_ke = comoving_path_proton_for_ke_now(z_now, ke_nows, z_ini, smoothing=True)

    # Invert via interpolation (implicit monotonicity assumption)
    interp_func = interp1d(np.log(r_for_ke), np.log(ke_nows), kind='linear', bounds_error=False, fill_value=np.NAN,
                           assume_sorted=False)
    return lambda r: np.exp(interp_func(np.log(r)))  # Log-log interpolation used for improved accuracy


# Free Streaming window function computation
def free_stream_w_given_r_func(z_now: float, z_ini: float, alpha: float, ke_ini_min: float, ke_ini_max: float,
                               func_ke_now_given_r: Callable, cr_coeff: float) -> Callable:
    """ Return a function to compute the free streaming window function value given the cell distance r

    Parameters
    ----------
    z_now : Float
        Redshift at which heating is taking place
    z_ini : Float
        Redshift of emission
    alpha : Float
        Exponent of the power-law spectrum of CRs
    ke_ini_min : Float
        Minimum KE cutoff of the power-law in MeV
    ke_ini_max : Float
        Maximum KE cutoff of the power-law in MeV
    func_ke_now_given_r : Callable
        Function which computes ke_now given a travelled comoving path length r
    cr_coeff : Float
        Cosmic ray heating coefficient that has already been computed for this z_now, z_ini pair

    Returns
    -------
    Callable
        Function to compute the free-streaming window function at a given r via interpolation
    """

    # Find r values to sample at and the corresponding ke_now (sample in log delta r for improved interp accuracy)
    ke_now_min, ke_now_max = range_of_surviving_ke(z_now, z_ini, ke_ini_min, ke_ini_max)
    r_min, r_max = comoving_path_proton_for_ke_now(z_now, np.array([ke_now_min, ke_now_max]), z_ini, smoothing=True)

    delta_rs = np.logspace(np.log10(r_min*1e-6), np.log10(r_max*(1-1e-6) - r_min), INTERP_NUM_SAMPLES)
    rs = r_min + delta_rs
    ke_nows = func_ke_now_given_r(rs)

    # Compute the r' derivative term due to having delta function shells
    dke = np.maximum(ke_nows*1e-6, 1e-5)
    r_primes = (comoving_path_proton_for_ke_now(z_now, ke_nows + dke, z_ini, smoothing=True) -
                comoving_path_proton_for_ke_now(z_now, ke_nows, z_ini, smoothing=True)) / dke

    # Compute the window function at the chosen rs
    integrand_term = np.array([crc.cosmic_ray_c_integrand(ke_now, z_now, z_ini, alpha, ke_ini_min, ke_ini_max) for
                               ke_now in ke_nows])
    ws = integrand_term / (4 * cr_coeff * np.pi * (rs**2) * np.abs(r_primes))

    # Window function interpolator
    w_interp = interp1d(np.log(rs), np.log(ws), kind='linear', bounds_error=False, fill_value=-np.infty)
    return lambda r: np.exp(w_interp(np.log(r)))  # Log-log interpolation used for improved accuracy


def free_stream_window_function(z_now: float, z_ini: float, alpha: float, ke_ini_min: float, ke_ini_max: float,
                                length_in_cells: int, cell_side_length: float, cr_coeff: float) -> np.ndarray:
    """ Compute the free-streaming window function for emission at z_ini and heating at z_now

    Parameters
    ----------
    z_now : Float
        Redshift at which heating is taking place
    z_ini : Float
        Redshift of emission
    alpha : Float
        Exponent of the power-law spectrum of CRs
    ke_ini_min : Float
        Minimum KE cutoff of the power-law in MeV
    ke_ini_max : Float
        Maximum KE cutoff of the power-law in MeV
    length_in_cells : int
        Number of cells along each direction the simulation cube has, e.g. the cube is length_in_cells**3 cells total
    cell_side_length : Float
        Side length of an individual cell in cMpc
    cr_coeff : Float
        Cosmic ray heating coefficient that has been pre-computed for this z_now, z_ini pair

    Returns
    -------
    Array
        length_in_cells x length_in_cells x length_in_cells array representing the window function in real-space
    """

    # Check for valid z_now and z_ini relationship, and trivial cases
    if z_now > z_ini:
        raise ValueError(f'z_now ({z_now}) must be less than z_ini ({z_ini})')

    if z_now == z_ini:  # Causality dictates all heating is in origin cell
        window_func = np.zeros((length_in_cells, length_in_cells, length_in_cells))
        window_func[0, 0, 0] = 1
        return window_func

    ke_now_min, ke_now_max = range_of_surviving_ke(z_now, z_ini, ke_ini_min, ke_ini_max)
    if ke_now_max == 0:  # All protons absorbed so window function is ill-defined
        window_func = np.zeros((length_in_cells, length_in_cells, length_in_cells))
        return window_func

    # Find minimum and maximum distance a proton can be at
    r_min, r_max = comoving_path_proton_for_ke_now(z_now, np.array([ke_now_min, ke_now_max]), z_ini, smoothing=True)
    sim_box_border_r = cell_side_length*mt.ceil(length_in_cells/2 - 1)

    if r_min >= sim_box_border_r:  # If r_min exceeds distance to border then return a uniform window function
        return np.ones((length_in_cells, length_in_cells, length_in_cells)) / length_in_cells**3

    if r_max < cell_side_length/REFINE_FACTOR:  # In current model not travelled far enough for any to escape the cell
        window_func = np.zeros((length_in_cells, length_in_cells, length_in_cells))
        window_func[0, 0, 0] = 1
        return window_func

    # NOTE: until the fftshift back we are not working PBC from this point
    # Initialize ke_now(r) and window function interpolators and datastructure
    func_ke_now_r = ke_now_given_r_func(z_now, z_ini, ke_now_min, ke_now_max)
    func_w_r = free_stream_w_given_r_func(z_now, z_ini, alpha, ke_ini_min, ke_ini_max, func_ke_now_r, cr_coeff)
    w = np.zeros((length_in_cells, length_in_cells, length_in_cells))

    # Find distances of cells from the emitting cell
    rs = distance_from_emitting_cell(length_in_cells, cell_side_length)
    origin_position = mt.floor(length_in_cells / 2)
    rs[origin_position, origin_position, origin_position] = cell_side_length/4  # Very simple handling of origin

    # Compute window function at locations that are inside the simulation cube at normal resolution
    valid_rs = np.logical_and(rs <= min(sim_box_border_r, r_max), rs >= r_min)
    w[valid_rs] = func_w_r(rs[valid_rs])*(cell_side_length**3)  # Set the window function for that region

    # Calculate window function on the refined grid if required
    if r_min < mt.floor(length_in_cells / REFINE_REGION)*cell_side_length:
        # Get refined region size (forced to odd for isotropy) and location at original resolution
        refined_region_or_cell_size = mt.floor(length_in_cells / REFINE_REGION)
        refined_region_or_cell_size = refined_region_or_cell_size - np.mod(refined_region_or_cell_size, 2)
        refine_region = slice(mt.floor(length_in_cells/2)-mt.floor(refined_region_or_cell_size/2),
                              mt.floor(length_in_cells/2)+mt.ceil(refined_region_or_cell_size/2))

        # Get refined region size at increased resolution, +2 -2 is padding for convolution
        refined_region_rr_cell_size = (refined_region_or_cell_size+2)*REFINE_FACTOR - 2

        # Compute the window function on the refined grid
        refined_rs = distance_from_emitting_cell(refined_region_rr_cell_size, cell_side_length/REFINE_FACTOR)
        valid_refined_rs = np.logical_and(refined_rs <= r_max, refined_rs >= r_min)
        w_refined = np.zeros((refined_region_rr_cell_size, refined_region_rr_cell_size, refined_region_rr_cell_size))
        w_refined[valid_refined_rs] = func_w_r(refined_rs[valid_refined_rs])*((cell_side_length/REFINE_FACTOR)**3)

        # Convolve over possible source locations in the refined grid
        source_window_function = np.zeros(w_refined.shape)
        source_window_function[:REFINE_FACTOR, :REFINE_FACTOR, :REFINE_FACTOR] = 1/(REFINE_FACTOR**3)
        w_refined = np.real(np.fft.ifftn(np.fft.fftn(source_window_function)*np.fft.fftn(w_refined)))

        # Strip padding and coarsen
        slice_without_padding = slice(REFINE_FACTOR - 1, refined_region_rr_cell_size - REFINE_FACTOR + 1)
        w_refined = w_refined[slice_without_padding, slice_without_padding, slice_without_padding]
        w[refine_region, refine_region, refine_region] = 0
        for offset_x in range(REFINE_FACTOR):
            for offset_y in range(REFINE_FACTOR):
                for offset_z in range(REFINE_FACTOR):
                    w[refine_region, refine_region, refine_region] += \
                        w_refined[offset_x::REFINE_FACTOR, offset_y::REFINE_FACTOR, offset_z::REFINE_FACTOR]

    # For compression efficiency set any coefficient that is very small to 0. These tiny coefficients can occur at very
    # large distances and due to numerical noise from the convolution in the refinement step
    w[np.isclose(w, 0, atol=1e-10)] = 0

    # Uniform component if CRs reach outside the simulation box
    if r_max > sim_box_border_r:
        ke_escape = func_ke_now_r(sim_box_border_r)
        partial_coeff = quad(crc.cosmic_ray_c_integrand, ke_escape, np.inf,
                             args=(z_now, z_ini, alpha, ke_ini_min, ke_ini_max), epsabs=0, epsrel=1e-3, limit=100)[0]
        w = w + partial_coeff/(cr_coeff * (length_in_cells**3))

    # Shift back into PBC (as per MATLAB sim convention)
    return np.fft.fftshift(w)


# Store window function
def save_window_function(window_function: np.ndarray, file_name: str):
    """ Store the given window function in the format required by 21cmSim

    Parameters
    ----------
    window_function: Array
        The window function to be saved to disk
    file_name : String
        File to save the window function to
    """
    # Normalize
    window_function = window_function/np.sum(window_function)

    # Store compressed in fourier space
    fourier_window = np.real(np.fft.fftn(window_function))  # Only real needed as window functions are Isotropic
    fourier_window = fourier_window.astype(np.single)
    savemat(file_name, {'w': fourier_window}, do_compression=True)
