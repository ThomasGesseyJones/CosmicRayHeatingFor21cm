"""
Generate Cosmic Rays
====================
Controller script that generates the cosmic ray coefficients and window-functions required for a simulation with the
given arguments

Arguments
---------
path_to_storage : String
    Path of folder where window functions and coefficients are to be stored
window_function_type : String
    ID string of the type of window function to be generated
alpha : Float
    Exponent of cosmic ray power-law spectrum
ke_min : Float
    Minimum of cosmic ray power-law spectrum
ke_max : Float
    Maximum of cosmic ray power-law spectrum
"""

# Required modules and packages
import sys
import os
import numpy as np
import scipy.io
from typing import Tuple
from joblib import Parallel, delayed
from functools import partial
import cosmic_ray_coeffs as crc
import cosmic_ray_window_funcs as crw


# Currently fixed simulation parameters (perhaps should be taken as inputs in future)
SIMULATION_N = 128  # Number of pixels per side of the simulation box
SIMULATION_LPIX = 3.0  # Side length of a pixel in cMpc

# Behaviour controlling datastructures
window_funcs_required = {'global': False,
                         'locally_confined': False,
                         'free_streaming': True,
                         'diffusive': True}


# Input parsing
def parse_cl_input(args: list) -> Tuple:
    """ Unpack the command-line arguments into relevant python variables

    Parameters
    ----------
    args : list
        List of command-line arguments script

    Returns
    -------
    Tuple(String, String, Float, Float, Float)
        Given arguments: path_to_storage, window_function_type, alpha, ke_min, ke_max
    """

    # Check we have the correct number of arguments
    expected_nof_args = 5
    if len(args) < (expected_nof_args + 1):
        raise ValueError(f'Expected at least {expected_nof_args + 1} command line arguments, {len(args)} received.')

    # Unpack
    path_to_storage = args[1]
    alpha = float(args[3])
    ke_min = float(args[4])
    ke_max = float(args[5])

    window_function_type = args[2]
    if window_function_type not in window_funcs_required.keys():
        raise ValueError(f'Unrecognised window function type {window_function_type}. Recognised types are '
                         f'{list(window_funcs_required.keys())}')

    # Return the unpacked variables
    return path_to_storage, window_function_type, alpha, ke_min, ke_max


# Compute all the required window functions
def generate_free_streaming_window_functions(z: float, zps: np.array, cs: np.array, window_function_folder: str,
                                             alpha: float, ke_min: float, ke_max: float):
    """ Generate all the required free streaming window functions and store them to disk for a given heating redshift

    Parameters
    ----------
    z : Float
        Heating redshift window functions are being computed for
    zps : Array
        Emission redshifts window functions are to be computed for
    cs : Array
        Precomputed cosmic ray coefficients
    window_function_folder : String
        Path to the folder that will store cosmic the produced window functions
    alpha : Float
        Exponent of cosmic ray power-law spectrum
    ke_min : Float
        Minimum of cosmic ray power-law spectrum
    ke_max : Float
        Maximum of cosmic ray power-law spectrum
    """

    # Handle edge case where zps and cs only contain one item because z is at the maximum allowed value
    if len(zps) == 1:
        return

    # Skip the trivial window function when z == zp
    zps = zps[1:]
    cs = cs[1:]

    # Otherwise, calculate and store the window function in each case
    for idx, (zp, c) in enumerate(zip(zps, cs)):
        window_func = crw.free_stream_window_function(z, zp, alpha, ke_min, ke_max, SIMULATION_N, SIMULATION_LPIX, c)
        crw.save_window_function(window_func, os.path.join(window_function_folder, f'cr_w_{z}_{idx + 2}.mat'))


# Primary controller script
def main():
    # Get input and required redshifts
    path_to_storage, window_function_type, alpha, ke_min, ke_max = parse_cl_input(sys.argv)
    zs = np.arange(6, crc.C_UPPER_Z_CUTOFF + 1)

    # Generate coefficients in parallel for the required redshifts using auto-sampling of emission redshift
    auto_sampling_output = Parallel(n_jobs=-1)(
        delayed(crc.autosample_cosmic_ray_c)(z, partial(crc.cosmic_ray_c, alpha=alpha, ke_min=ke_min,
                                                        ke_max=ke_max)) for z in zs)
    cs, zps = list(zip(*auto_sampling_output))
    cs = np.array(cs, dtype='object')
    zps = np.array(zps, dtype='object')

    # Store coefficients and coefficient z_p index
    scipy.io.savemat(os.path.join(path_to_storage, f'cr_coeffs_{sys.argv[3]}_{sys.argv[4]}_{sys.argv[5]}.mat'),
                     {'cs': cs, 'zps': zps})

    # Generate corresponding window functions if required
    if window_function_type in ['global', 'locally_confined']:
        return

    window_function_folder = os.path.join(path_to_storage, f'wfs_{"_".join(sys.argv[2:])}')
    if not os.path.exists(window_function_folder):
        os.makedirs(window_function_folder)

    if window_function_type == 'free_streaming':
        Parallel(n_jobs=-1)(
            delayed(generate_free_streaming_window_functions)(z, zp, c, window_function_folder, alpha, ke_min, ke_max)
            for z, zp, c in zip(zs, zps, cs))
        return

    raise ValueError(f'Window function type {window_function_type} has not yet been Implemented')


if __name__ == "__main__":
    main()
