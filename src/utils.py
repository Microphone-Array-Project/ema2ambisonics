"""Implementations of some basic functions used in the code."""
import scipy.special as special
import numpy as np


def _derivative_sph_hankel(n, k, z):
    """
    Compute the derivative of the spherical hankel function of kind k.

    Parameters
    ----------
    n : int
        Order of the spherical Hankel function.
    k : int
        Kind of the Hankel function. 0 for the first kind and 1 for the second
        kind.
    z : array
        Argument of the Hankel function.
    """
    if n == 0:
        return -_spherical_hankel_function(1, k, z)
    else:
        return (1 / (2 * n + 1)) * \
            (n * _spherical_hankel_function(n - 1, k, z) -
             (n + 1) * _spherical_hankel_function(n + 1, k, z))


def _limiting(data, type, limit_db):
    """Apply limiting based on passed type and limit in db.

    Soft clipping as in [#]_.

    Parameters
    ----------
    data : np.array
        Data to be limited.
    type : str
        Type of limiting. 'hard' for hard limiting and 'soft' for soft
        limiting.
    limit_db : float
        Limit in dB.

    Returns
    -------
    data : np.array
        Limited data.

    References
    ----------
    .. [#] Bernschütz, B. (2016). “Microphone arrays and sound field 
           decomposition for dynamic binaural recording,” Ph.D. thesis, 
           Technische Universita€t Berlin, Berlin, Germany.
    """
    # convert dB to amplitude
    limit = 10**(limit_db / 20)

    if type == 'soft':
        data = \
            (2*limit)/np.pi * data/np.abs(data) * np.arctan(
                np.pi/(2*limit) * np.abs(data))
    elif type == 'hard':
        idx = np.where(np.abs(data) > limit)
        data[idx] = data[idx] / np.abs(data[idx]) * limit

    return data


def _spherical_hankel_function(nu, k, z):
    """
    Compute the spherical Hankel function of order nu and kind k.
    
    Parameters
    ----------
    nu : int
        Order of the Hankel function.
    k : int
        Kind of the Hankel function. 0 for the first kind and 1 for the second
        kind.
    z : array
        Argument of the Hankel function.
    """
    if k not in [1, 2]:
        raise ValueError("k must be 1 or 2.")
    
    # Determine sign, which is 1 for the first kind and -1 for the second kind
    # spherical hankel function.
    if k == 1:
        sign = 1
    elif k == 2:
        sign = -1
    
    sph_h_n = \
        special.spherical_jn(nu, z) + 1j * sign * special.spherical_yn(nu, z)

    return sph_h_n


def _tikhonov_regularization(data, limit_db):
    """Apply Tikhonov regularization to the data.

    Tikhonov regularization as used in [#]_.

    Parameters
    ----------
    data : np.array
        Data to be regularized.
    limit : float
        Limit in dB.

    Returns
    -------
    data : np.array
        Regularized data.
    
    References
    ----------
    .. [#] Moreau, Sébastien, Jérôme Daniel, und Stéphanie Bertet. „3D Sound 
           Field Recording with Higher Order Ambisonics – Objective 
           Measurements and Validation of a 4th Order Spherical Microphone“, 
           2006.

    """
    # convert dB to amplitude
    limit = 10**(limit_db / 20)

    lambda_squared = \
        (1 - np.sqrt(1 - 1/limit**2)) / (1 + np.sqrt(1 - 1/limit**2))
    data_regu = np.conj(data) / (np.abs(data)**2 + lambda_squared)
    return data_regu
