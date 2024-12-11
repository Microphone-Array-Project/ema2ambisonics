"""
Implementation of ema radial filters, as first part of Eq. 13 in [1].

[1] Ahrens, J. (2022). Ambisonic Encoding of Signals From Equatorial Microphone
    Arrays (No. arXiv:2211.00584). arXiv. http://arxiv.org/abs/2211.00584
"""
#%%
import scipy.special as special
import scipy as sc
import numpy as np
import pyfar as pf
from .utils import _derivative_sph_hankel, _limiting, _tikhonov_regularization


def inverse_radial_filters_ema(f, R, N, limit_dB=None,
                               regularization_type=None, hankel_type=2):
    r"""
    Compute the radial filters used in [1]_ for encoding signals from
    equatorial microphone arrays (EMAs) to spherical harmonics.

    Parameters
    ----------
    f : array
        Frequencies in Hz at which the filters are calculated.
    R : float
        Radius of the microphone array.
    N : int
        Max order of array.
    limit_dB : float
        Maximum allowed gain in dB for the filters.
    regularization_type : string
        Type of regularization used for regularized inversion.
        
        'soft'
            Soft limiting.
        'hard'
            Hard limiting.
        'tikhonov'
            Tikhonov regularization 
    hankel_type : int
        Type of Hankel function. 1 for Hankel function of the first kind and 2
        for Hankel function of the second kind. Default is 2.

    Returns
    -------
    radial_filters : array
        Radial filters for encoding signals from EMAs to spherical harmonics.

    References
    ----------
    .. [1] Ahrens, J. (2022). Ambisonic Encoding of Signals From Equatorial
    Microphone Arrays (No. arXiv:2211.00584). arXiv.
    http://arxiv.org/abs/2211.00584
    """
    k = 2 * np.pi * f / 343
    b_n = np.zeros((N + 1, k.shape[0]), dtype=np.complex128)
    kR = k * R + 5 * np.finfo(float).eps

    # calculate derivative of 2nd type hankel function h_n'(w R/c) with
    # respect to its argument w R/c as used in Eq. 7.
    for n in range(0, N + 1):
        hankel_derivative = _derivative_sph_hankel(n, hankel_type, kR)
        b_n[n, :] = -4 * np.pi * 1j**n * (1j / kR**2) * (1 / hankel_derivative)

    # catch NANs
    idx_nan = np.argwhere(np.isnan(b_n))
    b_n[idx_nan] = np.abs(b_n[idx_nan+1])

    radial_filters = np.zeros((2*N+1, b_n.shape[-1]), dtype=b_n.dtype)

    # iterate over degree m and order n
    for m in range(-N, N+1):
        for n_prime in range(np.abs(m), N+1):
            radial_filters[m + n, :] += \
                b_n[n_prime, :] * special.sph_harm(m, n_prime, 0, np.pi/2)**2

    # inverse
    inverse_radial_filter = 1 / radial_filters

    if limit_dB is not None:
        # dB limiting / regularization
        if regularization_type in ['soft', 'hard']:
            inverse_radial_filter = _limiting(inverse_radial_filter,
                                              regularization_type, limit_dB)
        elif regularization_type == 'tikhonov':
            inverse_radial_filter = _tikhonov_regularization(radial_filters,
                                                             limit_dB)
        else:
            raise ValueError("Invalid regularization parameter. Choose 'soft'," 
                             " 'hard' or 'tikhonov'.")

    # catch NANs introduced by inversion
    idx_nan = np.argwhere(np.isnan(inverse_radial_filter))
    inverse_radial_filter[idx_nan] = np.abs(inverse_radial_filter[idx_nan+1])

    inverse_radial_filter_t = sc.fft.irfft(inverse_radial_filter)

    inverse_radial_filter_t = pf.Signal(data=inverse_radial_filter_t,
                                        sampling_rate=48e3)

    shift = inverse_radial_filter_t.n_samples / 2

    inverse_radial_filter_t = \
        pf.dsp.fractional_time_shift(inverse_radial_filter_t, shift, 
                                     mode='cyclic')

    return inverse_radial_filter_t
