"""
Implementation of ema radial filters, as first part of Eq. 13 in [1].

[1] Ahrens, J. (2022). Ambisonic Encoding of Signals From Equatorial Microphone
    Arrays (No. arXiv:2211.00584). arXiv. http://arxiv.org/abs/2211.00584
"""

import scipy.special as special
import scipy as sc
import numpy as np
import pyfar as pf
from numbers import Number
from .utils import _derivative_sph_hankel, _limiting, _tikhonov_regularization


def radial_filters_ema(f, R, N, limit_dB,
                       regularization_type, hankel_type=2, sampling_rate=48e3):
    r"""
    Compute the radial filters used in [1]_ for encoding signals from
    equatorial microphone arrays (EMAs) to spherical harmonics.

    Parameters
    ----------
    f : array
        Frequencies in Hz at which the filters are calculated.
    R : float
        Radius of the microphone array in meters.
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
    sampling_rate: int
        Sampling rate of signals to be filtered. Necessary for design of FIR
        filter. Default is 48000 Hz.

    Returns
    -------
    radial_filters : pyfar.FIRFilter
        Radial filters for encoding signals from EMAs to spherical harmonics.

    References
    ----------
    .. [1] Ahrens, J. (2022). Ambisonic Encoding of Signals From Equatorial
    Microphone Arrays (No. arXiv:2211.00584). arXiv.
    http://arxiv.org/abs/2211.00584
    """
    # Check input values
    if not isinstance(R, Number):
        raise ValueError('`R` must be a single value')
    if not isinstance(limit_dB, Number):
        raise ValueError('`limit_dB` must be a single value')
    if hankel_type not in [1, 2]:
        raise ValueError('`hankel_type` must be 1 or 2 (for first or '
                         'second kind hankel function)')

    k = 2 * np.pi * f / 343
    b_n = np.zeros((N + 1, k.shape[0]), dtype=np.complex128)
    kR = k * R + 5 * np.finfo(float).eps

    # calculate derivative of 2nd type hankel function h_n'(w R/c) with
    # respect to its argument w R/c as used in Eq. 7.
    for n in range(0, N + 1):
        hankel_derivative = _derivative_sph_hankel(n, hankel_type, kR)
        b_n[n, :] = -4 * np.pi * 1j**n * (1j / kR**2) * (1 / hankel_derivative)

    # catch NANs (Usually happens at DC bin)
    idx_nan = np.where(np.isnan(b_n))
    # replace NANs
    b_n[idx_nan] = np.abs(np.roll(b_n, -1, axis=-1)[idx_nan])

    radial_filters = np.zeros((2*N+1, b_n.shape[-1]), dtype=b_n.dtype)

    # iterate over degree m and order n
    for m in range(-N, N+1):
        for n_prime in range(np.abs(m), N+1):
            radial_filters[m + n, :] += \
                b_n[n_prime, :] * special.sph_harm(m, n_prime, 0, np.pi/2)**2

    # invert the radial filters
    inverse_radial_filter = 1 / radial_filters

    # dB limiting / regularization
    if regularization_type in ['soft', 'hard']:
        inverse_radial_filter = _limiting(inverse_radial_filter,
                                          regularization_type, limit_dB)
    elif regularization_type == 'tikhonov':
        inverse_radial_filter = _tikhonov_regularization(radial_filters,
                                                         limit_dB)
    else:
        raise ValueError("Invalid regularization type. Choose 'soft',"
                         " 'hard' or 'tikhonov'.")

    # catch NANs introduced by inversion
    idx_nan = np.where(np.isnan(inverse_radial_filter))
    # replace NANs
    inverse_radial_filter[idx_nan] = \
        np.abs(np.roll(inverse_radial_filter, -1, axis=-1)[idx_nan])

    # inverse fourier transform to get time-signal
    inverse_radial_filter_t = sc.fft.irfft(inverse_radial_filter)

    # create pyfar.Signal object
    inverse_radial_filter_t = pf.Signal(data=inverse_radial_filter_t,
                                        sampling_rate=sampling_rate)

    # shift signal to get causal filter
    shift = inverse_radial_filter_t.n_samples / 2
    inverse_radial_filter_t = \
        pf.dsp.fractional_time_shift(inverse_radial_filter_t, shift,
                                     mode='cyclic')

    return inverse_radial_filter_t
