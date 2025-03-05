"""
Get sh soundfield coefficients from EMA data as in Eq. 13 / 15 in [1].
Requires ema radial filters.

[1] Ahrens, J. (2022). Ambisonic Encoding of Signals From Equatorial Microphone
    Arrays (No. arXiv:2211.00584). arXiv. http://arxiv.org/abs/2211.00584
"""

import numpy as np
import scipy.special as special
import scipy.signal as spsignal
import pyfar as pf


def get_sh_soundfield_coeffs_ema(signals, radial_filters, N, alpha):
    r"""
    Get coeffcients of the soundfield in spherical harmonics domain 
    from EMA signals.
    """
    # flatten input ararys tbd

    # Soundfield coefficients on surface
    s_surf_m = np.zeros((2*N+1, signals.n_samples))

    # Evaluate eq. 8
    # for m < 0
    for m in range(-N, 0):
        s_surf_m[m + N, :] = \
            np.sum(signals.time * np.sqrt(2) *
                   np.sin(np.abs(m) * alpha)[:, None],
                   axis=0) / signals.cshape[0]

    # for m = 0
    s_surf_m[0+N, :] = np.sum(signals.time, axis=0) / signals.cshape[0]

    # for m > 0
    for m in range(1, N+1):
        s_surf_m[m + N, :] = \
            np.sum(signals.time * np.sqrt(2) * np.cos(m * alpha)[:, None],
                   axis=0) / signals.cshape[0]

    s_surf_m = pf.Signal(s_surf_m, signals.sampling_rate)

    # Get coefficients of soundfield by multiplying with radial filters eq. 13
    # Realization as time domain convoluton
    s_sh = np.zeros((2*N+1, signals.n_samples))

    for i in range(2*N + 1):
        b = radial_filters.time[i, :]
        data = s_surf_m.time[i, :]

        out_full = spsignal.oaconvolve(data, b, mode='full')
        # Ensure that the output has the same shape as the input
        # logic from scipy.signal.lfilter
        idx = out_full.shape[-1] - radial_filters.n_samples + 1
        s_sh[i, :] = out_full[..., 0:idx]

    # get ambisonics signal by multiplying with spherical harmonics function
    # Eq. 15
    s_ambisonics = np.zeros(((N+1)**2, signals.n_samples))

    for n in range(N+1):
        for m in range(-n, n+1):
            # Note that since we are using abs(m) as an input parameter, we
            # have to add the condon shortley phase factor (-1)^m
            s_ambisonics[n**2 + n + m, :] = \
                s_sh[m+N, :] * (-1)**m*special.sph_harm(abs(m), n, 0, np.pi/2)

    return pf.Signal(s_ambisonics, signals.sampling_rate)
