"""
This file contains validation tests for the ambisonics signals.
"""

import sys
import os
import unittest
import scipy as sc
import numpy as np
import pyfar as pf

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module
from src.ema_radial_filters import radial_filters_ema
from src.get_soundfield_coeffs_from_ema import get_sh_soundfield_coeffs_ema


class ValidateSHSoundfieldCoefficients(unittest.TestCase):
    def test_compare_unnormalized_ambi_signals(self):
        """
        Compare the unnormalized ambisonics signals implementation with 
        results from Matlab.

        The radial filters are computed using Tikhonov regularization with a
        limit of 40 dB.

        Radius of EMA: 0.0875 m
        Order: 7
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ambi_signals_raw.mat')
        signals = sc.io.loadmat(r'tests/resources/ema_recording_chalmers.mat')

        array_signals = signals['array_signals'].T
        signals = pf.Signal(array_signals, sampling_rate=signals['fs'])

        alpha = np.linspace(0, 2*np.pi, 16, endpoint=False)
        alpha = np.roll(np.flip(alpha), 1)

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = radial_filters_ema(f, 0.0875, 7, 40, 'tikhonov')

        ambisonics_signals = \
            get_sh_soundfield_coeffs_ema(signals, radial_filters, 7, alpha)

        np.testing.assert_allclose(ambisonics_signals.time,
                                   expected['ambi_signals_raw'].T, atol=1e-15)
