"""
This file contains validation tests for the EMA radial filters.

The results from the implementation are compared with results from Matlab
using different parameter configurations.
"""

import sys
import os
import unittest
import scipy as sc
import numpy as np

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module
from src.ema_radial_filters import inverse_radial_filters_ema


class ValidateEMARadialFilters(unittest.TestCase):

    def test_compare_tikhonov_40db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using Tikhonov regularization with a
        limit of 40 dB.

        Radius of EMA: 0.0875 m
        Order: 7
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_thikonov_40db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = inverse_radial_filters_ema(f, 0.0875, 7, 40,
                                                    'tikhonov')

        np.testing.assert_allclose(radial_filters.time,
                                   expected['ema_inv_rf_t'].T)

    def test_compare_soft_limiting_40db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using soft limiting with a
        limit of 40 dB.

        Radius of EMA: 0.0875 m
        Order: 7
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_soft_40db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = inverse_radial_filters_ema(f, 0.0875, 7, 40,
                                                    'soft')

        np.testing.assert_allclose(radial_filters.time,
                                   expected['ema_inv_rf_t'].T)
        
    def test_compare_hard_limiting_40db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using hard limiting with a
        limit of 40 dB.

        Radius of EMA: 0.0875 m
        Order: 7
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_hard_40db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = inverse_radial_filters_ema(f, 0.0875, 7, 40,
                                                    'hard')

        np.testing.assert_allclose(radial_filters.time,
                                   expected['ema_inv_rf_t'].T)

