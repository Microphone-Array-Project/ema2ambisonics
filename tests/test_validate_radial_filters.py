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
import pyfar as pf

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module
from src.ema_radial_filters import radial_filters_ema


class ValidateEMARadialFilters(unittest.TestCase):
    impulse = pf.signals.impulse(2048, sampling_rate=48000)

    def test_compare_tikhonov_40db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using Tikhonov regularization with a
        limit of 40 dB.

        Radius of EMA: 0.0875 m
        Order: 7
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_tikhonov_40db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = radial_filters_ema(f, 0.0875, 7, 40, 'tikhonov')
        filt_response = radial_filters.process(self.impulse)

        np.testing.assert_allclose(filt_response.time.squeeze(),
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
        radial_filters = radial_filters_ema(f, 0.0875, 7, 40, 'soft')
        filt_response = radial_filters.process(self.impulse)

        np.testing.assert_allclose(filt_response.time.squeeze(),
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
        radial_filters = radial_filters_ema(f, 0.0875, 7, 40, 'hard')
        filt_response = radial_filters.process(self.impulse)

        np.testing.assert_allclose(filt_response.time.squeeze(),
                                   expected['ema_inv_rf_t'].T)

    def test_compare_tikhonov_20db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using Tikhonov regularization with a
        limit of 20 dB.

        Radius of EMA: 0.15 m
        Order: 13
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_tikhonov_20db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = radial_filters_ema(f, 0.15, 13, 20, 'tikhonov')
        filt_response = radial_filters.process(self.impulse)

        np.testing.assert_allclose(filt_response.time.squeeze(),
                                   expected['ema_inv_rf_t'].T)

    def test_compare_soft_limiting_20db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using soft limiting with a
        limit of 20 dB.

        Radius of EMA: 0.15 m
        Order: 13
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_soft_20db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = radial_filters_ema(f, 0.15, 13, 20, 'soft')
        filt_response = radial_filters.process(self.impulse)

        np.testing.assert_allclose(filt_response.time.squeeze(),
                                   expected['ema_inv_rf_t'].T)

    def test_compare_hard_limiting_20db(self):
        """
        Compare the ema radial filter implementation with results from Matlab.

        The radial filters are computed using hard limiting with a
        limit of 20 dB.

        Radius of EMA: 0.15 m
        Order: 13
        Filter lentgh: 2048
        """
        expected = sc.io.loadmat(r'tests/resources/ema_rf_t_hard_20db.mat')

        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)
        radial_filters = radial_filters_ema(f, 0.15, 13, 20, 'hard')
        filt_response = radial_filters.process(self.impulse)

        np.testing.assert_allclose(filt_response.time.squeeze(),
                                   expected['ema_inv_rf_t'].T)
