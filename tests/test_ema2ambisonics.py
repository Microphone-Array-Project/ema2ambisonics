"""
Test for steps in encoding using unittests.
"""

import sys
import os
import unittest
import re
import numpy as np

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module
from src.ema_radial_filters import radial_filters_ema


class TestEMARadialFilters(unittest.TestCase):
    def test_warnings_and_errors(self):
        filter_length = 2048
        f = np.linspace(0, 48e3/2, filter_length//2+1)

        with self.assertRaisesRegex(ValueError, '`R` must be a single value'):
            radial_filters_ema(f, (1, 2), 7, 40, 'tikhonov')

        with self.assertRaisesRegex(ValueError,
                                    '`limit_dB` must be a single value'):
            radial_filters_ema(f, 0.0875, 7, None, 'tikhonov')

        with self.assertRaisesRegex(ValueError,
                                    re.escape('`hankel_type` must be 1 or 2 '
                                              '(for first or second kind '
                                              'hankel function)')):
            radial_filters_ema(f, 0.0875, 7, 20, 'tikhonov', hankel_type=3)

        with self.assertRaisesRegex(ValueError,
                                    "Invalid regularization type. Choose "
                                    "'soft', 'hard' or 'tikhonov'."):
            radial_filters_ema(f, 0.875, 7, 40, 'asd')
