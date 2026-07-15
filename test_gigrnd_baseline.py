#!/usr/bin/env python3

"""Distributional characterization tests for the existing GIG sampler."""


import unittest

import numpy as np
from numba import njit
from scipy.special import kv

import gigrnd


@njit
def _seeded_draws(shape, gig_a, gig_b, size, seed):
    np.random.seed(seed)
    draws = np.empty(size)
    for index in range(size):
        draws[index] = gigrnd.gigrnd(shape, gig_a, gig_b)
    return draws


class GigSamplerCharacterizationTests(unittest.TestCase):
    def test_draws_are_reproducible_positive_and_match_moments(self):
        size = 30_000
        shape = 0.5
        gig_a = 3.0
        gig_b = 2.0
        first = _seeded_draws(shape, gig_a, gig_b, size, 123)
        second = _seeded_draws(shape, gig_a, gig_b, size, 123)

        omega = np.sqrt(gig_a * gig_b)
        expected_mean = (
            np.sqrt(gig_b / gig_a) *
            kv(shape + 1.0, omega) / kv(shape, omega)
        )
        expected_second = (
            (gig_b / gig_a) *
            kv(shape + 2.0, omega) / kv(shape, omega)
        )
        expected_variance = expected_second - expected_mean**2
        mean_standard_error = np.sqrt(expected_variance / size)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first > 0.0).all())
        self.assertLess(
            abs(float(first.mean()) - expected_mean),
            5.0 * mean_standard_error,
        )
        self.assertLess(
            abs(float(first.var()) - expected_variance) /
            expected_variance,
            0.05,
        )


if __name__ == '__main__':
    unittest.main()
