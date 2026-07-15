#!/usr/bin/env python3

"""Correctness tests for the CPU local-shrinkage backend."""


import unittest

import numpy as np
from scipy.special import kv

import gigrnd
from psi_backend import CpuPsiBackend, CudaPsiBackend, make_psi_backend


def _cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class CpuPsiBackendTests(unittest.TestCase):
    def test_vector_update_matches_scalar_draw_sequence(self):
        size = 100
        delta = np.linspace(0.1, 2.0, size)
        beta = np.linspace(0.0001, 0.01, size)
        vector = np.empty(size)
        scalar = np.empty(size)

        gigrnd.seed_rng(456)
        gigrnd.gig_rvs_vec(
            vector, 0.5, delta, beta, 1.0, 200_000
        )
        gigrnd.seed_rng(456)
        for index in range(size):
            scalar[index] = gigrnd.gigrnd(
                0.5,
                2.0 * delta[index],
                200_000 * beta[index] * beta[index],
            )

        np.testing.assert_allclose(vector, scalar, rtol=1e-15, atol=1e-15)

    def test_backend_preserves_existing_seeded_stream(self):
        size = 100
        delta = np.linspace(0.1, 2.0, size)
        beta = np.linspace(0.0001, 0.01, size)
        expected = np.empty(size)
        actual = np.empty(size)

        gigrnd.seed_rng(123)
        gigrnd.gig_rvs_vec(
            expected, 0.5, delta, beta, 1.0, 200_000
        )
        gigrnd.seed_rng(123)
        CpuPsiBackend().sample(
            actual, 0.5, delta, beta, 1.0, 200_000
        )
        np.testing.assert_array_equal(actual, expected)

    def test_factory_seeds_numba_stream(self):
        size = 100
        delta = np.linspace(0.1, 2.0, size)
        beta = np.linspace(0.0001, 0.01, size)
        first = np.empty(size)
        second = np.empty(size)

        make_psi_backend('cpu', size, seed=321).sample(
            first, 0.5, delta, beta, 1.0, 200_000
        )
        make_psi_backend('cpu', size, seed=321).sample(
            second, 0.5, delta, beta, 1.0, 200_000
        )
        np.testing.assert_array_equal(first, second)

    def test_factory_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, 'unknown psi backend'):
            make_psi_backend('warp-drive', 10)


@unittest.skipUnless(_cuda_available(), 'CuPy and a CUDA device are required')
class CudaPsiBackendTests(unittest.TestCase):
    def test_seeded_backend_is_reproducible(self):
        size = 1000
        delta = np.linspace(0.2, 2.0, size)
        beta = np.linspace(0.0001, 0.01, size)
        first = np.empty(size)
        second = np.empty(size)

        CudaPsiBackend(size, seed=123).sample(
            first, 0.5, delta, beta, 1.0, 200_000
        )
        CudaPsiBackend(size, seed=123).sample(
            second, 0.5, delta, beta, 1.0, 200_000
        )
        np.testing.assert_array_equal(first, second)

    def test_draw_moments_match_gig_distribution(self):
        size = 50_000
        shape = 0.5
        gig_a = 3.0
        gig_b = 2.0
        n = 1000
        sigma = 1.0
        delta = np.full(size, gig_a / 2.0)
        beta = np.full(size, np.sqrt(gig_b * sigma / n))
        draws = np.empty(size)

        backend = CudaPsiBackend(size, seed=456)
        backend.sample(draws, shape, delta, beta, sigma, n)

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

        self.assertTrue(np.isfinite(draws).all())
        self.assertTrue((draws > 0.0).all())
        self.assertLess(
            abs(float(draws.mean()) - expected_mean),
            5.0 * mean_standard_error,
        )
        self.assertLess(
            abs(float(draws.var()) - expected_variance) / expected_variance,
            0.04,
        )
        self.assertIn('rejection rounds', backend.profile_summary())


if __name__ == '__main__':
    unittest.main()
