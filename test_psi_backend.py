#!/usr/bin/env python3

"""Correctness tests for the CPU local-shrinkage backend."""


import unittest

import numpy as np

import gigrnd
from psi_backend import CpuPsiBackend, make_psi_backend


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


if __name__ == '__main__':
    unittest.main()
