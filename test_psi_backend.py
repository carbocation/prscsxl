#!/usr/bin/env python3

"""Correctness tests for the CPU local-shrinkage backend."""


import unittest

import numpy as np
from scipy.special import kv

import gigrnd
from psi_backend import (
    CpuPsiBackend,
    CudaFusedPsiBackend,
    CudaPsiBackend,
    CudaRawPsiBackend,
    make_psi_backend,
)


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


@unittest.skipUnless(_cuda_available(), 'CuPy and a CUDA device are required')
class CudaRawPsiBackendTests(unittest.TestCase):
    def test_factory_selects_fused_single_kernel_backend(self):
        self.assertIsInstance(
            make_psi_backend('cuda', 10, seed=123),
            CudaFusedPsiBackend,
        )

    def test_seeded_backend_is_reproducible_across_calls(self):
        size = 1000
        delta = np.linspace(0.2, 2.0, size)
        beta = np.linspace(0.0001, 0.01, size)
        first = np.empty((2, size))
        second = np.empty((2, size))

        first_backend = CudaRawPsiBackend(size, seed=123)
        second_backend = CudaRawPsiBackend(size, seed=123)
        for index in range(2):
            first_backend.sample(
                first[index], 0.5, delta, beta, 1.0, 200_000
            )
            second_backend.sample(
                second[index], 0.5, delta, beta, 1.0, 200_000
            )

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_positive_and_negative_shape_moments_match_gig(self):
        size = 50_000
        gig_a = 3.0
        gig_b = 2.0
        n = 1000
        sigma = 1.0
        delta = np.full(size, gig_a / 2.0)
        beta = np.full(size, np.sqrt(gig_b * sigma / n))

        for index, shape in enumerate((0.5, -0.5)):
            with self.subTest(shape=shape):
                draws = np.empty(size)
                backend = CudaRawPsiBackend(size, seed=456 + index)
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
                    abs(float(draws.var()) - expected_variance) /
                    expected_variance,
                    0.04,
                )
                self.assertIn(
                    'rejection rounds', backend.profile_summary()
                )


@unittest.skipUnless(_cuda_available(), 'CuPy and a CUDA device are required')
class CudaFusedPsiBackendTests(unittest.TestCase):
    def test_seeded_joint_update_is_reproducible(self):
        size = 1000
        old_psi = np.linspace(0.2, 1.0, size)
        beta = np.linspace(0.0001, 0.01, size)
        first = np.empty(size)
        second = np.empty(size)

        first_sum = CudaFusedPsiBackend(size, seed=123).sample_joint(
            first, 0.5, 1.5, old_psi, 0.1, beta, 1.0, 200_000,
            need_delta_sum=True,
        )
        second_sum = CudaFusedPsiBackend(size, seed=123).sample_joint(
            second, 0.5, 1.5, old_psi, 0.1, beta, 1.0, 200_000,
            need_delta_sum=True,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_sum, second_sum)

    def test_joint_update_matches_cpu_transition_moments(self):
        size = 100_000
        old_psi = np.full(size, 0.8)
        phi = 0.2
        gamma_shape = 1.5
        gig_shape = 0.5
        n = 1000
        sigma = 1.0
        beta = np.full(size, np.sqrt(2.0 * sigma / n))
        actual = np.empty(size)

        backend = CudaFusedPsiBackend(size, seed=456)
        delta_sum = backend.sample_joint(
            actual, gig_shape, gamma_shape, old_psi, phi, beta,
            sigma, n, need_delta_sum=True,
        )

        delta_scale = 1.0 / (old_psi[0] + phi)
        expected_delta_sum = size * gamma_shape * delta_scale
        delta_sum_sd = np.sqrt(size * gamma_shape) * delta_scale
        self.assertLess(
            abs(delta_sum - expected_delta_sum), 5.0 * delta_sum_sd
        )
        delta_draws = backend._cp.asnumpy(backend._delta)
        expected_delta_variance = gamma_shape * delta_scale**2
        self.assertTrue((delta_draws > 0.0).all())
        self.assertLess(
            abs(float(delta_draws.var()) - expected_delta_variance) /
            expected_delta_variance,
            0.03,
        )

        np.random.seed(789)
        gigrnd.seed_rng(789)
        reference_delta = np.random.gamma(
            gamma_shape, delta_scale, size
        )
        reference = np.empty(size)
        gigrnd.gig_rvs_vec(
            reference, gig_shape, reference_delta, beta, sigma, n
        )

        # Mixing over delta gives the untruncated positive-shape draw an
        # infinite second moment here. PRS-CS immediately caps psi at 1, so
        # compare the bounded transition the chain actually consumes.
        actual_clipped = np.minimum(actual, 1.0)
        reference_clipped = np.minimum(reference, 1.0)
        combined_mean_se = np.sqrt(
            (actual_clipped.var() + reference_clipped.var()) / size
        )
        self.assertLess(
            abs(float(actual_clipped.mean() - reference_clipped.mean())),
            6.0 * combined_mean_se,
        )
        self.assertLess(
            abs(float(actual_clipped.var() - reference_clipped.var())) /
            float(reference_clipped.var()),
            0.05,
        )
        self.assertLess(
            abs(float((actual >= 1.0).mean()) -
                float((reference >= 1.0).mean())),
            0.01,
        )
        self.assertIn('delta + GIG', backend.profile_summary())


if __name__ == '__main__':
    unittest.main()
