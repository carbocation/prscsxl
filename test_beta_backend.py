#!/usr/bin/env python3

"""Correctness tests for the CPU beta block backend."""


import unittest

import numpy as np
from scipy import linalg

from beta_backend import (
    CpuBetaBackend,
    CudaBetaBackend,
    CudaDirectBetaBackend,
    CudaHybridBetaBackend,
    CudaStreamsBetaBackend,
    diagnose_ld_blocks,
    format_ld_diagnostics,
    ld_layout_diagnostics,
    make_beta_backend,
)


def _cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _inputs():
    rng = np.random.default_rng(7)
    sizes = [3, 0, 5, 7]
    blocks = []
    for size in sizes:
        if not size:
            blocks.append(np.array([]))
            continue
        values = rng.normal(size=(size, size))
        blocks.append(values @ values.T / size + np.eye(size) * 0.2)

    variant_count = sum(sizes)
    beta_mrg = rng.normal(size=(variant_count, 1))
    psi = rng.uniform(0.1, 1.0, size=(variant_count, 1))
    return blocks, sizes, beta_mrg, psi


def _legacy_sample(blocks, sizes, beta_mrg, psi, sigma, n_gwas):
    beta = np.zeros_like(beta_mrg)
    quad = 0.0
    start = 0
    for ld, size in zip(blocks, sizes):
        if not size:
            continue
        block_slice = slice(start, start + size)
        precision = ld + np.diag(1.0 / psi[block_slice, 0])
        chol = linalg.cholesky(precision)
        beta_tmp = linalg.solve_triangular(
            chol, beta_mrg[block_slice], trans='T'
        )
        beta_tmp += np.sqrt(sigma / n_gwas) * np.random.standard_normal(
            (size, 1)
        )
        beta[block_slice] = linalg.solve_triangular(chol, beta_tmp)
        quad += (
            beta[block_slice].T @ precision @ beta[block_slice]
        ).item()
        start += size
    return beta, quad


class CpuBetaBackendTests(unittest.TestCase):
    def test_matches_original_block_update(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        n_gwas = 1000

        np.random.seed(123)
        expected_beta, expected_quad = _legacy_sample(
            blocks, sizes, beta_mrg, psi, sigma, n_gwas
        )
        np.random.seed(123)
        actual_beta, actual_quad = CpuBetaBackend(
            blocks, sizes, beta_mrg, n_gwas
        ).sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-12, atol=1e-12
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=12)

    def test_reuses_workspaces_without_mutating_sources(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        original_blocks = [block.copy() for block in blocks]
        backend = CpuBetaBackend(blocks, sizes, beta_mrg, 1000)

        np.random.seed(456)
        first_beta, first_quad = backend.sample(psi, 0.7)
        first_beta = first_beta.copy()
        np.random.seed(789)
        second_beta, second_quad = backend.sample(psi, 0.9)
        second_beta = second_beta.copy()

        np.random.seed(456)
        expected_first_beta, expected_first_quad = _legacy_sample(
            blocks, sizes, beta_mrg, psi, 0.7, 1000
        )
        np.random.seed(789)
        expected_second_beta, expected_second_quad = _legacy_sample(
            blocks, sizes, beta_mrg, psi, 0.9, 1000
        )

        np.testing.assert_allclose(
            first_beta, expected_first_beta, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            second_beta, expected_second_beta, rtol=1e-12, atol=1e-12
        )
        self.assertAlmostEqual(first_quad, expected_first_quad, places=12)
        self.assertAlmostEqual(second_quad, expected_second_quad, places=12)
        for actual, expected in zip(blocks, original_blocks):
            np.testing.assert_array_equal(actual, expected)
        for source in backend._ld_blocks.values():
            self.assertTrue(source.flags.f_contiguous)

    def test_reports_cholesky_failure(self):
        backend = CpuBetaBackend(
            [-2.0 * np.eye(2)], [2], np.ones((2, 1)), 1000
        )
        with self.assertRaisesRegex(
                np.linalg.LinAlgError, 'CPU Cholesky failed for LD block 0'):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_rejects_inconsistent_layout(self):
        blocks, sizes, beta_mrg, _ = _inputs()
        sizes = list(sizes)
        sizes[-1] -= 1
        with self.assertRaisesRegex(ValueError, 'block sizes cover'):
            CpuBetaBackend(blocks, sizes, beta_mrg, 1000)

    def test_factory_rejects_unknown_backend(self):
        blocks, sizes, beta_mrg, _ = _inputs()
        with self.assertRaisesRegex(ValueError, 'unknown beta backend'):
            make_beta_backend('quantum', blocks, sizes, beta_mrg, 1000)

    def test_layout_diagnostics_measure_padding_cost(self):
        diagnostics = ld_layout_diagnostics(
            [3, 0, 5, 7], bucket_size=4
        )
        self.assertEqual(diagnostics['active_blocks'], 3)
        self.assertEqual(diagnostics['variants'], 15)
        self.assertAlmostEqual(
            diagnostics['padding_memory_ratio'],
            (4**2 + 8**2 + 8**2) / float(3**2 + 5**2 + 7**2),
        )
        self.assertAlmostEqual(
            diagnostics['padding_cubic_ratio'],
            (4**3 + 8**3 + 8**3) / float(3**3 + 5**3 + 7**3),
        )

    def test_rank_diagnostics_detect_low_rank_ld(self):
        block = np.diag([3.0, 1.0, 1e-12])
        diagnostics = diagnose_ld_blocks(
            [block], [3], bucket_size=1, rank_rtol=1e-8
        )
        self.assertEqual(diagnostics['rank_min'], 2)
        self.assertEqual(diagnostics['rank_max'], 2)
        self.assertAlmostEqual(diagnostics['rank_fraction_median'], 2/3)
        self.assertIn(
            'numerical rank', format_ld_diagnostics(diagnostics)
        )


@unittest.skipUnless(_cuda_available(), 'CuPy and a CUDA device are required')
class CudaBetaBackendTests(unittest.TestCase):
    def test_factory_selects_streamed_cuda_backend(self):
        blocks, sizes, beta_mrg, _ = _inputs()
        self.assertIsInstance(
            make_beta_backend(
                'cuda', blocks, sizes, beta_mrg, 1000,
                seed=123, cuda_bucket_size=4,
            ),
            CudaStreamsBetaBackend,
        )

    def test_direct_backend_reports_cholesky_failure(self):
        backend = CudaDirectBetaBackend(
            [-2.0 * np.eye(2)],
            [2],
            np.ones((2, 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
        )
        with self.assertRaisesRegex(
                RuntimeError, 'CUDA Cholesky failed'):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_direct_backend_matches_generic_cuda_draw(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        generic = CudaBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        direct = CudaDirectBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )

        expected_beta, expected_quad = generic.sample(psi, sigma)
        actual_beta, actual_quad = direct.sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-11, atol=1e-11
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=10)
        self.assertIn('potrfBatched', direct.describe())

    def test_hybrid_backend_matches_direct_cuda_draw(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        direct = CudaDirectBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        hybrid = CudaHybridBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )

        expected_beta, expected_quad = direct.sample(psi, sigma)
        actual_beta, actual_quad = hybrid.sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-10, atol=1e-10
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=10)
        self.assertIn('regular potrf/trsm for 3 matrices', hybrid.describe())

    def test_hybrid_backend_keeps_dense_buckets_batched(self):
        blocks = [np.eye(2)] * 8
        backend = CudaHybridBetaBackend(
            blocks,
            [2] * len(blocks),
            np.ones((2 * len(blocks), 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
        )

        self.assertIn('batched for 8 matrices in 1 buckets', backend.describe())

    def test_hybrid_backend_reports_cholesky_failure(self):
        backend = CudaHybridBetaBackend(
            [-2.0 * np.eye(2)],
            [2],
            np.ones((2, 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
        )
        with self.assertRaisesRegex(RuntimeError, 'CUDA Cholesky failed'):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_stream_backend_matches_hybrid_cuda_draw(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        hybrid = CudaHybridBetaBackend(
            blocks, sizes, beta_mrg, 1000,
            seed=123, cuda_bucket_size=4,
        )
        streamed = CudaStreamsBetaBackend(
            blocks, sizes, beta_mrg, 1000,
            seed=123, cuda_bucket_size=4, cuda_streams=2,
        )

        expected_beta, expected_quad = hybrid.sample(psi, sigma)
        actual_beta, actual_quad = streamed.sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-10, atol=1e-10
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=10)
        self.assertIn('2 concurrent streams', streamed.describe())

    def test_stream_backend_is_seeded_reproducibly(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        backends = [
            CudaStreamsBetaBackend(
                blocks, sizes, beta_mrg, 1000,
                seed=987, cuda_bucket_size=4, cuda_streams=2,
            )
            for _ in range(2)
        ]

        first_beta, first_quad = backends[0].sample(psi, 0.7)
        second_beta, second_quad = backends[1].sample(psi, 0.7)

        np.testing.assert_array_equal(first_beta, second_beta)
        self.assertEqual(first_quad, second_quad)

    def test_stream_backend_reports_cholesky_failure(self):
        backend = CudaStreamsBetaBackend(
            [-2.0 * np.eye(2)], [2], np.ones((2, 1)), 1000,
            seed=123, cuda_bucket_size=1, cuda_streams=2,
        )
        with self.assertRaisesRegex(RuntimeError, 'CUDA Cholesky failed'):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_irregular_padded_blocks_have_correct_quadratic_form(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        backend = CudaBetaBackend(
            blocks, sizes, beta_mrg, 1000,
            seed=123, cuda_bucket_size=4,
        )
        beta, quad = backend.sample(psi, 0.7)

        expected_quad = 0.0
        start = 0
        for ld, size in zip(blocks, sizes):
            if not size:
                continue
            block_slice = slice(start, start + size)
            precision = ld + np.diag(1.0 / psi[block_slice, 0])
            expected_quad += (
                beta[block_slice].T @ precision @ beta[block_slice]
            ).item()
            start += size

        self.assertTrue(np.isfinite(beta).all())
        self.assertAlmostEqual(quad, expected_quad, places=9)

    def test_seeded_backend_is_reproducible(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        backends = [
            CudaBetaBackend(
                blocks, sizes, beta_mrg, 1000,
                seed=987, cuda_bucket_size=4,
            )
            for _ in range(2)
        ]
        first_beta, first_quad = backends[0].sample(psi, 0.7)
        second_beta, second_quad = backends[1].sample(psi, 0.7)

        np.testing.assert_array_equal(first_beta, second_beta)
        self.assertEqual(first_quad, second_quad)


if __name__ == '__main__':
    unittest.main()
