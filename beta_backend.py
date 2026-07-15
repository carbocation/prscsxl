#!/usr/bin/env python3

"""Implementations of the PRS-CS beta block update."""


import numpy as np
from scipy.linalg.blas import dtrsv
from scipy.linalg.lapack import dpotrf


def _block_layout(ld_blocks, block_sizes, variant_count):
    """Validate block inputs and return non-empty block slices."""
    if len(ld_blocks) != len(block_sizes):
        raise ValueError('ld_blocks and block_sizes must have equal length')

    sizes = [int(size) for size in block_sizes]
    if any(size < 0 for size in sizes):
        raise ValueError('block sizes must be non-negative')
    if sum(sizes) != variant_count:
        raise ValueError(
            'block sizes cover %d variants, but beta_mrg contains %d' %
            (sum(sizes), variant_count)
        )

    layout = []
    start = 0
    for block_index, (ld, size) in enumerate(zip(ld_blocks, sizes)):
        if size:
            if np.shape(ld) != (size, size):
                raise ValueError(
                    'LD block %d has shape %s; expected (%d, %d)' %
                    (block_index, np.shape(ld), size, size)
                )
            layout.append((block_index, slice(start, start + size)))
        start += size
    return layout


class CpuBetaBackend:
    """Preallocated FP64 LAPACK/BLAS beta block sampler."""

    name = 'cpu'

    def __init__(self, ld_blocks, block_sizes, beta_mrg, n_gwas):
        self._beta_mrg = np.asarray(
            beta_mrg, dtype=np.float64
        ).reshape(-1, 1)
        self._n_gwas = int(n_gwas)
        self._layout = _block_layout(
            ld_blocks, block_sizes, self._beta_mrg.shape[0]
        )
        self._beta = np.empty_like(self._beta_mrg)

        # LAPACK consumes column-major matrices. Keeping both the pristine
        # sources and workspaces in that order makes each iteration's copy
        # contiguous and avoids rebuilding temporary diagonal matrices.
        self._ld_blocks = {
            block_index: np.asfortranarray(
                ld_blocks[block_index], dtype=np.float64
            )
            for block_index, _ in self._layout
        }
        self._work = {
            block_index: np.empty_like(
                self._ld_blocks[block_index], order='F'
            )
            for block_index, _ in self._layout
        }
        maximum_size = max(
            (
                block_slice.stop - block_slice.start
                for _, block_slice in self._layout
            ),
            default=0,
        )
        self._rhs = np.empty(maximum_size, dtype=np.float64)
        self._inverse_psi = np.empty(maximum_size, dtype=np.float64)

    def sample(self, psi, sigma):
        """Draw beta for each non-empty LD block and return beta, quad."""
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._beta_mrg.shape[0]:
            raise ValueError('psi and beta_mrg must have equal length')

        sd = float(np.sqrt(float(sigma) / self._n_gwas))
        quad = 0.0

        for block_index, block_slice in self._layout:
            size = block_slice.stop - block_slice.start
            precision = self._work[block_index]
            np.copyto(precision, self._ld_blocks[block_index])

            inverse_psi = self._inverse_psi[:size]
            np.reciprocal(psi[block_slice], out=inverse_psi)
            diagonal = precision.ravel(order='K')[::size + 1]
            np.add(diagonal, inverse_psi, out=diagonal)

            # dpotrf returns U with precision = U.T @ U. clean=0 avoids
            # clearing the unused triangle, which neither dtrsv call reads.
            chol, info = dpotrf(
                precision, lower=0, overwrite_a=1, clean=0
            )
            if info < 0:
                raise ValueError(
                    'CPU Cholesky received invalid LAPACK argument %d for '
                    'LD block %d' % (-info, block_index)
                )
            if info > 0:
                raise np.linalg.LinAlgError(
                    'CPU Cholesky failed for LD block %d: leading minor %d '
                    'is not positive definite' % (block_index, info)
                )

            rhs = self._rhs[:size]
            np.copyto(rhs, self._beta_mrg[block_slice, 0])
            rhs = dtrsv(
                chol, rhs, lower=0, trans=1, diag=0, overwrite_x=1
            )
            rhs += sd * np.random.standard_normal(size)

            # U @ beta = rhs, so beta.T @ precision @ beta = ||rhs||^2.
            quad += float(np.dot(rhs, rhs))

            beta_block = self._beta[block_slice, 0]
            np.copyto(beta_block, rhs)
            solved = dtrsv(
                chol, beta_block, lower=0, trans=0, diag=0, overwrite_x=1
            )
            if solved is not beta_block:
                np.copyto(beta_block, solved)

        return self._beta, quad

    def describe(self):
        return 'cpu (direct FP64 LAPACK/BLAS; %d active LD blocks)' % len(
            self._layout
        )


def make_beta_backend(backend, ld_blocks, block_sizes, beta_mrg, n_gwas):
    """Construct the requested beta sampler."""
    backend = str(backend).lower()
    if backend == 'cpu':
        return CpuBetaBackend(ld_blocks, block_sizes, beta_mrg, n_gwas)
    raise ValueError("unknown beta backend %r; expected 'cpu'" % backend)
