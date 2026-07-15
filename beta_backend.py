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


def _fixed_block_groups(layout, bucket_size):
    """Group blocks by a fixed padded-size interval."""
    buckets = {}
    for block_index, block_slice in layout:
        size = block_slice.stop - block_slice.start
        padded_size = (
            (size + bucket_size - 1) // bucket_size
        ) * bucket_size
        buckets.setdefault(padded_size, []).append(
            (block_index, block_slice)
        )
    return sorted(buckets.items())


def ld_layout_diagnostics(block_sizes, bucket_size=32):
    """Return size and padding diagnostics for an LD block layout."""
    bucket_size = int(bucket_size)
    if bucket_size < 1:
        raise ValueError('bucket_size must be at least 1')

    sizes = np.asarray([int(size) for size in block_sizes if int(size) > 0])
    if not sizes.size:
        return {
            'active_blocks': 0,
            'variants': 0,
            'size_min': 0,
            'size_median': 0.0,
            'size_p90': 0.0,
            'size_max': 0,
            'padding_memory_ratio': 1.0,
            'padding_cubic_ratio': 1.0,
        }

    start = 0
    layout = []
    for block_index, size in enumerate(block_sizes):
        size = int(size)
        if size > 0:
            layout.append((block_index, slice(start, start + size)))
        start += max(size, 0)
    groups = _fixed_block_groups(layout, bucket_size)
    padded_memory = sum(
        len(blocks) * float(padded_size) ** 2
        for padded_size, blocks in groups
    )
    padded_cubic = sum(
        len(blocks) * float(padded_size) ** 3
        for padded_size, blocks in groups
    )
    return {
        'active_blocks': int(sizes.size),
        'variants': int(sizes.sum()),
        'size_min': int(sizes.min()),
        'size_median': float(np.median(sizes)),
        'size_p90': float(np.percentile(sizes, 90)),
        'size_max': int(sizes.max()),
        'padding_memory_ratio': float(
            padded_memory / np.sum(sizes.astype(np.float64) ** 2)
        ),
        'padding_cubic_ratio': float(
            padded_cubic / np.sum(sizes.astype(np.float64) ** 3)
        ),
    }


def diagnose_ld_blocks(ld_blocks, block_sizes, bucket_size=32,
                       rank_rtol=1e-8):
    """Measure padding, definiteness, and numerical rank of LD blocks."""
    rank_rtol = float(rank_rtol)
    if not 0.0 <= rank_rtol < 1.0:
        raise ValueError('rank_rtol must be in [0, 1)')

    diagnostics = ld_layout_diagnostics(block_sizes, bucket_size)
    ranks = []
    rank_fractions = []
    minimum_eigenvalues = []
    condition_estimates = []

    for block_index, (ld, size) in enumerate(zip(ld_blocks, block_sizes)):
        size = int(size)
        if not size:
            continue
        if np.shape(ld) != (size, size):
            raise ValueError(
                'LD block %d has shape %s; expected (%d, %d)' %
                (block_index, np.shape(ld), size, size)
            )
        symmetric_ld = (
            np.asarray(ld, dtype=np.float64) +
            np.asarray(ld, dtype=np.float64).T
        ) * 0.5
        eigenvalues = np.linalg.eigvalsh(symmetric_ld)
        maximum = max(float(eigenvalues[-1]), 0.0)
        cutoff = rank_rtol * maximum
        rank = int(np.count_nonzero(eigenvalues > cutoff))
        ranks.append(rank)
        rank_fractions.append(rank / float(size))
        minimum_eigenvalues.append(float(eigenvalues[0]))

        positive = eigenvalues[eigenvalues > cutoff]
        if positive.size:
            condition_estimates.append(float(maximum / positive[0]))

    diagnostics.update({
        'rank_rtol': rank_rtol,
        'rank_min': min(ranks, default=0),
        'rank_median': float(np.median(ranks)) if ranks else 0.0,
        'rank_max': max(ranks, default=0),
        'rank_fraction_median': (
            float(np.median(rank_fractions)) if rank_fractions else 0.0
        ),
        'minimum_eigenvalue': min(minimum_eigenvalues, default=0.0),
        'condition_median': (
            float(np.median(condition_estimates))
            if condition_estimates else float('inf')
        ),
    })
    return diagnostics


def format_ld_diagnostics(diagnostics):
    """Format diagnostics as stable, grep-friendly profile lines."""
    lines = [
        '[LD] %d active blocks, %d variants; size min/median/p90/max '
        '%d/%.1f/%.1f/%d' %
        (
            diagnostics['active_blocks'], diagnostics['variants'],
            diagnostics['size_min'], diagnostics['size_median'],
            diagnostics['size_p90'], diagnostics['size_max'],
        ),
        '[LD] padding memory %.3fx; estimated cubic work %.3fx' %
        (
            diagnostics['padding_memory_ratio'],
            diagnostics['padding_cubic_ratio'],
        ),
    ]
    if 'rank_rtol' in diagnostics:
        lines.extend([
            '[LD] numerical rank at rtol %.1e min/median/max %d/%.1f/%d; '
            'median fraction %.3f' %
            (
                diagnostics['rank_rtol'], diagnostics['rank_min'],
                diagnostics['rank_median'], diagnostics['rank_max'],
                diagnostics['rank_fraction_median'],
            ),
            '[LD] minimum eigenvalue %.3e; median retained condition %.3e' %
            (
                diagnostics['minimum_eigenvalue'],
                diagnostics['condition_median'],
            ),
        ])
    return '\n'.join(lines)


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


class CudaBetaBackend:
    """LD-resident, exact-FP64 batched CuPy beta block sampler."""

    name = 'cuda'

    def __init__(self, ld_blocks, block_sizes, beta_mrg, n_gwas,
                 seed=None, cuda_device=0, cuda_bucket_size=32):
        try:
            import cupy as cp
            from cupyx.scipy.linalg import solve_triangular
        except ImportError as exc:
            raise RuntimeError(
                'The CUDA backend requires CuPy 14.1 or newer. Install the '
                'package matching the host CUDA runtime (for example, '
                'cupy-cuda12x).'
            ) from exc

        version = tuple(
            int(part) for part in cp.__version__.split('.')[:2]
            if part.isdigit()
        )
        if version and version < (14, 1):
            raise RuntimeError(
                'The CUDA backend requires CuPy 14.1 or newer for batched '
                'triangular solves; found CuPy %s.' % cp.__version__
            )

        bucket_size = int(cuda_bucket_size)
        if bucket_size < 1:
            raise ValueError('cuda_bucket_size must be at least 1')

        device_id = int(cuda_device)
        if device_id < 0:
            raise ValueError('cuda_device must be non-negative')
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise RuntimeError(
                'CuPy is installed, but the CUDA runtime is unavailable'
            ) from exc
        if device_count < 1:
            raise RuntimeError(
                'CuPy is installed, but no CUDA device is visible'
            )
        if device_id >= device_count:
            raise ValueError(
                'cuda_device %d was requested, but only %d CUDA device(s) '
                'are visible' % (device_id, device_count)
            )

        self._cp = cp
        self._solve_triangular = solve_triangular
        self._device = cp.cuda.Device(device_id)
        self._n_gwas = int(n_gwas)
        self._bucket_size = bucket_size

        beta_mrg = np.asarray(beta_mrg, dtype=np.float64).reshape(-1, 1)
        self._p = beta_mrg.shape[0]
        layout = _block_layout(ld_blocks, block_sizes, self._p)
        block_groups = _fixed_block_groups(layout, self._bucket_size)

        self._groups = []
        self._resident_bytes = 0
        with self._device:
            self._rng = cp.random.RandomState(seed)
            self._beta_result = cp.empty(self._p + 1, dtype=cp.float64)
            self._psi_device = cp.empty(self._p, dtype=cp.float64)
            self._resident_bytes += (
                int(self._beta_result.nbytes) + int(self._psi_device.nbytes)
            )

            for padded_size, blocks in block_groups:
                count = len(blocks)
                host_ld = np.zeros(
                    (count, padded_size, padded_size), dtype=np.float64
                )
                host_beta_mrg = np.zeros(
                    (count, padded_size, 1), dtype=np.float64
                )
                host_indices = np.zeros(
                    (count, padded_size), dtype=np.int64
                )
                host_valid = np.zeros(
                    (count, padded_size), dtype=np.bool_
                )

                for row, (block_index, block_slice) in enumerate(blocks):
                    size = block_slice.stop - block_slice.start
                    host_ld[row, :size, :size] = ld_blocks[block_index]
                    if size < padded_size:
                        padding = np.arange(size, padded_size)
                        host_ld[row, padding, padding] = 1.0
                    host_beta_mrg[row, :size, 0] = beta_mrg[block_slice, 0]
                    host_indices[row, :size] = np.arange(
                        block_slice.start, block_slice.stop
                    )
                    host_valid[row, :size] = True

                group = {
                    'ld': cp.asarray(host_ld, blocking=True),
                    'beta_mrg': cp.asarray(host_beta_mrg, blocking=True),
                    'indices': cp.asarray(host_indices, blocking=True),
                    'valid': cp.asarray(host_valid, blocking=True),
                    'diag': cp.arange(padded_size),
                }
                self._resident_bytes += sum(
                    int(value.nbytes) for value in group.values()
                )
                self._groups.append(group)

    def sample(self, psi, sigma):
        """Draw beta on CUDA, copying only O(p) state per iteration."""
        cp = self._cp
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._p:
            raise ValueError('psi and beta_mrg must have equal length')

        with self._device:
            self._psi_device.set(psi)
            quad = cp.zeros((), dtype=cp.float64)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            noise = self._rng.standard_normal(self._p, dtype=cp.float64)

            for group in self._groups:
                precision = group['ld'].copy()
                safe_indices = group['indices']
                inverse_psi = cp.where(
                    group['valid'],
                    1.0 / self._psi_device[safe_indices],
                    0.0,
                )
                diagonal = group['diag']
                precision[:, diagonal, diagonal] += inverse_psi

                # CuPy returns L with precision = L @ L.T. Both calls
                # operate on the complete batch, including padded entries.
                chol = cp.linalg.cholesky(precision)
                beta_tmp = self._solve_triangular(
                    chol,
                    group['beta_mrg'],
                    trans='N',
                    lower=True,
                    check_finite=False,
                )
                beta_tmp += (
                    sd * noise[group['indices']][..., None] *
                    group['valid'][..., None]
                )
                beta_batch = self._solve_triangular(
                    chol,
                    beta_tmp,
                    trans='T',
                    lower=True,
                    check_finite=False,
                )

                flat_valid = group['valid'].ravel()
                self._beta_result[
                    group['indices'].ravel()[flat_valid]
                ] = beta_batch[..., 0].ravel()[flat_valid]
                # ||L.T @ beta||^2 is beta.T @ precision @ beta.
                quad += cp.sum(beta_tmp * beta_tmp)

            # One device-to-host transfer and synchronization per iteration.
            self._beta_result[self._p] = quad
            result = cp.asnumpy(self._beta_result)

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        return (
            'cuda:%d (CuPy batched Cholesky; %d size buckets; %.1f MiB '
            'static resident)' %
            (self._device.id, len(self._groups),
             self._resident_bytes / (1024.0 * 1024.0))
        )


def make_beta_backend(backend, ld_blocks, block_sizes, beta_mrg, n_gwas,
                      seed=None, cuda_device=0, cuda_bucket_size=32):
    """Construct a beta sampler without importing CUDA on CPU runs."""
    backend = str(backend).lower()
    if backend == 'cpu':
        return CpuBetaBackend(ld_blocks, block_sizes, beta_mrg, n_gwas)
    if backend == 'cuda':
        return CudaBetaBackend(
            ld_blocks, block_sizes, beta_mrg, n_gwas,
            seed=seed, cuda_device=cuda_device,
            cuda_bucket_size=cuda_bucket_size,
        )
    raise ValueError(
        "unknown beta backend %r; expected 'cpu' or 'cuda'" % backend
    )
