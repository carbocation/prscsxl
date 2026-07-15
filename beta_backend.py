#!/usr/bin/env python3

"""Implementations of the PRS-CS beta block update."""


import time
import weakref

import numpy as np
from scipy.linalg.blas import dtrsv
from scipy.linalg.lapack import dpotrf


def _destroy_stream_handles(cublas, cusolver, cublas_handles,
                            cusolver_handles):
    """Best-effort release of CUDA library handles owned by one backend."""
    for handle in cublas_handles:
        try:
            cublas.destroy(handle)
        except Exception:
            pass
    for handle in cusolver_handles:
        try:
            cusolver.destroy(handle)
        except Exception:
            pass


_CUDA_STREAM_KERNEL_SOURCE = r"""
extern "C" __global__
void assemble_precision(
        const double* ld,
        const long long* indices,
        const unsigned char* valid,
        const double* psi,
        double* precision,
        const unsigned long long entries,
        const int size) {
    const unsigned long long offset =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= entries) return;

    const int column = (int)(offset % (unsigned long long)size);
    const int row = (int)(
        (offset / (unsigned long long)size) % (unsigned long long)size
    );
    double value = ld[offset];
    if (row == column) {
        const unsigned long long matrix = offset /
            ((unsigned long long)size * (unsigned long long)size);
        const unsigned long long vector_offset =
            matrix * (unsigned long long)size + (unsigned long long)row;
        if (valid[vector_offset]) {
            value += 1.0 / psi[indices[vector_offset]];
        }
    }
    precision[offset] = value;
}

extern "C" __global__
void perturb_quad(
        double* rhs,
        const long long* indices,
        const unsigned char* valid,
        const double* noise,
        const double sd,
        double* quad,
        const int size) {
    extern __shared__ double partial[];
    const int matrix = blockIdx.x;
    const unsigned long long base =
        (unsigned long long)matrix * (unsigned long long)size;
    double sum = 0.0;

    for (int row = threadIdx.x; row < size; row += blockDim.x) {
        const unsigned long long offset = base + (unsigned long long)row;
        double value = rhs[offset];
        if (valid[offset]) {
            value += sd * noise[indices[offset]];
        }
        rhs[offset] = value;
        sum += value * value;
    }

    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        quad[matrix] = partial[0];
    }
}

extern "C" __global__
void scatter_beta(
        const double* rhs,
        const long long* indices,
        const unsigned char* valid,
        double* beta,
        const unsigned long long entries) {
    const unsigned long long offset =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < entries && valid[offset]) {
        beta[indices[offset]] = rhs[offset];
    }
}
"""


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


class CudaDirectBetaBackend(CudaBetaBackend):
    """Preallocated direct cuSOLVER/cuBLAS FP64 batched implementation."""

    name = 'cuda'

    def __init__(self, *args, profile='FALSE', **kwargs):
        self._direct_profile_enabled = str(profile).upper() == 'TRUE'
        super().__init__(*args, **kwargs)
        try:
            from cupy.cuda import cublas, device
            from cupy_backends.cuda.libs import cusolver
        except ImportError as exc:
            raise RuntimeError(
                "The CUDA beta backend requires CuPy's cuBLAS and "
                'cuSOLVER bindings'
            ) from exc

        cp = self._cp
        self._cublas = cublas
        self._cusolver = cusolver
        self._host_dtype = np.dtype(np.float64)
        self._cuda_dtype = cp.float64
        self._potrf_batched = self._cusolver.dpotrfBatched
        self._one = np.array(1.0, dtype=np.float64)
        self._potrf_checked = False
        self._direct_profile_calls = 0
        self._direct_profile_host_total = 0.0
        self._direct_profile_stage_totals = np.zeros(6)

        def matrix_pointers(array):
            count = array.shape[0]
            step = int(array[0].nbytes)
            start = int(array.data.ptr)
            return cp.arange(
                start,
                start + step * count,
                step,
                dtype=cp.uintp,
            )

        with self._device:
            self._cublas_handle = device.get_cublas_handle()
            self._cusolver_handle = device.get_cusolver_handle()
            if self._direct_profile_enabled:
                self._direct_profile_preamble = (
                    cp.cuda.Event(), cp.cuda.Event()
                )
            for group in self._groups:
                precision = group['ld'].copy()
                rhs = cp.empty_like(group['beta_mrg'])
                direct_arrays = {
                    'precision': precision,
                    'rhs': rhs,
                    'precision_ptrs': matrix_pointers(precision),
                    'rhs_ptrs': matrix_pointers(rhs),
                    'potrf_info': cp.empty(
                        precision.shape[0], dtype=cp.int32
                    ),
                }
                group.update(direct_arrays)
                if self._direct_profile_enabled:
                    group['direct_profile_events'] = [
                        cp.cuda.Event() for _ in range(6)
                    ]
                self._resident_bytes += sum(
                    int(value.nbytes) for value in direct_arrays.values()
                )

    def _triangular_solve(self, group, trans):
        self._triangular_solve_with_handle(
            group, trans, self._cublas_handle
        )

    def _triangular_solve_with_handle(self, group, trans, handle):
        size = group['precision'].shape[-1]
        count = group['precision'].shape[0]
        self._cublas.dtrsmBatched(
            handle,
            self._cublas.CUBLAS_SIDE_LEFT,
            self._cublas.CUBLAS_FILL_MODE_UPPER,
            trans,
            self._cublas.CUBLAS_DIAG_NON_UNIT,
            size,
            1,
            self._one.ctypes.data,
            group['precision_ptrs'].data.ptr,
            size,
            group['rhs_ptrs'].data.ptr,
            size,
            count,
        )

    def sample(self, psi, sigma):
        """Draw beta with fixed device workspaces and in-place CUDA calls."""
        cp = self._cp
        profile_started = (
            time.perf_counter() if self._direct_profile_enabled else None
        )
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._p:
            raise ValueError('psi and beta_mrg must have equal length')

        with self._device:
            if self._direct_profile_enabled:
                self._direct_profile_preamble[0].record()
            self._psi_device.set(psi)
            quad = cp.zeros((), dtype=cp.float64)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            noise = self._rng.standard_normal(self._p, dtype=cp.float64)
            if self._direct_profile_enabled:
                self._direct_profile_preamble[1].record()

            for group_index, group in enumerate(self._groups):
                events = group.get('direct_profile_events')
                if events is not None:
                    events[0].record()
                precision = group['precision']
                cp.copyto(precision, group['ld'])
                safe_indices = group['indices']
                inverse_psi = cp.where(
                    group['valid'],
                    1.0 / self._psi_device[safe_indices],
                    0.0,
                )
                diagonal = group['diag']
                precision[:, diagonal, diagonal] += inverse_psi
                if events is not None:
                    events[1].record()

                size = precision.shape[-1]
                count = precision.shape[0]
                self._cusolver.dpotrfBatched(
                    self._cusolver_handle,
                    self._cublas.CUBLAS_FILL_MODE_UPPER,
                    size,
                    group['precision_ptrs'].data.ptr,
                    size,
                    group['potrf_info'].data.ptr,
                    count,
                )
                if events is not None:
                    events[2].record()
                if not self._potrf_checked:
                    info = cp.asnumpy(group['potrf_info'])
                    failures = np.flatnonzero(info)
                    if failures.size:
                        first = int(failures[0])
                        raise RuntimeError(
                            'CUDA Cholesky failed in size bucket %d, '
                            'matrix %d with info=%d' %
                            (group_index, first, int(info[first]))
                        )

                rhs = group['rhs']
                cp.copyto(rhs, group['beta_mrg'])
                # Row-major L is seen by cuBLAS as column-major U=L.T.
                # U.T y=b therefore performs the first solve L y=b.
                self._triangular_solve(
                    group, self._cublas.CUBLAS_OP_T
                )
                if events is not None:
                    events[3].record()
                rhs += (
                    sd * noise[group['indices']][..., None] *
                    group['valid'][..., None]
                )
                quad += cp.sum(rhs * rhs)
                if events is not None:
                    events[4].record()

                # U beta=y is the second solve L.T beta=y.
                self._triangular_solve(
                    group, self._cublas.CUBLAS_OP_N
                )
                flat_valid = group['valid'].ravel()
                self._beta_result[
                    group['indices'].ravel()[flat_valid]
                ] = rhs[..., 0].ravel()[flat_valid]
                if events is not None:
                    events[5].record()

            self._potrf_checked = True
            self._beta_result[self._p] = quad
            result = cp.asnumpy(self._beta_result)

            if self._direct_profile_enabled:
                stages = np.zeros(6)
                stages[0] = cp.cuda.get_elapsed_time(
                    *self._direct_profile_preamble
                ) / 1000.0
                for group in self._groups:
                    events = group['direct_profile_events']
                    for stage in range(1, 6):
                        stages[stage] += cp.cuda.get_elapsed_time(
                            events[stage - 1], events[stage]
                        ) / 1000.0
                if self._direct_profile_calls:
                    self._direct_profile_stage_totals += stages
                    self._direct_profile_host_total += (
                        time.perf_counter() - profile_started
                    )
                self._direct_profile_calls += 1

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        return (
            'cuda:%d (preallocated FP64 potrfBatched + '
            'trsmBatched; %d size buckets; %.1f MiB static resident)' %
            (
                self._device.id,
                len(self._groups),
                self._resident_bytes / (1024.0 * 1024.0),
            )
        )

    def profile_summary(self):
        measured = self._direct_profile_calls - 1
        if not self._direct_profile_enabled:
            return 'CUDA stages: profiling disabled'
        if measured < 1:
            return 'CUDA stages: no steady-state draws recorded'
        means = 1000.0 * self._direct_profile_stage_totals / measured
        host_mean = 1000.0 * self._direct_profile_host_total / measured
        unattributed = max(host_mean - float(means.sum()), 0.0)
        return (
            'CUDA stages (mean): preamble %.3f ms, '
            'precision %.3f ms, potrf %.3f ms, solve-1 %.3f ms, '
            'perturb/quad %.3f ms, solve-2/scatter %.3f ms, '
            'host/unattributed %.3f ms' %
            (*means, unattributed)
        )


class CudaHybridBetaBackend(CudaDirectBetaBackend):
    """Exact FP64 backend selecting dense routines by bucket occupancy."""

    name = 'cuda'
    _minimum_batched_count = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._potrf_single = self._cusolver.dpotrf
        self._potrf_buffer_size = self._cusolver.dpotrf_bufferSize
        self._trsm_single = self._cublas.dtrsm

        cp = self._cp
        maximum_lwork = 0
        self._unbatched_groups = 0
        self._unbatched_matrices = 0
        self._batched_groups = 0
        self._batched_matrices = 0
        with self._device:
            for group in self._groups:
                count = int(group['precision'].shape[0])
                use_unbatched = count < self._minimum_batched_count
                group['use_unbatched'] = use_unbatched
                if use_unbatched:
                    size = int(group['precision'].shape[-1])
                    lwork = int(self._potrf_buffer_size(
                        self._cusolver_handle,
                        self._cublas.CUBLAS_FILL_MODE_UPPER,
                        size,
                        group['precision'][0].data.ptr,
                        size,
                    ))
                    group['single_lwork'] = lwork
                    maximum_lwork = max(maximum_lwork, lwork)
                    self._unbatched_groups += 1
                    self._unbatched_matrices += count
                else:
                    self._batched_groups += 1
                    self._batched_matrices += count

            self._single_workspace = cp.empty(
                maximum_lwork, dtype=self._cuda_dtype
            )
            self._resident_bytes += int(self._single_workspace.nbytes)
            if self._direct_profile_enabled:
                self._hybrid_profile_events = [
                    cp.cuda.Event() for _ in range(6)
                ]

    def _factor_group(self, group, cusolver_handle=None, workspace=None):
        if cusolver_handle is None:
            cusolver_handle = self._cusolver_handle
        if workspace is None:
            workspace = self._single_workspace
        size = int(group['precision'].shape[-1])
        count = int(group['precision'].shape[0])
        if not group['use_unbatched']:
            self._potrf_batched(
                cusolver_handle,
                self._cublas.CUBLAS_FILL_MODE_UPPER,
                size,
                group['precision_ptrs'].data.ptr,
                size,
                group['potrf_info'].data.ptr,
                count,
            )
            return

        for matrix_index in range(count):
            self._factor_matrix(
                group, matrix_index, cusolver_handle, workspace
            )

    def _factor_matrix(self, group, matrix_index, cusolver_handle,
                       workspace):
        size = int(group['precision'].shape[-1])
        self._potrf_single(
            cusolver_handle,
            self._cublas.CUBLAS_FILL_MODE_UPPER,
            size,
            group['precision'][matrix_index].data.ptr,
            size,
            workspace.data.ptr,
            group['single_lwork'],
            group['potrf_info'][
                matrix_index:matrix_index + 1
            ].data.ptr,
        )

    def _solve_group(self, group, trans, cublas_handle=None):
        if cublas_handle is None:
            cublas_handle = self._cublas_handle
        if not group['use_unbatched']:
            self._triangular_solve_with_handle(
                group, trans, cublas_handle
            )
            return

        count = int(group['precision'].shape[0])
        for matrix_index in range(count):
            self._solve_matrix(
                group, matrix_index, trans, cublas_handle
            )

    def _solve_matrix(self, group, matrix_index, trans, cublas_handle):
        size = int(group['precision'].shape[-1])
        self._trsm_single(
            cublas_handle,
            self._cublas.CUBLAS_SIDE_LEFT,
            self._cublas.CUBLAS_FILL_MODE_UPPER,
            trans,
            self._cublas.CUBLAS_DIAG_NON_UNIT,
            size,
            1,
            self._one.ctypes.data,
            group['precision'][matrix_index].data.ptr,
            size,
            group['rhs'][matrix_index].data.ptr,
            size,
        )

    def sample(self, psi, sigma):
        """Draw beta with regular dense routines for sparse size buckets."""
        cp = self._cp
        profile_started = (
            time.perf_counter() if self._direct_profile_enabled else None
        )
        psi = np.asarray(psi, dtype=self._host_dtype).reshape(-1)
        if psi.size != self._p:
            raise ValueError('psi and beta_mrg must have equal length')

        with self._device:
            if self._direct_profile_enabled:
                self._direct_profile_preamble[0].record()
            self._psi_device.set(psi)
            quad = cp.zeros((), dtype=self._cuda_dtype)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            noise = self._rng.standard_normal(
                self._p, dtype=self._cuda_dtype
            )
            if self._direct_profile_enabled:
                self._direct_profile_preamble[1].record()
                self._hybrid_profile_events[0].record()

            for group in self._groups:
                precision = group['precision']
                cp.copyto(precision, group['ld'])
                safe_indices = group['indices']
                inverse_psi = cp.where(
                    group['valid'],
                    1.0 / self._psi_device[safe_indices],
                    0.0,
                )
                diagonal = group['diag']
                precision[:, diagonal, diagonal] += inverse_psi
            if self._direct_profile_enabled:
                self._hybrid_profile_events[1].record()

            for group in self._groups:
                self._factor_group(group)
            if self._direct_profile_enabled:
                self._hybrid_profile_events[2].record()

            if not self._potrf_checked:
                for group_index, group in enumerate(self._groups):
                    info = cp.asnumpy(group['potrf_info'])
                    failures = np.flatnonzero(info)
                    if failures.size:
                        first = int(failures[0])
                        raise RuntimeError(
                            'CUDA Cholesky failed in size bucket %d, '
                            'matrix %d with info=%d' %
                            (group_index, first, int(info[first]))
                        )

            for group in self._groups:
                cp.copyto(group['rhs'], group['beta_mrg'])
                self._solve_group(group, self._cublas.CUBLAS_OP_T)
            if self._direct_profile_enabled:
                self._hybrid_profile_events[3].record()

            for group in self._groups:
                rhs = group['rhs']
                rhs += (
                    sd * noise[group['indices']][..., None] *
                    group['valid'][..., None]
                )
                quad += cp.sum(rhs * rhs)
            if self._direct_profile_enabled:
                self._hybrid_profile_events[4].record()

            for group in self._groups:
                self._solve_group(group, self._cublas.CUBLAS_OP_N)
                flat_valid = group['valid'].ravel()
                self._beta_result[
                    group['indices'].ravel()[flat_valid]
                ] = group['rhs'][..., 0].ravel()[flat_valid]
            if self._direct_profile_enabled:
                self._hybrid_profile_events[5].record()

            self._potrf_checked = True
            self._beta_result[self._p] = quad
            result = cp.asnumpy(self._beta_result)

            if self._direct_profile_enabled:
                stages = np.zeros(6)
                stages[0] = cp.cuda.get_elapsed_time(
                    *self._direct_profile_preamble
                ) / 1000.0
                for stage in range(1, 6):
                    stages[stage] = cp.cuda.get_elapsed_time(
                        self._hybrid_profile_events[stage - 1],
                        self._hybrid_profile_events[stage],
                    ) / 1000.0
                if self._direct_profile_calls:
                    self._direct_profile_stage_totals += stages
                    self._direct_profile_host_total += (
                        time.perf_counter() - profile_started
                    )
                self._direct_profile_calls += 1

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        return (
            'cuda:%d (exact FP64; regular potrf/trsm for %d matrices in '
            '%d sparse buckets, batched for %d matrices in %d buckets; '
            '%.1f MiB static resident)' %
            (
                self._device.id,
                self._unbatched_matrices,
                self._unbatched_groups,
                self._batched_matrices,
                self._batched_groups,
                self._resident_bytes / (1024.0 * 1024.0),
            )
        )


class CudaStreamsBetaBackend(CudaHybridBetaBackend):
    """Exact FP64 backend overlapping independent buckets on CUDA streams."""

    name = 'cuda'

    def __init__(self, *args, **kwargs):
        requested_streams = int(kwargs.pop('cuda_streams', 4))
        if requested_streams < 1:
            raise ValueError('cuda_streams must be at least 1')
        super().__init__(*args, **kwargs)
        if not self._groups:
            raise ValueError('CUDA requires at least one active LD block')

        cp = self._cp
        factor_tasks = []
        for group in self._groups:
            count = int(group['precision'].shape[0])
            if group['use_unbatched']:
                factor_tasks.extend(
                    (group, matrix_index)
                    for matrix_index in range(count)
                )
            else:
                factor_tasks.append((group, None))

        self._factor_task_count = len(factor_tasks)
        self._stream_count = min(requested_streams, self._factor_task_count)
        solve_tasks = [
            (group, matrix_index)
            for group in self._groups
            for matrix_index in range(int(group['precision'].shape[0]))
        ]
        self._solve_task_count = len(solve_tasks)
        self._auxiliary_stream_count = min(
            8, requested_streams, self._solve_task_count
        )
        self._worker_stream_count = max(
            self._stream_count, self._auxiliary_stream_count
        )

        def distribute(items, cost, lane_count):
            lanes = [[] for _ in range(lane_count)]
            lane_work = [0] * lane_count
            for item in sorted(items, key=cost, reverse=True):
                lane = min(
                    range(lane_count), key=lane_work.__getitem__
                )
                lanes[lane].append(item)
                lane_work[lane] += cost(item)
            return lanes

        def factor_cost(task):
            group, matrix_index = task
            size = int(group['precision'].shape[-1])
            count = (
                int(group['precision'].shape[0])
                if matrix_index is None else 1
            )
            return count * size ** 3

        self._lane_factor_tasks = distribute(
            factor_tasks, factor_cost, self._stream_count
        )
        self._lane_solve_tasks = distribute(
            solve_tasks,
            lambda task: int(task[0]['precision'].shape[-1]) ** 2,
            self._auxiliary_stream_count,
        )
        self._lane_groups = distribute(
            self._groups,
            lambda group: (
                int(group['precision'].shape[0]) *
                int(group['precision'].shape[-1]) ** 2
            ),
            min(self._auxiliary_stream_count, len(self._groups)),
        )

        self._lane_cublas_handles = []
        self._lane_cusolver_handles = []
        try:
            with self._device:
                self._stream_kernel_module = cp.RawModule(
                    code=_CUDA_STREAM_KERNEL_SOURCE,
                    options=('--std=c++11',),
                )
                self._assemble_precision_kernel = (
                    self._stream_kernel_module.get_function(
                        'assemble_precision'
                    )
                )
                self._scatter_beta_kernel = (
                    self._stream_kernel_module.get_function('scatter_beta')
                )
                self._perturb_quad_kernel = (
                    self._stream_kernel_module.get_function('perturb_quad')
                )
                self._kernel_threads = 256
                self._streams = [
                    cp.cuda.Stream(non_blocking=True)
                    for _ in range(self._worker_stream_count)
                ]
                for stream in self._streams:
                    cublas_handle = self._cublas.create()
                    self._lane_cublas_handles.append(cublas_handle)
                    cusolver_handle = self._cusolver.create()
                    self._lane_cusolver_handles.append(cusolver_handle)
                    self._cublas.setStream(cublas_handle, stream.ptr)
                    self._cusolver.setStream(cusolver_handle, stream.ptr)

                self._lane_workspaces = []
                for tasks in self._lane_factor_tasks:
                    maximum_lwork = max(
                        (
                            int(group.get('single_lwork', 0))
                            for group, matrix_index in tasks
                            if matrix_index is not None
                        ),
                        default=0,
                    )
                    workspace = cp.empty(
                        maximum_lwork, dtype=self._cuda_dtype
                    )
                    self._lane_workspaces.append(workspace)
                    self._resident_bytes += int(workspace.nbytes)

                matrix_count = sum(
                    int(group['precision'].shape[0])
                    for group in self._groups
                )
                self._quad_matrices = cp.empty(
                    matrix_count, dtype=self._cuda_dtype
                )
                matrix_start = 0
                for group in self._groups:
                    count = int(group['precision'].shape[0])
                    group['quad_results'] = self._quad_matrices[
                        matrix_start:matrix_start + count
                    ]
                    matrix_start += count
                self._resident_bytes += int(self._quad_matrices.nbytes)

                self._stream_boundaries = [
                    cp.cuda.Event() for _ in range(6)
                ]
                self._lane_completion = [
                    [cp.cuda.Event() for _ in range(
                        self._worker_stream_count
                    )]
                    for _ in range(5)
                ]
        except Exception:
            _destroy_stream_handles(
                self._cublas,
                self._cusolver,
                self._lane_cublas_handles,
                self._lane_cusolver_handles,
            )
            raise

        self._handle_finalizer = weakref.finalize(
            self,
            _destroy_stream_handles,
            self._cublas,
            self._cusolver,
            self._lane_cublas_handles,
            self._lane_cusolver_handles,
        )

    def _dispatch_stage(self, stage_index, lane_items, operation):
        default_stream = self._cp.cuda.get_current_stream()
        for lane, (stream, items) in enumerate(zip(
                self._streams, lane_items)):
            with stream:
                stream.wait_event(self._stream_boundaries[stage_index])
                for item in items:
                    operation(lane, item)
                stream.record(self._lane_completion[stage_index][lane])

        for event in self._lane_completion[stage_index][:len(lane_items)]:
            default_stream.wait_event(event)
        default_stream.record(self._stream_boundaries[stage_index + 1])

    def sample(self, psi, sigma):
        """Draw beta while overlapping independent block computations."""
        cp = self._cp
        profile_started = (
            time.perf_counter() if self._direct_profile_enabled else None
        )
        psi = np.asarray(psi, dtype=self._host_dtype).reshape(-1)
        if psi.size != self._p:
            raise ValueError('psi and beta_mrg must have equal length')

        with self._device:
            default_stream = cp.cuda.get_current_stream()
            if self._direct_profile_enabled:
                default_stream.record(self._direct_profile_preamble[0])
            self._psi_device.set(psi)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            noise = self._rng.standard_normal(
                self._p, dtype=self._cuda_dtype
            )
            if self._direct_profile_enabled:
                default_stream.record(self._direct_profile_preamble[1])
            default_stream.record(self._stream_boundaries[0])

            def assemble_precision(_lane, group):
                entries = int(group['precision'].size)
                blocks = (
                    (entries + self._kernel_threads - 1) //
                    self._kernel_threads
                )
                self._assemble_precision_kernel(
                    (blocks,),
                    (self._kernel_threads,),
                    (
                        group['ld'],
                        group['indices'],
                        group['valid'],
                        self._psi_device,
                        group['precision'],
                        np.uint64(entries),
                        np.int32(group['precision'].shape[-1]),
                    ),
                )

            self._dispatch_stage(
                0, self._lane_groups, assemble_precision
            )

            def factor(lane, task):
                group, matrix_index = task
                if matrix_index is None:
                    self._factor_group(
                        group,
                        self._lane_cusolver_handles[lane],
                        self._lane_workspaces[lane],
                    )
                else:
                    self._factor_matrix(
                        group,
                        matrix_index,
                        self._lane_cusolver_handles[lane],
                        self._lane_workspaces[lane],
                    )

            self._dispatch_stage(
                1, self._lane_factor_tasks, factor
            )

            if not self._potrf_checked:
                for group_index, group in enumerate(self._groups):
                    info = cp.asnumpy(group['potrf_info'])
                    failures = np.flatnonzero(info)
                    if failures.size:
                        first = int(failures[0])
                        raise RuntimeError(
                            'CUDA Cholesky failed in size bucket %d, '
                            'matrix %d with info=%d' %
                            (group_index, first, int(info[first]))
                        )

            def solve_first(lane, task):
                group, matrix_index = task
                cp.copyto(
                    group['rhs'][matrix_index],
                    group['beta_mrg'][matrix_index],
                )
                self._solve_matrix(
                    group,
                    matrix_index,
                    self._cublas.CUBLAS_OP_T,
                    self._lane_cublas_handles[lane],
                )

            self._dispatch_stage(
                2, self._lane_solve_tasks, solve_first
            )

            def perturb(_lane, group):
                self._perturb_quad_kernel(
                    (int(group['precision'].shape[0]),),
                    (self._kernel_threads,),
                    (
                        group['rhs'],
                        group['indices'],
                        group['valid'],
                        noise,
                        np.float64(sd),
                        group['quad_results'],
                        np.int32(group['precision'].shape[-1]),
                    ),
                    shared_mem=(
                        self._kernel_threads * self._host_dtype.itemsize
                    ),
                )

            self._dispatch_stage(3, self._lane_groups, perturb)

            def solve_second(lane, task):
                group, matrix_index = task
                self._solve_matrix(
                    group,
                    matrix_index,
                    self._cublas.CUBLAS_OP_N,
                    self._lane_cublas_handles[lane],
                )
                entries = int(group['indices'].shape[-1])
                blocks = (
                    (entries + self._kernel_threads - 1) //
                    self._kernel_threads
                )
                self._scatter_beta_kernel(
                    (blocks,),
                    (self._kernel_threads,),
                    (
                        group['rhs'][matrix_index],
                        group['indices'][matrix_index],
                        group['valid'][matrix_index],
                        self._beta_result,
                        np.uint64(entries),
                    ),
                )

            self._dispatch_stage(
                4, self._lane_solve_tasks, solve_second
            )

            self._potrf_checked = True
            self._beta_result[self._p] = cp.sum(self._quad_matrices)
            result = cp.asnumpy(self._beta_result)

            if self._direct_profile_enabled:
                stages = np.zeros(6)
                stages[0] = cp.cuda.get_elapsed_time(
                    *self._direct_profile_preamble
                ) / 1000.0
                for stage in range(1, 6):
                    stages[stage] = cp.cuda.get_elapsed_time(
                        self._stream_boundaries[stage - 1],
                        self._stream_boundaries[stage],
                    ) / 1000.0
                if self._direct_profile_calls:
                    self._direct_profile_stage_totals += stages
                    self._direct_profile_host_total += (
                        time.perf_counter() - profile_started
                    )
                self._direct_profile_calls += 1

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        stream_word = 'stream' if self._stream_count == 1 else 'streams'
        auxiliary_word = (
            'stream' if self._auxiliary_stream_count == 1 else 'streams'
        )
        task_word = 'task' if self._factor_task_count == 1 else 'tasks'
        return (
            'cuda:%d (exact FP64; %d factorization %s, %d auxiliary %s; '
            '%d independently scheduled factor %s and %d matrix solve '
            'tasks; fused assembly/perturb/scatter; %d regular and %d '
            'batched matrices; %.1f MiB static resident)' %
            (
                self._device.id,
                self._stream_count,
                stream_word,
                self._auxiliary_stream_count,
                auxiliary_word,
                self._factor_task_count,
                task_word,
                self._solve_task_count,
                self._unbatched_matrices,
                self._batched_matrices,
                self._resident_bytes / (1024.0 * 1024.0),
            )
        )


def make_beta_backend(backend, ld_blocks, block_sizes, beta_mrg, n_gwas,
                      seed=None, cuda_device=0, cuda_bucket_size=32,
                      cuda_streams=4, profile='FALSE'):
    """Construct a beta sampler without importing CUDA on CPU runs."""
    backend = str(backend).lower()
    if backend == 'cpu':
        return CpuBetaBackend(ld_blocks, block_sizes, beta_mrg, n_gwas)
    if backend == 'cuda':
        return CudaStreamsBetaBackend(
            ld_blocks, block_sizes, beta_mrg, n_gwas,
            seed=seed, cuda_device=cuda_device,
            cuda_bucket_size=cuda_bucket_size,
            cuda_streams=cuda_streams, profile=profile,
        )
    raise ValueError(
        "unknown beta backend %r; expected 'cpu' or 'cuda'" % backend
    )
