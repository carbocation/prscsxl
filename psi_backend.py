#!/usr/bin/env python3

"""Implementations of the PRS-CS local shrinkage update."""


import numpy as np

import gigrnd


class CpuPsiBackend:
    """Fused Numba implementation of the per-variant GIG update."""

    name = 'cpu'

    def __init__(self, seed=None):
        if seed is not None:
            gigrnd.seed_rng(seed)

    def sample(self, out, a_minus_half, delta, beta, sigma, n):
        gigrnd.gig_rvs_vec(
            out,
            float(a_minus_half),
            delta,
            beta,
            float(sigma),
            int(n),
        )

    def describe(self):
        return 'cpu (Numba fused GIG)'


class CudaPsiBackend:
    """Vectorized Devroye GIG sampler using CuPy's device RNG."""

    name = 'cuda'

    def __init__(self, size, seed=None, cuda_device=0,
                 cuda_gig_max_rounds=1000):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                'The CUDA psi backend requires CuPy. Install the package '
                'matching the host CUDA runtime (for example, '
                'cupy-cuda12x).'
            ) from exc

        size = int(size)
        if size < 1:
            raise ValueError('psi size must be positive')
        max_rounds = int(cuda_gig_max_rounds)
        if max_rounds < 1:
            raise ValueError('cuda_gig_max_rounds must be at least 1')
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
        self._device = cp.cuda.Device(device_id)
        self._size = size
        self._max_rounds = max_rounds
        self._sample_calls = 0
        self._rounds_total = 0
        self._rounds_maximum = 0
        with self._device:
            self._rng = cp.random.RandomState(seed)
            self._delta = cp.empty(size, dtype=cp.float64)
            self._beta = cp.empty(size, dtype=cp.float64)

    @staticmethod
    def _psi(cp, value, alpha, lam):
        return (
            -alpha * (cp.cosh(value) - 1.0) -
            lam * (cp.exp(value) - value - 1.0)
        )

    @staticmethod
    def _dpsi(cp, value, alpha, lam):
        return (
            -alpha * cp.sinh(value) -
            lam * (cp.exp(value) - 1.0)
        )

    def sample(self, out, a_minus_half, delta, beta, sigma, n):
        """Fill a host psi vector with independent CUDA GIG draws."""
        cp = self._cp
        out = np.asarray(out)
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        if out.size != self._size:
            raise ValueError(
                'out has %d values; expected %d' %
                (out.size, self._size)
            )
        if delta.size != self._size or beta.size != self._size:
            raise ValueError('delta, beta and out must have equal length')
        sigma = float(sigma)
        n = int(n)
        lam = abs(float(a_minus_half))
        swap = float(a_minus_half) < 0.0

        with self._device:
            self._delta.set(delta)
            self._beta.set(beta)
            gig_a = 2.0 * self._delta
            gig_b = n * self._beta * self._beta / sigma
            omega = cp.sqrt(gig_a * gig_b)
            alpha = cp.sqrt(omega * omega + lam * lam) - lam
            tiny = cp.finfo(cp.float64).tiny

            x_t = -self._psi(cp, 1.0, alpha, lam)
            zero_case = (alpha == 0.0) & (lam == 0.0)
            t_large = cp.where(
                zero_case,
                1.0,
                cp.sqrt(2.0 / cp.maximum(alpha + lam, tiny)),
            )
            t_small = cp.where(
                zero_case,
                1.0,
                cp.log(4.0 / cp.maximum(alpha + 2.0 * lam, tiny)),
            )
            t = cp.where(
                x_t > 2.0,
                t_large,
                cp.where(x_t < 0.5, t_small, 1.0),
            )

            x_s = -self._psi(cp, -1.0, alpha, lam)
            s_large = cp.where(
                zero_case,
                1.0,
                cp.sqrt(
                    4.0 / cp.maximum(
                        alpha * np.cosh(1.0) + lam, tiny
                    )
                ),
            )
            safe_alpha = cp.maximum(alpha, tiny)
            s_alpha = cp.log(
                1.0 + 1.0 / safe_alpha +
                cp.sqrt(
                    1.0 / (safe_alpha * safe_alpha) + 2.0 / safe_alpha
                )
            )
            if lam == 0.0:
                s_small = cp.where(zero_case, 1.0, s_alpha)
            else:
                s_small = cp.where(
                    zero_case,
                    1.0,
                    cp.where(
                        alpha == 0.0, 1.0 / lam,
                        cp.minimum(1.0 / lam, s_alpha),
                    ),
                )
            s = cp.where(
                x_s > 2.0,
                s_large,
                cp.where(x_s < 0.5, s_small, 1.0),
            )

            eta = -self._psi(cp, t, alpha, lam)
            zeta = -self._dpsi(cp, t, alpha, lam)
            theta = -self._psi(cp, -s, alpha, lam)
            xi = self._dpsi(cp, -s, alpha, lam)
            p_aux = 1.0 / cp.maximum(xi, tiny)
            r_aux = 1.0 / cp.maximum(zeta, tiny)
            td = t - r_aux * eta
            sd = s - p_aux * theta
            q = td + sd
            total = p_aux + q + r_aux
            root = cp.sqrt(lam * lam + omega * omega)

            active = cp.ones(self._size, dtype=cp.bool_)
            draws = cp.empty(self._size, dtype=cp.float64)
            rounds = 0
            while rounds < self._max_rounds:
                rounds += 1
                uniform_u = self._rng.random_sample(
                    self._size, dtype=cp.float64
                )
                uniform_v = cp.maximum(
                    self._rng.random_sample(
                        self._size, dtype=cp.float64
                    ),
                    tiny,
                )
                uniform_w = self._rng.random_sample(
                    self._size, dtype=cp.float64
                )

                first = uniform_u < q / total
                second = uniform_u < (q + r_aux) / total
                candidate = cp.where(
                    first,
                    -sd + q * uniform_v,
                    cp.where(
                        second,
                        td - r_aux * cp.log(uniform_v),
                        -sd + p_aux * cp.log(uniform_v),
                    ),
                )
                f1 = cp.exp(-eta - zeta * (candidate - t))
                f2 = cp.exp(-theta + xi * (candidate + s))
                envelope = cp.where(
                    candidate < -sd,
                    f2,
                    cp.where(candidate > td, f1, 1.0),
                )
                accepted = active & (
                    uniform_w * envelope <=
                    cp.exp(self._psi(cp, candidate, alpha, lam))
                )

                if swap:
                    transformed = (
                        cp.exp(-candidate) * gig_b /
                        cp.maximum(root + lam, tiny)
                    )
                else:
                    transformed = (
                        cp.exp(candidate) * (root + lam) / gig_a
                    )
                draws[accepted] = transformed[accepted]
                active &= ~accepted
                if not bool(cp.any(active).item()):
                    break

            if bool(cp.any(active).item()):
                remaining = int(cp.count_nonzero(active).item())
                raise RuntimeError(
                    'CUDA GIG rejection sampler exceeded %d rounds with '
                    '%d active draw(s)' %
                    (self._max_rounds, remaining)
                )
            out[:] = cp.asnumpy(draws)

        self._sample_calls += 1
        self._rounds_total += rounds
        self._rounds_maximum = max(self._rounds_maximum, rounds)

    def describe(self):
        return 'cuda:%d (CuPy vectorized GIG)' % self._device.id

    def profile_summary(self):
        if not self._sample_calls:
            return 'CUDA GIG: no draws recorded'
        return (
            'CUDA GIG: mean %.2f rejection rounds/vector, max %d' %
            (self._rounds_total / float(self._sample_calls),
             self._rounds_maximum)
        )


def make_psi_backend(backend, size, seed=None, cuda_device=0,
                     cuda_gig_max_rounds=1000):
    """Construct the requested local-shrinkage sampler."""
    backend = str(backend).lower()
    if backend == 'cpu':
        return CpuPsiBackend(seed=seed)
    if backend == 'cuda':
        return CudaPsiBackend(
            size, seed=seed, cuda_device=cuda_device,
            cuda_gig_max_rounds=cuda_gig_max_rounds,
        )
    raise ValueError(
        "unknown psi backend %r; expected 'cpu' or 'cuda'" % backend
    )
