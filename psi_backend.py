#!/usr/bin/env python3

"""Implementations of the PRS-CS local shrinkage update."""


import numpy as np

import gigrnd


_CUDA_GIG_KERNEL = r"""
struct PhiloxState {
    unsigned int key0;
    unsigned int key1;
    unsigned long long call;
    unsigned long long block;
    unsigned int values[4];
    int position;
};

__device__ __forceinline__ unsigned long long splitmix64(
        unsigned long long value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ void philox_refill(PhiloxState* state) {
    const unsigned int multiply0 = 0xd2511f53U;
    const unsigned int multiply1 = 0xcd9e8d57U;
    unsigned int counter0 = (unsigned int) state->block;
    unsigned int counter1 = (unsigned int) (state->block >> 32);
    unsigned int counter2 = (unsigned int) state->call;
    unsigned int counter3 = (unsigned int) (state->call >> 32);
    unsigned int key0 = state->key0;
    unsigned int key1 = state->key1;

    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        unsigned int high0 = __umulhi(multiply0, counter0);
        unsigned int high1 = __umulhi(multiply1, counter2);
        unsigned int low0 = multiply0 * counter0;
        unsigned int low1 = multiply1 * counter2;
        unsigned int next0 = high1 ^ counter1 ^ key0;
        unsigned int next1 = low1;
        unsigned int next2 = high0 ^ counter3 ^ key1;
        unsigned int next3 = low0;
        counter0 = next0;
        counter1 = next1;
        counter2 = next2;
        counter3 = next3;
        key0 += 0x9e3779b9U;
        key1 += 0xbb67ae85U;
    }

    state->values[0] = counter0;
    state->values[1] = counter1;
    state->values[2] = counter2;
    state->values[3] = counter3;
    state->position = 0;
    state->block += 1ULL;
}

__device__ __forceinline__ unsigned int philox_next(PhiloxState* state) {
    if (state->position == 4) {
        philox_refill(state);
    }
    return state->values[state->position++];
}

__device__ __forceinline__ double uniform_open(PhiloxState* state) {
    unsigned long long high = (unsigned long long) (philox_next(state) >> 5);
    unsigned long long low = (unsigned long long) (philox_next(state) >> 6);
    unsigned long long bits = high * 67108864ULL + low;
    return ((double) bits + 0.5) * 1.1102230246251565e-16;
}

__device__ __forceinline__ double log_density(
        double value, double alpha, double lambda) {
    return -alpha * (cosh(value) - 1.0)
           -lambda * (exp(value) - value - 1.0);
}

__device__ __forceinline__ double density_derivative(
        double value, double alpha, double lambda) {
    return -alpha * sinh(value) - lambda * (exp(value) - 1.0);
}

extern "C" __global__ void gig_sample_kernel(
        const double* delta,
        const double* beta,
        double* draws,
        int* round_counts,
        long long size,
        double shape,
        double sigma,
        double n,
        unsigned long long seed,
        unsigned long long call,
        int max_rounds) {
    long long index = (long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    const double tiny = 2.2250738585072014e-308;
    const double lambda = fabs(shape);
    const bool swap = shape < 0.0;
    const double gig_a = 2.0 * delta[index];
    const double gig_b = n * beta[index] * beta[index] / sigma;
    const double omega = sqrt(gig_a * gig_b);
    const double root = sqrt(lambda * lambda + omega * omega);
    const double alpha = root - lambda;
    const bool zero_case = alpha == 0.0 && lambda == 0.0;

    const double x_t = -log_density(1.0, alpha, lambda);
    const double t_large = zero_case
        ? 1.0 : sqrt(2.0 / fmax(alpha + lambda, tiny));
    const double t_small = zero_case
        ? 1.0 : log(4.0 / fmax(alpha + 2.0 * lambda, tiny));
    const double t = x_t > 2.0
        ? t_large : (x_t < 0.5 ? t_small : 1.0);

    const double x_s = -log_density(-1.0, alpha, lambda);
    const double s_large = zero_case
        ? 1.0
        : sqrt(4.0 / fmax(alpha * 1.5430806348152437 + lambda, tiny));
    const double safe_alpha = fmax(alpha, tiny);
    const double inverse_alpha = 1.0 / safe_alpha;
    const double s_alpha = log(
        1.0 + inverse_alpha
        + sqrt(inverse_alpha * inverse_alpha + 2.0 * inverse_alpha)
    );
    double s_small;
    if (lambda == 0.0) {
        s_small = zero_case ? 1.0 : s_alpha;
    } else if (zero_case) {
        s_small = 1.0;
    } else if (alpha == 0.0) {
        s_small = 1.0 / lambda;
    } else {
        s_small = fmin(1.0 / lambda, s_alpha);
    }
    const double s = x_s > 2.0
        ? s_large : (x_s < 0.5 ? s_small : 1.0);

    const double eta = -log_density(t, alpha, lambda);
    const double zeta = -density_derivative(t, alpha, lambda);
    const double theta = -log_density(-s, alpha, lambda);
    const double xi = density_derivative(-s, alpha, lambda);
    const double p_aux = 1.0 / fmax(xi, tiny);
    const double r_aux = 1.0 / fmax(zeta, tiny);
    const double td = t - r_aux * eta;
    const double sd = s - p_aux * theta;
    const double q = td + sd;
    const double total = p_aux + q + r_aux;

    unsigned long long stream = splitmix64(
        seed + 0x9e3779b97f4a7c15ULL
        * ((unsigned long long) index + 1ULL)
    );
    PhiloxState rng;
    rng.key0 = (unsigned int) stream;
    rng.key1 = (unsigned int) (stream >> 32);
    rng.call = call;
    rng.block = 0ULL;
    rng.position = 4;

    for (int round = 1; round <= max_rounds; ++round) {
        const double uniform_u = uniform_open(&rng);
        const double uniform_v = uniform_open(&rng);
        const double uniform_w = uniform_open(&rng);
        double candidate;
        if (uniform_u < q / total) {
            candidate = -sd + q * uniform_v;
        } else if (uniform_u < (q + r_aux) / total) {
            candidate = td - r_aux * log(uniform_v);
        } else {
            candidate = -sd + p_aux * log(uniform_v);
        }

        const double f1 = exp(-eta - zeta * (candidate - t));
        const double f2 = exp(-theta + xi * (candidate + s));
        const double envelope = candidate < -sd
            ? f2 : (candidate > td ? f1 : 1.0);
        if (uniform_w * envelope
                <= exp(log_density(candidate, alpha, lambda))) {
            draws[index] = swap
                ? exp(-candidate) * gig_b / fmax(root + lambda, tiny)
                : exp(candidate) * (root + lambda) / gig_a;
            round_counts[index] = round;
            return;
        }
    }

    draws[index] = nan("");
    round_counts[index] = max_rounds + 1;
}
"""


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


class CudaRawPsiBackend:
    """One-thread-per-draw CUDA GIG sampler using a single raw kernel."""

    name = 'cuda'

    def __init__(self, size, seed=None, cuda_device=0,
                 cuda_gig_max_rounds=1000):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                'The CUDA psi backend requires CuPy. Install the '
                'package matching the host CUDA runtime (for example, '
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
        self._seed = np.random.SeedSequence(seed).generate_state(
            1, dtype=np.uint64
        )[0]
        self._call_index = 0
        self._sample_calls = 0
        self._rounds_total = 0
        self._rounds_maximum = 0
        with self._device:
            self._delta = cp.empty(size, dtype=cp.float64)
            self._beta = cp.empty(size, dtype=cp.float64)
            self._draws = cp.empty(size, dtype=cp.float64)
            self._round_counts = cp.empty(size, dtype=cp.int32)
            self._kernel = cp.RawKernel(
                _CUDA_GIG_KERNEL,
                'gig_sample_kernel',
                options=('-std=c++11',),
            )

    def sample(self, out, a_minus_half, delta, beta, sigma, n):
        """Fill a host psi vector with single-kernel CUDA GIG draws."""
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

        threads = 256
        blocks = (self._size + threads - 1) // threads
        with self._device:
            self._delta.set(delta)
            self._beta.set(beta)
            self._kernel(
                (blocks,),
                (threads,),
                (
                    self._delta,
                    self._beta,
                    self._draws,
                    self._round_counts,
                    np.int64(self._size),
                    np.float64(a_minus_half),
                    np.float64(sigma),
                    np.float64(n),
                    np.uint64(self._seed),
                    np.uint64(self._call_index),
                    np.int32(self._max_rounds),
                ),
            )
            draws = cp.asnumpy(self._draws)
            round_counts = cp.asnumpy(self._round_counts)

        failed = round_counts > self._max_rounds
        if np.any(failed):
            raise RuntimeError(
                'CUDA GIG rejection sampler exceeded %d rounds with '
                '%d active draw(s)' %
                (self._max_rounds, int(np.count_nonzero(failed)))
            )
        if not np.isfinite(draws).all() or not np.all(draws > 0.0):
            raise RuntimeError(
                'CUDA GIG rejection sampler produced invalid draws'
            )

        out[:] = draws
        rounds = int(round_counts.max())
        self._call_index += 1
        self._sample_calls += 1
        self._rounds_total += rounds
        self._rounds_maximum = max(self._rounds_maximum, rounds)

    def describe(self):
        return 'cuda:%d (single-kernel GIG; Philox 4x32-10)' % (
            self._device.id
        )

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
        return CudaRawPsiBackend(
            size, seed=seed, cuda_device=cuda_device,
            cuda_gig_max_rounds=cuda_gig_max_rounds,
        )
    raise ValueError(
        "unknown psi backend %r; expected 'cpu' or 'cuda'" % backend
    )
