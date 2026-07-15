#!/usr/bin/env python3

"""Implementations of the PRS-CS local shrinkage update."""


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


def make_psi_backend(backend, size, seed=None):
    """Construct the requested local-shrinkage sampler."""
    del size
    backend = str(backend).lower()
    if backend == 'cpu':
        return CpuPsiBackend(seed=seed)
    raise ValueError("unknown psi backend %r; expected 'cpu'" % backend)
