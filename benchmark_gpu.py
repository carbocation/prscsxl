#!/usr/bin/env python3

"""Synthetic within-chromosome benchmark for PRS-CS beta backends."""


import argparse
import tempfile
import time

import numpy as np

import gigrnd
from mcmc_gtb import mcmc
from parse_genet import _project_ld_psd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--backends', default='cpu,cuda',
        help='comma-separated subset of cpu,cuda',
    )
    parser.add_argument('--block-size', type=int, default=400)
    parser.add_argument('--n-blocks', type=int, default=100)
    parser.add_argument('--n-iter', type=int, default=30)
    parser.add_argument('--n-burnin', type=int, default=10)
    parser.add_argument('--thin', type=int, default=2)
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('--cuda-bucket-size', type=int, default=32)
    parser.add_argument('--cuda-streams', type=int, default=4)
    parser.add_argument(
        '--psi-backend', choices=('cpu', 'cuda'), default='cpu'
    )
    parser.add_argument('--cuda-gig-max-rounds', type=int, default=1000)
    return parser.parse_args()


def validate_args(args):
    backends = [value.strip() for value in args.backends.split(',')]
    unknown = set(backends) - {'cpu', 'cuda'}
    if unknown:
        raise ValueError(
            'unknown backend(s): %s' % ', '.join(sorted(unknown))
        )
    if args.block_size < 1 or args.n_blocks < 1:
        raise ValueError('block-size and n-blocks must be positive')
    if args.n_iter < 1 or not 0 <= args.n_burnin < args.n_iter:
        raise ValueError('n-burnin must be in [0, n-iter)')
    if args.thin < 1:
        raise ValueError('thin must be positive')
    if args.cuda_streams < 1:
        raise ValueError('cuda-streams must be positive')
    return backends


def gpu_name(device_id):
    try:
        import cupy as cp
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties['name']
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return 'unavailable'


def main():
    args = parse_args()
    backends = validate_args(args)
    variant_count = args.block_size * args.n_blocks

    indices = np.arange(args.block_size)
    raw_ld = 0.9 ** np.abs(indices[:, None] - indices[None, :])
    ld = _project_ld_psd(raw_ld)
    ld_blocks = [ld] * args.n_blocks
    block_sizes = [args.block_size] * args.n_blocks

    rng = np.random.default_rng(123)
    summary = {
        'SNP': ['rs%d' % (index + 1) for index in range(variant_count)],
        'BP': np.arange(1, variant_count + 1),
        'A1': ['A'] * variant_count,
        'A2': ['G'] * variant_count,
        'BETA': rng.normal(0.0, 1e-3, variant_count),
        'MAF': np.full(variant_count, 0.25),
    }

    # Compile the Numba psi kernel before any backend is timed.
    gigrnd.gig_rvs_vec(
        np.ones(1), 0.5, np.ones(1), np.full(1, 1e-3),
        1.0, 100_000,
    )

    print('GPU: %s' % gpu_name(args.cuda_device))
    print(
        'Synthetic workload: %d variants, %d blocks x %d, %d iterations' %
        (variant_count, args.n_blocks, args.block_size, args.n_iter)
    )
    timings = {}
    with tempfile.TemporaryDirectory() as directory:
        for backend in backends:
            print('\n===== %s BENCHMARK =====' % backend.upper())
            started = time.perf_counter()
            mcmc(
                a=1.0,
                b=0.5,
                phi=1e-2,
                sst_dict=summary,
                n=100_000,
                ld_blk=ld_blocks,
                blk_size=block_sizes,
                n_iter=args.n_iter,
                n_burnin=args.n_burnin,
                thin=args.thin,
                chrom=22,
                out_dir='%s/%s' % (directory, backend),
                beta_std='TRUE',
                write_psi='FALSE',
                write_pst='FALSE',
                seed=123,
                backend=backend,
                cuda_device=args.cuda_device,
                cuda_bucket_size=args.cuda_bucket_size,
                cuda_streams=args.cuda_streams,
                profile='TRUE',
                psi_backend=args.psi_backend,
                cuda_gig_max_rounds=args.cuda_gig_max_rounds,
            )
            timings[backend] = time.perf_counter() - started
            print(
                '%s wall time: %.3f seconds' %
                (backend.upper(), timings[backend])
            )

    print('\n===== RESULT =====')
    for backend in backends:
        print('%-6s %.3f seconds' % (backend + ':', timings[backend]))
    if 'cpu' in timings and 'cuda' in timings:
        print('Speedup: %.2fx' % (timings['cpu'] / timings['cuda']))


if __name__ == '__main__':
    main()
