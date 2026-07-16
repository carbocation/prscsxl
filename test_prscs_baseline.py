#!/usr/bin/env python3

"""Characterization tests for the long-lived fork's current master."""


import contextlib
import io
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import h5py
import numpy as np
from scipy import linalg

import PRScs
import benchmark_gpu
import mcmc_gtb
import parse_genet


class CommandLineCharacterizationTests(unittest.TestCase):
    def test_existing_options_and_defaults_are_preserved(self):
        argv = [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
            '--chrom=1,22',
            '--seed=123',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                parameters = PRScs.parse_param()

        self.assertEqual(parameters['chrom'], ['1', '22'])
        self.assertEqual(parameters['seed'], 123)
        self.assertEqual(parameters['a'], 1)
        self.assertEqual(parameters['b'], 0.5)
        self.assertIsNone(parameters['phi'])
        self.assertEqual(parameters['n_iter'], 1000)
        self.assertEqual(parameters['n_burnin'], 500)
        self.assertEqual(parameters['thin'], 5)
        self.assertEqual(parameters['chromosome_model'], 'independent')
        self.assertEqual(parameters['beta_std'], 'FALSE')
        self.assertEqual(parameters['write_psi'], 'FALSE')
        self.assertEqual(parameters['write_pst'], 'FALSE')
        self.assertEqual(parameters['backend'], 'cpu')
        self.assertEqual(parameters['cuda_device'], 0)
        self.assertEqual(parameters['cuda_bucket_size'], 32)
        self.assertEqual(parameters['cuda_streams'], 4)
        self.assertEqual(parameters['cuda_gig_max_rounds'], 1000)
        self.assertEqual(parameters['ld_diagnostics'], 'FALSE')
        self.assertEqual(parameters['profile'], 'FALSE')

    def test_cuda_backend_and_diagnostic_options_are_parsed(self):
        argv = [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
            '--backend=CUDA',
            '--cuda_device=1',
            '--cuda_bucket_size=64',
            '--cuda_gig_max_rounds=77',
            '--ld_diagnostics=true',
            '--ld_rank_tol=1e-7',
            '--profile=true',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                parameters = PRScs.parse_param()

        self.assertEqual(parameters['backend'], 'cuda')
        self.assertEqual(parameters['cuda_device'], 1)
        self.assertEqual(parameters['cuda_bucket_size'], 64)
        self.assertEqual(parameters['cuda_gig_max_rounds'], 77)
        self.assertEqual(parameters['ld_diagnostics'], 'TRUE')
        self.assertEqual(parameters['ld_rank_tol'], 1e-7)
        self.assertEqual(parameters['profile'], 'TRUE')

    def test_unknown_backend_is_rejected(self):
        argv = [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
            '--backend=warp-drive',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_context:
                    PRScs.parse_param()
        self.assertEqual(exit_context.exception.code, 2)

    def test_invalid_sampling_schedules_are_rejected(self):
        base_argv = [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
        ]
        invalid_options = [
            ['--n_iter=0'],
            ['--n_iter=10', '--n_burnin=-1'],
            ['--n_iter=10', '--n_burnin=10'],
            ['--thin=0'],
            ['--n_iter=10', '--n_burnin=8', '--thin=3'],
        ]

        for options in invalid_options:
            with self.subTest(options=options):
                with mock.patch.object(sys, 'argv', base_argv + options):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as exit_context:
                            PRScs.parse_param()
                self.assertEqual(exit_context.exception.code, 2)

    def test_benchmark_accepts_only_cpu_and_cuda_backends(self):
        with mock.patch.object(
                sys, 'argv',
                ['benchmark_gpu.py', '--backends=warp-drive']):
            arguments = benchmark_gpu.parse_args()
        with self.assertRaisesRegex(ValueError, 'unknown backend'):
            benchmark_gpu.validate_args(arguments)


class SummaryStatisticsCharacterizationTests(unittest.TestCase):
    def test_direct_swapped_and_complement_alleles_match_reference_order(self):
        reference = {
            'CHR': [1, 1, 1, 1],
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'BP': [1, 2, 3, 4],
            'A1': ['A', 'G', 'A', 'C'],
            'A2': ['C', 'A', 'G', 'T'],
            'MAF': [0.1, 0.2, 0.3, 0.4],
        }
        validation = {
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'A1': ['A', 'A', 'T', 'A'],
            'A2': ['C', 'G', 'C', 'G'],
        }

        with tempfile.TemporaryDirectory() as directory:
            sumstats = os.path.join(directory, 'sumstats.txt')
            with open(sumstats, 'w') as handle:
                handle.write('SNP A1 A2 BETA SE\n')
                handle.write('rs1 A C 0.10 0.01\n')
                handle.write('rs2 G A 0.20 0.02\n')
                handle.write('rs3 A G 0.30 0.03\n')
                handle.write('rs4 C T 0.40 0.04\n')

            with contextlib.redirect_stdout(io.StringIO()):
                actual = parse_genet.parse_sumstats(
                    reference, validation, sumstats, 100
                )

        self.assertEqual(actual['SNP'], reference['SNP'])
        self.assertEqual(actual['A1'], validation['A1'])
        self.assertEqual(actual['A2'], validation['A2'])
        np.testing.assert_allclose(actual['BETA'], [1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(actual['MAF'], [0.1, 0.8, 0.3, 0.6])
        self.assertEqual(actual['FLP'], [1, -1, 1, -1])


class LinkageDisequilibriumCharacterizationTests(unittest.TestCase):
    def test_subsetting_flips_and_projects_like_the_legacy_parser(self):
        raw = np.array([
            [1.0, 1.2, 0.1],
            [1.2, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        selected = raw[np.ix_([0, 2], [0, 2])]
        selected *= np.outer([1, -1], [1, -1])
        _, singular_values, right = linalg.svd(selected)
        expected = (
            selected + right.T @ np.diag(singular_values) @ right
        ) / 2.0

        with tempfile.TemporaryDirectory(
                prefix='ldblk_1kg_') as directory:
            filename = os.path.join(directory, 'ldblk_1kg_chr22.hdf5')
            with h5py.File(filename, 'w') as handle:
                group = handle.create_group('blk_1')
                group.create_dataset('ldblk', data=raw)
                group.create_dataset(
                    'snplist', data=np.asarray([b'rs1', b'rs2', b'rs3'])
                )

            summary = {'SNP': ['rs1', 'rs3'], 'FLP': [1, -1]}
            with contextlib.redirect_stdout(io.StringIO()):
                blocks, sizes = parse_genet.parse_ldblk(
                    directory, summary, 22
                )

        self.assertEqual(sizes, [2])
        np.testing.assert_allclose(
            blocks[0], expected, rtol=1e-12, atol=1e-12
        )


class McmcCharacterizationTests(unittest.TestCase):
    def test_seeded_fixed_phi_beta_transition_matches_master(self):
        summary = {
            'SNP': ['rs1', 'rs2', 'rs3'],
            'BP': [1, 2, 3],
            'A1': ['A', 'C', 'G'],
            'A2': ['G', 'T', 'A'],
            'MAF': [0.2, 0.3, 0.4],
            'BETA': [0.02, -0.01, 0.03],
        }
        blocks = [
            np.array([[1.0, 0.2], [0.2, 1.0]]),
            np.array([[1.0]]),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'result')
            psi_sampler = mock.Mock()
            psi_sampler.describe.return_value = 'cpu test double'
            psi_sampler.sample.side_effect = (
                lambda out, *_args: out.fill(0.5)
            )
            with mock.patch.object(
                    mcmc_gtb, 'make_psi_backend',
                    return_value=psi_sampler):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        message='Conversion of an array with ndim > 0',
                        category=DeprecationWarning,
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        mcmc_gtb.mcmc(
                            1, 0.5, 0.01, summary, 1000, blocks,
                            [2, 1], 1, 0, 1, 22, output, 'FALSE',
                            'FALSE', 'FALSE', 123,
                        )

            effect_file = (
                output + '_pst_eff_a1_b0.5_phi1e-02_chr22.txt'
            )
            with open(effect_file) as handle:
                rows = [line.split() for line in handle]

        self.assertEqual([row[1] for row in rows], summary['SNP'])
        np.testing.assert_allclose(
            [float(row[5]) for row in rows],
            [-2.812649e-02, 2.523333e-02, 3.078373e-02],
            rtol=0.0,
            atol=5e-9,
        )


if __name__ == '__main__':
    unittest.main()
