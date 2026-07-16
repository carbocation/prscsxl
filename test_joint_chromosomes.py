#!/usr/bin/env python3

"""Tests for opt-in joint chromosome sampling."""


import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import PRScs
import mcmc_gtb
import parse_genet


def _summary(chromosome, snps):
    size = len(snps)
    return {
        'CHR': [chromosome] * size,
        'SNP': list(snps),
        'BP': list(range(1, size + 1)),
        'A1': ['A'] * size,
        'A2': ['C'] * size,
        'MAF': [0.25] * size,
        'BETA': [0.0] * size,
        'FLP': [1] * size,
    }


class JointChromosomeCliTests(unittest.TestCase):
    def test_joint_chromosomes_flag_uses_existing_boolean_style(self):
        argv = [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
            '--joint_chromosomes=true',
            '--ld_cache_dir=/tmp/ld-cache',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                parameters = PRScs.parse_param()

        self.assertEqual(parameters['joint_chromosomes'], 'TRUE')
        self.assertEqual(parameters['ld_cache_dir'], '/tmp/ld-cache')

    def test_invalid_joint_chromosomes_value_is_rejected(self):
        argv = [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
            '--joint_chromosomes=sometimes',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, '2'):
                    PRScs.parse_param()


class JointTextParserTests(unittest.TestCase):
    def test_joint_parser_matches_chromosome_wise_allele_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, 'snpinfo_ukbb_hm3')
            bim_prefix = os.path.join(directory, 'target')
            sumstats = os.path.join(directory, 'sumstats.txt')

            with open(reference, 'w') as ff:
                ff.write('CHR SNP BP A1 A2 MAF\n')
                ff.write('1 rs1 1 A C 0.10\n')
                ff.write('1 rs2 2 G A 0.20\n')
                ff.write('2 rs3 3 A G 0.30\n')
                ff.write('2 rs4 4 C T 0.40\n')

            with open(bim_prefix + '.bim', 'w') as ff:
                ff.write('1 rs1 0 1 A C\n')
                ff.write('1 rs2 0 2 A G\n')
                ff.write('2 rs3 0 3 T C\n')
                ff.write('2 rs4 0 4 A G\n')

            with open(sumstats, 'w') as ff:
                ff.write('SNP A1 A2 BETA SE\n')
                ff.write('rs1 A C 0.10 0.01\n')
                ff.write('rs2 G A 0.20 0.02\n')
                ff.write('rs3 A G 0.30 0.03\n')
                ff.write('rs4 C T 0.40 0.04\n')
                ff.write('rsjunk A C not-a-number invalid\n')

            with contextlib.redirect_stdout(io.StringIO()):
                expected = {}
                for chromosome in (1, 2):
                    ref_dict = parse_genet.parse_ref(
                        reference, chromosome
                    )
                    vld_dict = parse_genet.parse_bim(
                        bim_prefix, chromosome
                    )
                    expected[chromosome] = parse_genet.parse_sumstats(
                        ref_dict, vld_dict, sumstats, 100
                    )

                ref_dicts = parse_genet.parse_ref_chromosomes(
                    reference, (1, 2)
                )
                vld_dict = parse_genet.parse_bim_chromosomes(
                    bim_prefix, (1, 2)
                )
                actual = parse_genet.parse_sumstats_chromosomes(
                    ref_dicts, vld_dict, sumstats, 100
                )

            self.assertEqual(actual, expected)


class JointChromosomeInputTests(unittest.TestCase):
    def test_chromosome_inputs_are_concatenated_with_output_slices(self):
        inputs = [
            {
                'chrom': 1,
                'sst_dict': _summary(1, ['rs1', 'rs2']),
                'ld_blk': [np.eye(2)],
                'blk_size': [2],
            },
            {
                'chrom': 2,
                'sst_dict': _summary(2, ['rs3', 'rs4', 'rs5']),
                'ld_blk': [np.eye(3)],
                'blk_size': [3],
            },
        ]

        combined = PRScs._combine_chromosomes(inputs)

        self.assertEqual(
            combined['sst_dict']['SNP'],
            ['rs1', 'rs2', 'rs3', 'rs4', 'rs5'],
        )
        self.assertEqual(combined['blk_size'], [2, 3])
        self.assertEqual(
            combined['chromosome_slices'], [(1, 0, 2), (2, 2, 5)]
        )

    def test_empty_selected_chromosome_is_rejected(self):
        inputs = [
            {
                'chrom': 1,
                'sst_dict': _summary(1, []),
                'ld_blk': [],
                'blk_size': [],
            },
            {
                'chrom': 2,
                'sst_dict': _summary(2, ['rs2']),
                'ld_blk': [np.eye(1)],
                'blk_size': [1],
            },
        ]

        with self.assertRaisesRegex(ValueError, 'selected chromosome 1'):
            PRScs._combine_chromosomes(inputs)

    def test_duplicate_selected_chromosome_is_rejected(self):
        item = {
            'chrom': 1,
            'sst_dict': _summary(1, ['rs1']),
            'ld_blk': [np.eye(1)],
            'blk_size': [1],
        }
        with self.assertRaisesRegex(ValueError, 'chromosome 1 twice'):
            PRScs._combine_chromosomes([item, item])


class JointChromosomeMainTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [
            {
                'chrom': chromosome,
                'sst_dict': _summary(chromosome, ['rs%d' % chromosome]),
                'ld_blk': [np.eye(1)],
                'blk_size': [1],
            }
            for chromosome in (1, 2)
        ]

    def test_joint_mode_invokes_one_sampler_for_selected_chromosomes(self):
        parameters = {
            'chrom': [1, 2],
            'joint_chromosomes': 'TRUE',
        }
        with mock.patch.object(PRScs, 'parse_param', return_value=parameters):
            with mock.patch.object(
                    PRScs, '_load_joint_chromosomes',
                    return_value=self.inputs):
                with mock.patch.object(PRScs, '_run_mcmc') as run_mcmc:
                    with contextlib.redirect_stdout(io.StringIO()):
                        PRScs.main()

        run_mcmc.assert_called_once()
        self.assertEqual(run_mcmc.call_args.args[2], [1, 2])
        self.assertEqual(
            run_mcmc.call_args.kwargs['chromosome_slices'],
            [(1, 0, 1), (2, 1, 2)],
        )

    def test_default_mode_preserves_one_sampler_per_chromosome(self):
        parameters = {
            'chrom': [1, 2],
            'joint_chromosomes': 'FALSE',
        }
        with mock.patch.object(PRScs, 'parse_param', return_value=parameters):
            with mock.patch.object(
                    PRScs, '_load_chromosome', side_effect=self.inputs):
                with mock.patch.object(PRScs, '_run_mcmc') as run_mcmc:
                    with contextlib.redirect_stdout(io.StringIO()):
                        PRScs.main()

        self.assertEqual(run_mcmc.call_count, 2)
        self.assertEqual(
            [call.args[2] for call in run_mcmc.call_args_list], [1, 2]
        )
        self.assertTrue(
            all('chromosome_slices' not in call.kwargs
                for call in run_mcmc.call_args_list)
        )


class _FixedBetaBackend:
    def __init__(self, size, beta=0.0, quad=0.0):
        self._size = size
        self._beta = float(beta)
        self._quad = float(quad)

    def describe(self):
        return 'fixed test beta backend'

    def sample(self, psi, sigma):
        return np.full((self._size, 1), self._beta), self._quad


class _FixedPsiBackend:
    def describe(self):
        return 'fixed test psi backend'

    def sample_joint(self, output, *args, **kwargs):
        output.fill(0.5)
        return 6.0


class JointChromosomeSamplerTests(unittest.TestCase):
    def test_sigma_residual_safeguard_reports_material_activations(self):
        summary = _summary(1, ['rs1'])
        summary['BETA'][0] = 1.0
        beta_factory = mock.Mock(
            return_value=_FixedBetaBackend(1, beta=1.0, quad=1.0)
        )
        psi_factory = mock.Mock(return_value=_FixedPsiBackend())

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'sigma-floor')
            with mock.patch.object(
                    mcmc_gtb, 'make_beta_backend', beta_factory):
                with mock.patch.object(
                        mcmc_gtb, 'make_psi_backend', psi_factory):
                    with mock.patch.object(
                            mcmc_gtb.np.random, 'gamma', return_value=1.0):
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            mcmc_gtb.mcmc(
                                1.0, 0.5, 0.1, summary, 100,
                                [np.eye(1)], [1], 2, 0, 1, 1, output,
                                'TRUE', 'FALSE', 'FALSE', 123,
                            )

        self.assertIn(
            'WARNING: sigma residual safeguard: 2/2 activations '
            '(2 material; first iteration 1; worst relative deficit ',
            stdout.getvalue(),
        )

    def test_sigma_residual_safeguard_reports_clean_run(self):
        summary = _summary(1, ['rs1'])
        beta_factory = mock.Mock(return_value=_FixedBetaBackend(1))
        psi_factory = mock.Mock(return_value=_FixedPsiBackend())

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'sigma-clean')
            with mock.patch.object(
                    mcmc_gtb, 'make_beta_backend', beta_factory):
                with mock.patch.object(
                        mcmc_gtb, 'make_psi_backend', psi_factory):
                    with mock.patch.object(
                            mcmc_gtb.np.random, 'gamma', return_value=1.0):
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            mcmc_gtb.mcmc(
                                1.0, 0.5, 0.1, summary, 100,
                                [np.eye(1)], [1], 1, 0, 1, 1, output,
                                'TRUE', 'FALSE', 'FALSE', 123,
                            )

        self.assertIn(
            'sigma residual safeguard: 0/1 activations', stdout.getvalue()
        )

    def test_nonfinite_sigma_statistics_fail_with_iteration_context(self):
        with self.assertRaisesRegex(
                FloatingPointError,
                'non-finite sigma sufficient statistic at iteration 7'):
            mcmc_gtb._sigma_floor_relative_deficit(np.nan, 1.0, 7)

    def test_posterior_thinning_is_measured_from_end_of_burnin(self):
        self.assertEqual(
            list(mcmc_gtb._posterior_iterations(10, 3, 4)), [7]
        )
        self.assertEqual(
            list(mcmc_gtb._posterior_iterations(10, 4, 2)), [6, 8, 10]
        )

    def test_non_aligned_burnin_writes_the_expected_number_of_draws(self):
        summary = _summary(1, ['rs1'])
        beta_factory = mock.Mock(return_value=_FixedBetaBackend(1))
        psi_factory = mock.Mock(return_value=_FixedPsiBackend())

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'retention')
            with mock.patch.object(
                    mcmc_gtb, 'make_beta_backend', beta_factory):
                with mock.patch.object(
                        mcmc_gtb, 'make_psi_backend', psi_factory):
                    with mock.patch.object(
                            mcmc_gtb.np.random, 'gamma', return_value=1.0):
                        with contextlib.redirect_stdout(io.StringIO()):
                            mcmc_gtb.mcmc(
                                1.0, 0.5, 0.1, summary, 100,
                                [np.eye(1)], [1], 10, 3, 4, 1, output,
                                'TRUE', 'FALSE', 'TRUE', 123,
                            )

            effect_file = output + '_pst_eff_a1_b0.5_phi1e-01_chr1.txt'
            with open(effect_file) as ff:
                fields = ff.readline().split()

        # Five identifying fields followed by the one retained beta draw.
        self.assertEqual(len(fields), 6)

    def test_joint_profile_label_compacts_consecutive_chromosomes(self):
        partitions = [(chromosome, 0, 0) for chromosome in range(1, 23)]
        self.assertEqual(
            mcmc_gtb._profile_label(partitions, True),
            'joint chr1-22',
        )

    def test_one_chain_updates_global_parameters_and_splits_outputs(self):
        summary = _summary(1, ['rs1', 'rs2'])
        second = _summary(2, ['rs3', 'rs4', 'rs5'])
        for key in summary:
            summary[key].extend(second[key])

        beta_factory = mock.Mock(
            return_value=_FixedBetaBackend(len(summary['SNP']))
        )
        psi_factory = mock.Mock(return_value=_FixedPsiBackend())

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'joint')
            with mock.patch.object(
                    mcmc_gtb, 'make_beta_backend', beta_factory):
                with mock.patch.object(
                        mcmc_gtb, 'make_psi_backend', psi_factory):
                    with mock.patch.object(
                            mcmc_gtb.np.random, 'gamma',
                            return_value=1.0) as gamma:
                        with contextlib.redirect_stdout(io.StringIO()):
                            mcmc_gtb.mcmc(
                                1.0, 0.5, None, summary, 100,
                                [np.eye(2), np.eye(3)], [2, 3],
                                1, 0, 1, [1, 2], output, 'TRUE',
                                'TRUE', 'FALSE', 123,
                                chromosome_slices=[(1, 0, 2), (2, 2, 5)],
                                backend='cuda',
                            )

            # Sigma, the auxiliary global-scale draw, and phi are sampled
            # once. The fused psi backend supplies the all-SNP delta sum.
            self.assertEqual(gamma.call_count, 3)
            self.assertAlmostEqual(gamma.call_args_list[2].args[0], 3.0)
            self.assertAlmostEqual(gamma.call_args_list[2].args[1], 1.0 / 7.0)
            self.assertEqual(beta_factory.call_args.args[0], 'cuda')
            psi_factory.assert_called_once()
            self.assertEqual(psi_factory.call_args.args[0], 'cuda')
            self.assertEqual(psi_factory.call_args.args[1], 5)

            chromosome_one = output + '_pst_eff_a1_b0.5_phiauto_chr1.txt'
            chromosome_two = output + '_pst_eff_a1_b0.5_phiauto_chr2.txt'
            with open(chromosome_one) as ff:
                chromosome_one_lines = ff.readlines()
            with open(chromosome_two) as ff:
                chromosome_two_lines = ff.readlines()

            self.assertEqual(len(chromosome_one_lines), 2)
            self.assertEqual(len(chromosome_two_lines), 3)
            self.assertTrue(
                all(line.startswith('1\t') for line in chromosome_one_lines)
            )
            self.assertTrue(
                all(line.startswith('2\t') for line in chromosome_two_lines)
            )

            psi_one = output + '_pst_psi_a1_b0.5_phiauto_chr1.txt'
            psi_two = output + '_pst_psi_a1_b0.5_phiauto_chr2.txt'
            with open(psi_one) as ff:
                self.assertEqual(len(ff.readlines()), 2)
            with open(psi_two) as ff:
                self.assertEqual(len(ff.readlines()), 3)

    def test_joint_output_slices_must_cover_every_snp(self):
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            mcmc_gtb._chromosome_partitions(
                [1, 2], [(1, 0, 2), (2, 3, 5)], 5
            )


if __name__ == '__main__':
    unittest.main()
