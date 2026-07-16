#!/usr/bin/env python3

"""Tests for the experimental projected-summary-statistics mode."""


import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

import PRScs
import parse_genet


class ProjectedSummaryCliTests(unittest.TestCase):
    def _argv(self):
        return [
            'PRScs.py',
            '--ref_dir=/tmp/ldblk_ukbb_eur',
            '--bim_prefix=/tmp/target',
            '--sst_file=/tmp/sumstats',
            '--n_gwas=1000',
            '--out_dir=/tmp/output',
        ]

    def test_projection_options_are_parsed(self):
        argv = self._argv() + [
            '--project_sumstats=true',
            '--projection_fraction=0.4',
            '--projection_min_eigenvalue=0.02',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                parameters = PRScs.parse_param()

        self.assertEqual(parameters['project_sumstats'], 'TRUE')
        self.assertEqual(parameters['projection_fraction'], 0.4)
        self.assertEqual(parameters['projection_min_eigenvalue'], 0.02)

    def test_projection_rejects_ld_cache(self):
        argv = self._argv() + [
            '--project_sumstats=true',
            '--ld_cache_dir=/tmp/ld-cache',
        ]
        with mock.patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, '2'):
                    PRScs.parse_param()

    def test_invalid_projection_controls_are_rejected(self):
        invalid_options = [
            ['--project_sumstats=sometimes'],
            ['--projection_fraction=-0.1'],
            ['--projection_fraction=1'],
            ['--projection_fraction=nan'],
            ['--projection_min_eigenvalue=-0.1'],
            ['--projection_min_eigenvalue=inf'],
        ]
        for options in invalid_options:
            with self.subTest(options=options):
                with mock.patch.object(
                        sys, 'argv', self._argv() + options):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(SystemExit, '2'):
                            PRScs.parse_param()


class ProjectedSummaryLinearAlgebraTests(unittest.TestCase):
    def test_minimum_eigenvalue_truncates_ld_and_beta_together(self):
        ld = np.diag([2.0, 1.0, 0.001])
        beta = np.array([1.0, 2.0, 3.0])

        projected_ld, projected_beta = \
            parse_genet._truncate_ld_and_project_sumstats(
                ld, beta, fraction=0.0, min_eigenvalue=0.01
            )

        np.testing.assert_allclose(
            projected_ld, np.diag([2.0, 1.0, 0.0]), atol=1e-14
        )
        np.testing.assert_allclose(projected_beta, [1.0, 2.0, 0.0])
        self.assertTrue(projected_ld.flags.f_contiguous)

    def test_fraction_can_truncate_beyond_eigenvalue_floor(self):
        ld = np.diag([2.0, 1.0, 0.5])
        beta = np.array([1.0, 2.0, 3.0])

        projected_ld, projected_beta = \
            parse_genet._truncate_ld_and_project_sumstats(
                ld, beta, fraction=2.0 / 3.0, min_eigenvalue=0.0
            )

        np.testing.assert_allclose(
            projected_ld, np.diag([2.0, 0.0, 0.0]), atol=1e-14
        )
        np.testing.assert_allclose(projected_beta, [1.0, 0.0, 0.0])

    def test_loader_projects_the_matching_summary_block(self):
        with tempfile.TemporaryDirectory(prefix='ldblk_1kg_') as directory:
            filename = os.path.join(
                directory, 'ldblk_1kg_chr22.hdf5'
            )
            with h5py.File(filename, 'w') as handle:
                group = handle.create_group('blk_1')
                group.create_dataset(
                    'ldblk', data=np.diag([2.0, 1.0, 0.001])
                )
                group.create_dataset(
                    'snplist', data=np.asarray([b'rs1', b'rs2', b'rs3'])
                )

            summary = {
                'SNP': ['rs1', 'rs2', 'rs3'],
                'FLP': [1, 1, 1],
                'BETA': [1.0, 2.0, 3.0],
            }
            with contextlib.redirect_stdout(io.StringIO()):
                blocks, sizes = parse_genet.parse_ldblk(
                    directory, summary, 22,
                    project_sumstats=True,
                    projection_fraction=0.0,
                    projection_min_eigenvalue=0.01,
                )

        self.assertEqual(sizes, [3])
        np.testing.assert_allclose(
            blocks[0], np.diag([2.0, 1.0, 0.0]), atol=1e-14
        )
        np.testing.assert_allclose(summary['BETA'], [1.0, 2.0, 0.0])


if __name__ == '__main__':
    unittest.main()
