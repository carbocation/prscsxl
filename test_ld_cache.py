#!/usr/bin/env python3

"""Tests for persistent filtered and projected LD caching."""


import contextlib
import glob
import io
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

import parse_genet


class ProjectedLdCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory_context = tempfile.TemporaryDirectory(
            prefix='ldblk_1kg_'
        )
        self.directory = self.directory_context.name
        self.cache_dir = os.path.join(self.directory, 'cache')
        filename = os.path.join(
            self.directory, 'ldblk_1kg_chr22.hdf5'
        )
        with h5py.File(filename, 'w') as handle:
            group = handle.create_group('blk_1')
            group.create_dataset(
                'ldblk',
                data=np.array([
                    [1.0, 0.2, 0.1],
                    [0.2, 1.0, 0.3],
                    [0.1, 0.3, 1.0],
                ]),
            )
            group.create_dataset(
                'snplist', data=np.asarray([b'rs1', b'rs2', b'rs3'])
            )

    def tearDown(self):
        self.directory_context.cleanup()

    def test_cache_reuses_exact_projection_and_tracks_allele_flips(self):
        summary = {'SNP': ['rs1', 'rs3'], 'FLP': [1, -1]}
        with mock.patch.object(
                parse_genet, '_project_ld_psd',
                wraps=parse_genet._project_ld_psd) as projection:
            with contextlib.redirect_stdout(io.StringIO()):
                first_blocks, first_sizes = parse_genet.parse_ldblk(
                    self.directory, summary, 22,
                    cache_dir=self.cache_dir,
                )
            self.assertEqual(projection.call_count, 1)

        with mock.patch.object(
                parse_genet, '_project_ld_psd',
                side_effect=AssertionError('cache miss')):
            with contextlib.redirect_stdout(io.StringIO()):
                cached_blocks, cached_sizes = parse_genet.parse_ldblk(
                    self.directory, summary, 22,
                    cache_dir=self.cache_dir,
                )

        self.assertEqual(cached_sizes, first_sizes)
        self.assertTrue(cached_blocks[0].flags.f_contiguous)
        np.testing.assert_array_equal(cached_blocks[0], first_blocks[0])

        changed = {'SNP': ['rs1', 'rs3'], 'FLP': [1, 1]}
        with mock.patch.object(
                parse_genet, '_project_ld_psd',
                wraps=parse_genet._project_ld_psd) as projection:
            with contextlib.redirect_stdout(io.StringIO()):
                changed_blocks, _ = parse_genet.parse_ldblk(
                    self.directory, changed, 22,
                    cache_dir=self.cache_dir,
                )
            self.assertEqual(projection.call_count, 1)
        self.assertFalse(np.array_equal(
            changed_blocks[0], cached_blocks[0]
        ))

    def test_corrupt_cache_is_ignored_and_replaced(self):
        summary = {'SNP': ['rs1', 'rs3'], 'FLP': [1, -1]}
        with contextlib.redirect_stdout(io.StringIO()):
            parse_genet.parse_ldblk(
                self.directory, summary, 22, cache_dir=self.cache_dir
            )
        cache_files = glob.glob(os.path.join(self.cache_dir, '*.hdf5'))
        self.assertEqual(len(cache_files), 1)
        with open(cache_files[0], 'wb') as ff:
            ff.write(b'not an HDF5 file')

        with mock.patch.object(
                parse_genet, '_project_ld_psd',
                wraps=parse_genet._project_ld_psd) as projection:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                blocks, sizes = parse_genet.parse_ldblk(
                    self.directory, summary, 22,
                    cache_dir=self.cache_dir,
                )

        self.assertEqual(projection.call_count, 1)
        self.assertIn('ignore unusable LD cache', output.getvalue())
        self.assertEqual(sizes, [2])
        self.assertEqual(blocks[0].shape, (2, 2))
        with h5py.File(cache_files[0], 'r') as cached:
            self.assertEqual(int(cached.attrs['version']), 1)


if __name__ == '__main__':
    unittest.main()
