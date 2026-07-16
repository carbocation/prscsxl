#!/usr/bin/env python3

"""
Parse the reference panel, summary statistics, and validation set.

"""


import hashlib
import os
import time
import numpy as np
from scipy.stats import norm
from scipy import linalg
import h5py


_LD_CACHE_VERSION = 1


def _project_ld_psd(ld):
    """Project a symmetric LD matrix onto the positive-semidefinite cone."""
    ld = np.asarray(ld, dtype=np.float64)
    symmetric_ld = (ld + ld.T) * 0.5
    eigenvalues, eigenvectors = linalg.eigh(
        symmetric_ld, check_finite=False
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return np.asfortranarray(
        np.dot(eigenvectors * eigenvalues[None, :], eigenvectors.T)
    )


def _truncate_ld_and_project_sumstats(
        ld, beta_marginal, fraction=0.2, min_eigenvalue=0.01):
    """Truncate LD and project marginal effects into the retained space."""
    fraction = float(fraction)
    min_eigenvalue = float(min_eigenvalue)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError('projection fraction must be in [0, 1)')
    if not np.isfinite(min_eigenvalue) or min_eigenvalue < 0.0:
        raise ValueError(
            'projection minimum eigenvalue must be finite and non-negative'
        )

    ld = np.asarray(ld, dtype=np.float64)
    beta_marginal = np.asarray(beta_marginal, dtype=np.float64).reshape(-1)
    if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
        raise ValueError('LD block must be a square matrix')
    if beta_marginal.size != ld.shape[0]:
        raise ValueError(
            'marginal effects and LD block must have equal dimensions'
        )

    symmetric_ld = (ld + ld.T) * 0.5
    eigenvalues, eigenvectors = linalg.eigh(
        symmetric_ld, check_finite=False
    )
    discard_count = max(
        int(np.count_nonzero(eigenvalues < min_eigenvalue)),
        int(np.floor(eigenvalues.size * fraction)),
    )
    retained_values = eigenvalues[discard_count:]
    retained_vectors = eigenvectors[:, discard_count:]

    if not retained_values.size:
        return (
            np.asfortranarray(np.zeros_like(symmetric_ld)),
            np.zeros_like(beta_marginal),
        )

    retained_values = np.maximum(retained_values, 0.0)
    projected_ld = np.asfortranarray(np.dot(
        retained_vectors * retained_values[None, :],
        retained_vectors.T,
    ))
    projected_beta = np.dot(
        retained_vectors,
        np.dot(retained_vectors.T, beta_marginal),
    )
    return projected_ld, projected_beta


def parse_ref(ref_file, chrom):
    print('... parse reference file: %s ...' % ref_file)

    ref_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[]}
    with open(ref_file) as ff:
        next(ff)
        for line in ff:
            ll = (line.strip()).split()
            if int(ll[0]) == chrom:
                ref_dict['CHR'].append(chrom)
                ref_dict['SNP'].append(ll[1])
                ref_dict['BP'].append(int(ll[2]))
                ref_dict['A1'].append(ll[3])
                ref_dict['A2'].append(ll[4])
                ref_dict['MAF'].append(float(ll[5]))

    print('... %d SNPs on chromosome %d read from %s ...' % (len(ref_dict['SNP']), chrom, ref_file))
    return ref_dict


def parse_bim(bim_file, chrom):
    print('... parse bim file: %s ...' % (bim_file + '.bim'))

    vld_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(bim_file + '.bim') as ff:
        for line in ff:
            ll = (line.strip()).split()
            if int(ll[0]) == chrom:
                vld_dict['SNP'].append(ll[1])
                vld_dict['A1'].append(ll[4])
                vld_dict['A2'].append(ll[5])

    print('... %d SNPs on chromosome %d read from %s ...' % (len(vld_dict['SNP']), chrom, bim_file + '.bim'))
    return vld_dict


def parse_ref_chromosomes(ref_file, chromosomes):
    """Read reference metadata for several chromosomes in one file scan."""
    chromosomes = tuple(int(chromosome) for chromosome in chromosomes)
    selected = set(chromosomes)
    ref_dicts = {
        chromosome: {
            'CHR': [], 'SNP': [], 'BP': [], 'A1': [], 'A2': [], 'MAF': []
        }
        for chromosome in chromosomes
    }

    print('... parse reference file once for joint chromosomes: %s ...' %
          ref_file)
    with open(ref_file) as ff:
        next(ff)
        for line in ff:
            ll = line.split()
            chromosome = int(ll[0])
            if chromosome not in selected:
                continue
            ref_dict = ref_dicts[chromosome]
            ref_dict['CHR'].append(chromosome)
            ref_dict['SNP'].append(ll[1])
            ref_dict['BP'].append(int(ll[2]))
            ref_dict['A1'].append(ll[3])
            ref_dict['A2'].append(ll[4])
            ref_dict['MAF'].append(float(ll[5]))

    for chromosome in chromosomes:
        print('... %d SNPs on chromosome %d read from %s ...' % (
            len(ref_dicts[chromosome]['SNP']), chromosome, ref_file
        ))
    return ref_dicts


def parse_bim_chromosomes(bim_file, chromosomes):
    """Read validation alleles for several chromosomes in one file scan."""
    chromosomes = tuple(int(chromosome) for chromosome in chromosomes)
    selected = set(chromosomes)
    counts = {chromosome: 0 for chromosome in chromosomes}
    vld_dict = {'SNP': [], 'A1': [], 'A2': []}
    path = bim_file + '.bim'

    print('... parse bim file once for joint chromosomes: %s ...' % path)
    with open(path) as ff:
        for line in ff:
            ll = line.split()
            chromosome = int(ll[0])
            if chromosome not in selected:
                continue
            vld_dict['SNP'].append(ll[1])
            vld_dict['A1'].append(ll[4])
            vld_dict['A2'].append(ll[5])
            counts[chromosome] += 1

    for chromosome in chromosomes:
        print('... %d SNPs on chromosome %d read from %s ...' % (
            counts[chromosome], chromosome, path
        ))
    return vld_dict


def _allele_orientations(snp, a1, a2, mapping):
    return (
        (snp, a1, a2),
        (snp, a2, a1),
        (snp, mapping[a1], mapping[a2]),
        (snp, mapping[a2], mapping[a1]),
    )


def _aligned_sumstats(ref_dict, comm_snp, sst_eff, mapping):
    sst_dict = {
        'CHR': [], 'SNP': [], 'BP': [], 'A1': [], 'A2': [],
        'MAF': [], 'BETA': [], 'FLP': []
    }
    for ii, snp in enumerate(ref_dict['SNP']):
        if snp not in sst_eff:
            continue
        sst_dict['SNP'].append(snp)
        sst_dict['CHR'].append(ref_dict['CHR'][ii])
        sst_dict['BP'].append(ref_dict['BP'][ii])
        sst_dict['BETA'].append(sst_eff[snp])

        a1 = ref_dict['A1'][ii]
        a2 = ref_dict['A2'][ii]
        if (snp, a1, a2) in comm_snp:
            sst_dict['A1'].append(a1)
            sst_dict['A2'].append(a2)
            sst_dict['MAF'].append(ref_dict['MAF'][ii])
            sst_dict['FLP'].append(1)
        elif (snp, a2, a1) in comm_snp:
            sst_dict['A1'].append(a2)
            sst_dict['A2'].append(a1)
            sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
            sst_dict['FLP'].append(-1)
        elif (snp, mapping[a1], mapping[a2]) in comm_snp:
            sst_dict['A1'].append(mapping[a1])
            sst_dict['A2'].append(mapping[a2])
            sst_dict['MAF'].append(ref_dict['MAF'][ii])
            sst_dict['FLP'].append(1)
        elif (snp, mapping[a2], mapping[a1]) in comm_snp:
            sst_dict['A1'].append(mapping[a2])
            sst_dict['A2'].append(mapping[a1])
            sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
            sst_dict['FLP'].append(-1)
    return sst_dict


def parse_sumstats_chromosomes(ref_dicts, vld_dict, sst_file, n_subj):
    """Align joint-chromosome summary statistics using two file scans."""
    print('... parse sumstats file twice for joint chromosomes: %s ...' %
          sst_file)
    alleles = {'A', 'T', 'G', 'C'}
    mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    vld_snp = set(zip(vld_dict['SNP'], vld_dict['A1'], vld_dict['A2']))
    ref_vld_snp = set()
    for ref_dict in ref_dicts.values():
        for snp, a1, a2 in zip(
                ref_dict['SNP'], ref_dict['A1'], ref_dict['A2']):
            for orientation in _allele_orientations(
                    snp, a1, a2, mapping):
                if orientation in vld_snp:
                    ref_vld_snp.add(orientation)

    comm_snp = set()
    valid_rows = 0
    with open(sst_file) as ff:
        next(ff)
        for line in ff:
            ll = line.split()
            if len(ll) < 3:
                continue
            a1 = ll[1]
            a2 = ll[2]
            if a1 not in alleles or a2 not in alleles:
                continue
            valid_rows += 1
            for orientation in _allele_orientations(
                    ll[0], a1, a2, mapping):
                if orientation in ref_vld_snp:
                    comm_snp.add(orientation)
    print('... %d SNPs read from %s ...' % (valid_rows, sst_file))

    n_sqrt = np.sqrt(n_subj)
    sst_eff = {}
    with open(sst_file) as ff:
        header = [column.upper() for column in next(ff).split()]
        for line in ff:
            ll = line.split()
            if len(ll) < 5:
                continue
            snp = ll[0]
            a1 = ll[1]
            a2 = ll[2]
            if a1 not in alleles or a2 not in alleles:
                continue

            if ((snp, a1, a2) in comm_snp or
                    (snp, mapping[a1], mapping[a2]) in comm_snp):
                direction = 1.0
            elif ((snp, a2, a1) in comm_snp or
                  (snp, mapping[a2], mapping[a1]) in comm_snp):
                direction = -1.0
            else:
                continue

            if 'BETA' in header:
                beta = float(ll[3])
            elif 'OR' in header:
                beta = np.log(float(ll[3]))

            if 'SE' in header:
                effect = beta / float(ll[4]) / n_sqrt
            elif 'P' in header:
                p = max(float(ll[4]), 1e-323)
                effect = np.sign(beta) * abs(norm.ppf(p/2.0)) / n_sqrt
            sst_eff[snp] = direction * effect

    sst_dicts = {}
    for chromosome, ref_dict in ref_dicts.items():
        sst_dict = _aligned_sumstats(
            ref_dict, comm_snp, sst_eff, mapping
        )
        sst_dicts[chromosome] = sst_dict
        print(
            '... %d common SNPs in the reference, sumstats, and '
            'validation set on chromosome %d ...' %
            (len(sst_dict['SNP']), chromosome)
        )
    return sst_dicts


def parse_sumstats(ref_dict, vld_dict, sst_file, n_subj):
    print('... parse sumstats file: %s ...' % sst_file)

    ATGC = ['A', 'T', 'G', 'C']
    sst_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(sst_file) as ff:
        header = next(ff)
        for line in ff:
            ll = (line.strip()).split()
            if ll[1] in ATGC and ll[2] in ATGC:
                sst_dict['SNP'].append(ll[0])
                sst_dict['A1'].append(ll[1])
                sst_dict['A2'].append(ll[2])

    print('... %d SNPs read from %s ...' % (len(sst_dict['SNP']), sst_file))


    mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    vld_snp = set(zip(vld_dict['SNP'], vld_dict['A1'], vld_dict['A2']))

    ref_snp = set(zip(ref_dict['SNP'], ref_dict['A1'], ref_dict['A2'])) | set(zip(ref_dict['SNP'], ref_dict['A2'], ref_dict['A1'])) | \
              set(zip(ref_dict['SNP'], [mapping[aa] for aa in ref_dict['A1']], [mapping[aa] for aa in ref_dict['A2']])) | \
              set(zip(ref_dict['SNP'], [mapping[aa] for aa in ref_dict['A2']], [mapping[aa] for aa in ref_dict['A1']]))
    
    sst_snp = set(zip(sst_dict['SNP'], sst_dict['A1'], sst_dict['A2'])) | set(zip(sst_dict['SNP'], sst_dict['A2'], sst_dict['A1'])) | \
              set(zip(sst_dict['SNP'], [mapping[aa] for aa in sst_dict['A1']], [mapping[aa] for aa in sst_dict['A2']])) | \
              set(zip(sst_dict['SNP'], [mapping[aa] for aa in sst_dict['A2']], [mapping[aa] for aa in sst_dict['A1']]))

    comm_snp = vld_snp & ref_snp & sst_snp

    print('... %d common SNPs in the reference, sumstats, and validation set ...' % len(comm_snp))


    n_sqrt = np.sqrt(n_subj)
    sst_eff = {}
    with open(sst_file) as ff:
        header = (next(ff).strip()).split()
        header = [col.upper() for col in header]
        for line in ff:
            ll = (line.strip()).split()
            snp = ll[0]; a1 = ll[1]; a2 = ll[2]
            if a1 not in ATGC or a2 not in ATGC:
                continue
            if (snp, a1, a2) in comm_snp or (snp, mapping[a1], mapping[a2]) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = np.log(float(ll[3]))

                if 'SE' in header:
                    se = float(ll[4])
                    beta_std = beta/se/n_sqrt
                elif 'P' in header:
                    p = max(float(ll[4]), 1e-323)
                    beta_std = np.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt

                sst_eff.update({snp: beta_std})

            elif (snp, a2, a1) in comm_snp or (snp, mapping[a2], mapping[a1]) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = np.log(float(ll[3]))

                if 'SE' in header:
                    se = float(ll[4])
                    beta_std = -1*beta/se/n_sqrt
                elif 'P' in header:
                    p = max(float(ll[4]), 1e-323)
                    beta_std = -1*np.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt

                sst_eff.update({snp: beta_std})


    sst_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[], 'BETA':[], 'FLP':[]}
    for (ii, snp) in enumerate(ref_dict['SNP']):
        if snp in sst_eff:
            sst_dict['SNP'].append(snp)
            sst_dict['CHR'].append(ref_dict['CHR'][ii])
            sst_dict['BP'].append(ref_dict['BP'][ii])
            sst_dict['BETA'].append(sst_eff[snp])

            a1 = ref_dict['A1'][ii]; a2 = ref_dict['A2'][ii]
            if (snp, a1, a2) in comm_snp:
                sst_dict['A1'].append(a1)
                sst_dict['A2'].append(a2)
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, a2, a1) in comm_snp:
                sst_dict['A1'].append(a2)
                sst_dict['A2'].append(a1)
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)
            elif (snp, mapping[a1], mapping[a2]) in comm_snp:
                sst_dict['A1'].append(mapping[a1])
                sst_dict['A2'].append(mapping[a2])
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, mapping[a2], mapping[a1]) in comm_snp:
                sst_dict['A1'].append(mapping[a2])
                sst_dict['A2'].append(mapping[a1])
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)

    return sst_dict


def _ldblk_filename(ldblk_dir, chrom):
    if '1kg' in os.path.basename(ldblk_dir):
        prefix = 'ldblk_1kg_chr'
    elif 'ukbb' in os.path.basename(ldblk_dir):
        prefix = 'ldblk_ukbb_chr'
    else:
        raise ValueError(
            'LD reference directory name must contain either 1kg or ukbb'
        )
    return os.path.join(ldblk_dir, prefix + str(chrom) + '.hdf5')


def _ld_cache_key(chr_name, sst_dict):
    if len(sst_dict['SNP']) != len(sst_dict['FLP']):
        raise ValueError('SNP and allele-flip arrays must have equal length')

    source = os.stat(chr_name)
    digest = hashlib.sha256()
    digest.update(b'PRScs projected LD cache')
    digest.update(str(_LD_CACHE_VERSION).encode('ASCII'))
    digest.update(os.path.realpath(chr_name).encode('UTF-8'))
    digest.update(str(source.st_size).encode('ASCII'))
    digest.update(str(source.st_mtime_ns).encode('ASCII'))
    for snp, flip in zip(sst_dict['SNP'], sst_dict['FLP']):
        encoded = snp.encode('UTF-8')
        digest.update(len(encoded).to_bytes(4, byteorder='little'))
        digest.update(encoded)
        digest.update(int(flip).to_bytes(2, byteorder='little', signed=True))
    return digest.hexdigest()


def _read_ld_cache(cache_file, cache_key):
    try:
        with h5py.File(cache_file, 'r') as cached:
            if (int(cached.attrs['version']) != _LD_CACHE_VERSION or
                    cached.attrs['key'] != cache_key):
                return None

            blk_size = np.asarray(cached['blk_size'], dtype=np.int64).tolist()
            blocks = cached['blocks']
            ld_blk = [
                np.asfortranarray(np.array(blocks[str(index)]['ld']))
                for index in range(len(blk_size))
            ]
            return ld_blk, blk_size
    except (KeyError, OSError, TypeError, ValueError) as error:
        print('... ignore unusable LD cache %s: %s ...' %
              (cache_file, error))
        return None


def _write_ld_cache(cache_file, cache_key, ld_blk, blk_size):
    temporary = '%s.tmp-%d' % (cache_file, os.getpid())
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with h5py.File(temporary, 'w') as cached:
            cached.attrs['version'] = _LD_CACHE_VERSION
            cached.attrs['key'] = cache_key
            cached.create_dataset(
                'blk_size', data=np.asarray(blk_size, dtype=np.int64)
            )
            blocks = cached.create_group('blocks')
            for index, ld in enumerate(ld_blk):
                block = blocks.create_group(str(index))
                block.create_dataset('ld', data=ld)
        os.replace(temporary, cache_file)
        return True
    except (OSError, RuntimeError, ValueError) as error:
        print('... unable to write LD cache %s: %s ...' %
              (cache_file, error))
        try:
            os.remove(temporary)
        except OSError:
            pass
        return False


def parse_ldblk(ldblk_dir, sst_dict, chrom, report_timing=False,
                cache_dir=None, project_sumstats=False,
                projection_fraction=0.2,
                projection_min_eigenvalue=0.01):
    print('... parse reference LD on chromosome %d ...' % chrom)
    total_started = time.perf_counter()
    chr_name = _ldblk_filename(ldblk_dir, chrom)

    if isinstance(project_sumstats, str):
        project_sumstats = project_sumstats.upper() == 'TRUE'
    else:
        project_sumstats = bool(project_sumstats)
    if project_sumstats:
        if cache_dir is not None:
            raise ValueError(
                'projected summary statistics cannot use the projected LD '
                'cache because it does not retain the projection basis'
            )
        if len(sst_dict.get('BETA', [])) != len(sst_dict['SNP']):
            raise ValueError(
                'projected summary statistics require one marginal effect '
                'per selected SNP'
            )

    cache_file = None
    cache_key = None
    if cache_dir is not None:
        cache_key = _ld_cache_key(chr_name, sst_dict)
        cache_file = os.path.join(
            os.path.expanduser(cache_dir),
            'chr%d-%s.hdf5' % (chrom, cache_key),
        )
        cache_started = time.perf_counter()
        cached = None
        if os.path.isfile(cache_file):
            cached = _read_ld_cache(cache_file, cache_key)
        if cached is not None:
            cache_elapsed = time.perf_counter() - cache_started
            print('... loaded projected LD cache: %s ...' % cache_file)
            if report_timing:
                print(
                    '[LOAD chr%d] LD cache HDF5 %.3fs, total %.3fs' %
                    (chrom, cache_elapsed,
                     time.perf_counter() - total_started)
                )
            return cached

    with h5py.File(chr_name, 'r') as hdf_chr:
        n_blk = len(hdf_chr)
        ld_blk = [
            np.array(hdf_chr['blk_'+str(blk)]['ldblk'])
            for blk in range(1, n_blk+1)
        ]
        snp_blk = [
            [bb.decode("UTF-8") for bb in
             list(hdf_chr['blk_'+str(blk)]['snplist'])]
            for blk in range(1, n_blk+1)
        ]
    read_elapsed = time.perf_counter() - total_started

    blk_size = []
    selected_snps = set(sst_dict['SNP'])
    projection_elapsed = 0.0
    mm = 0
    for blk in range(n_blk):
        idx = [
            ii for ii, snp in enumerate(snp_blk[blk])
            if snp in selected_snps
        ]
        blk_size.append(len(idx))
        if idx != []:
            idx_blk = range(mm,mm+len(idx))
            flip = [sst_dict['FLP'][jj] for jj in idx_blk]
            ld_blk[blk] = ld_blk[blk][np.ix_(idx,idx)]*np.outer(flip,flip)

            projection_started = time.perf_counter()
            if project_sumstats:
                beta_slice = slice(mm, mm + len(idx))
                ld_blk[blk], projected_beta = \
                    _truncate_ld_and_project_sumstats(
                        ld_blk[blk],
                        sst_dict['BETA'][beta_slice],
                        fraction=projection_fraction,
                        min_eigenvalue=projection_min_eigenvalue,
                    )
                sst_dict['BETA'][beta_slice] = projected_beta.tolist()
            else:
                ld_blk[blk] = _project_ld_psd(ld_blk[blk])
            projection_elapsed += time.perf_counter() - projection_started

            mm += len(idx)
        else:
            ld_blk[blk] = np.array([])

    compute_elapsed = time.perf_counter() - total_started
    if report_timing:
        filtering_elapsed = max(
            compute_elapsed - read_elapsed - projection_elapsed, 0.0
        )
        print(
            '[LOAD chr%d] LD HDF5 %.3fs, filtering %.3fs, '
            'PSD projection %.3fs, total %.3fs' %
            (chrom, read_elapsed, filtering_elapsed,
             projection_elapsed, compute_elapsed)
        )

    if cache_file is not None:
        cache_started = time.perf_counter()
        wrote_cache = _write_ld_cache(
            cache_file, cache_key, ld_blk, blk_size
        )
        if wrote_cache:
            print('... wrote projected LD cache: %s ...' % cache_file)
            if report_timing:
                print('[LOAD chr%d] LD cache write %.3fs' % (
                    chrom, time.perf_counter() - cache_started
                ))

    return ld_blk, blk_size
