#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import numpy as np
from beta_backend import make_beta_backend
from psi_backend import make_psi_backend


def _chromosome_partitions(chrom, chromosome_slices, p):
    """Return validated ``(chromosome, start, stop)`` output partitions."""
    if chromosome_slices is None:
        return [(int(chrom), 0, p)]

    partitions = []
    expected_start = 0
    for chromosome, start, stop in chromosome_slices:
        chromosome = int(chromosome)
        start = int(start)
        stop = int(stop)
        if start != expected_start or stop < start or stop > p:
            raise ValueError(
                'chromosome slices must be contiguous and cover every SNP'
            )
        partitions.append((chromosome, start, stop))
        expected_start = stop

    if not partitions or expected_start != p:
        raise ValueError(
            'chromosome slices must be contiguous and cover every SNP'
        )
    return partitions


def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin,
         thin, chrom, out_dir, beta_std, write_psi, write_pst, seed,
         chromosome_slices=None):
    print('... MCMC ...')

    # seed
    if seed is not None:
        np.random.seed(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    n_pst = int((n_iter-n_burnin)/thin)
    p = len(sst_dict['SNP'])
    joint_chromosomes = chromosome_slices is not None
    partitions = _chromosome_partitions(chrom, chromosome_slices, p)

    if joint_chromosomes:
        print(
            '... joint chromosome chain: %d chromosomes, %d SNPs, '
            '%d active LD blocks ...' % (
                len(partitions), p, sum(size > 0 for size in blk_size)
            )
        )

    # initialization
    beta = np.zeros((p,1))
    psi = np.ones((p,1))
    sigma = 1.0
    
    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    if write_pst == 'TRUE':
        beta_pst = np.zeros((p,n_pst))

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0

    beta_sampler = make_beta_backend(
        'cpu', ld_blk, blk_size, beta_mrg, n
    )
    print('... beta backend: %s ...' % beta_sampler.describe())
    psi_sampler = make_psi_backend('cpu', p, seed=seed)
    print('... psi backend: %s ...' % psi_sampler.describe())

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        if itr % 100 == 0:
            print('--- iter-' + str(itr) + ' ---')

        beta, quad = beta_sampler.sample(psi, sigma)

        s1 = float((beta * beta_mrg).sum())
        s2 = float((beta**2 / psi).sum())
        e1 = float(n/2.0*(1.0 - 2.0*s1 + quad))
        e2 = float(n/2.0*s2)
        err = max(e1, e2)

        # force sigma to be a Python float (not a 0-d array)
        sigma = float(1.0/np.random.gamma((n+p)/2.0, 1.0/err))

        delta = np.random.gamma(a+b, 1.0/(psi+phi))

        psi_sampler.sample(
            psi[:, 0],
            float(a - 0.5),
            delta[:, 0],
            beta[:, 0],
            float(sigma),
            int(n),
        )
        
        psi[psi>1] = 1.0

        if phi_updt == True:
            w = np.random.gamma(1.0, 1.0/(phi+1.0))
            phi = np.random.gamma(p*b+0.5, 1.0/(sum(delta)+w))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

    # convert standardized beta to per-allele beta
    if beta_std == 'FALSE':
        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= np.sqrt(2.0*maf*(1.0-maf))


    # Preserve the conventional per-chromosome output files even when the
    # selected chromosomes were sampled together in one chain.
    for chromosome, start, stop in partitions:
        if phi_updt == True:
            eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (
                a, b, chromosome
            )
        else:
            eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (
                a, b, phi, chromosome
            )

        with open(eff_file, 'w') as ff:
            if write_pst == 'TRUE':
                for snp, bp, a1, a2, beta in zip(
                        sst_dict['SNP'][start:stop],
                        sst_dict['BP'][start:stop],
                        sst_dict['A1'][start:stop],
                        sst_dict['A2'][start:stop],
                        beta_pst[start:stop]):
                    ff.write(
                        ('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') %
                        (chromosome, snp, bp, a1, a2, *beta)
                    )
            else:
                for snp, bp, a1, a2, beta in zip(
                        sst_dict['SNP'][start:stop],
                        sst_dict['BP'][start:stop],
                        sst_dict['A1'][start:stop],
                        sst_dict['A2'][start:stop],
                        beta_est[start:stop]):
                    ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' %
                             (chromosome, snp, bp, a1, a2, beta.item()))

        if write_psi == 'TRUE':
            if phi_updt == True:
                psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (
                    a, b, chromosome
                )
            else:
                psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (
                    a, b, phi, chromosome
                )

            with open(psi_file, 'w') as ff:
                for snp, psi_value in zip(
                        sst_dict['SNP'][start:stop], psi_est[start:stop]):
                    ff.write('%s\t%.6e\n' % (snp, psi_value.item()))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')
