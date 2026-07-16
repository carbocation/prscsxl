#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import time

import numpy as np
from beta_backend import make_beta_backend
from psi_backend import make_psi_backend


_SIGMA_FLOOR_RELATIVE_TOLERANCE = 1e-10


def _sigma_floor_relative_deficit(e1, e2, iteration):
    """Validate sigma statistics and quantify residual-floor activation."""
    if not np.isfinite(e1) or not np.isfinite(e2):
        raise FloatingPointError(
            'non-finite sigma sufficient statistic at iteration %d: '
            'e1=%r, e2=%r' % (iteration, e1, e2)
        )
    if e2 <= e1:
        return None
    return (e2 - e1) / max(abs(e1), abs(e2), 1.0)


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


def _profile_label(partitions, joint_chromosomes):
    if not joint_chromosomes:
        return 'chr%d' % partitions[0][0]

    chromosomes = [partition[0] for partition in partitions]
    ranges = []
    start = chromosomes[0]
    end = start
    for chromosome in chromosomes[1:]:
        if chromosome == end + 1:
            end = chromosome
            continue
        ranges.append(str(start) if start == end else '%d-%d' % (start, end))
        start = chromosome
        end = chromosome
    ranges.append(str(start) if start == end else '%d-%d' % (start, end))
    return 'joint chr%s' % ','.join(ranges)


def _posterior_iterations(n_iter, n_burnin, thin):
    """Return iterations retained by thinning after burn-in."""
    n_iter = int(n_iter)
    n_burnin = int(n_burnin)
    thin = int(thin)

    if n_iter < 1:
        raise ValueError('n_iter must be at least 1')
    if not 0 <= n_burnin < n_iter:
        raise ValueError('n_burnin must be in [0, n_iter)')
    if thin < 1:
        raise ValueError('thin must be at least 1')

    retained = range(n_burnin + thin, n_iter + 1, thin)
    if not retained:
        raise ValueError(
            'sampling schedule retains no posterior draws; increase n_iter '
            'or decrease n_burnin or thin'
        )
    return retained


def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin,
         thin, chrom, out_dir, beta_std, write_psi, write_pst, seed,
         chromosome_slices=None, backend='cpu', cuda_device=0,
         cuda_bucket_size=32, cuda_streams=4, profile='FALSE',
         cuda_gig_max_rounds=1000):
    print('... MCMC ...')

    # seed
    if seed is not None:
        np.random.seed(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    posterior_iterations = _posterior_iterations(
        n_iter, n_burnin, thin
    )
    n_pst = len(posterior_iterations)
    p = len(sst_dict['SNP'])
    joint_chromosomes = chromosome_slices is not None
    partitions = _chromosome_partitions(chrom, chromosome_slices, p)
    profile_label = _profile_label(partitions, joint_chromosomes)

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
        backend, ld_blk, blk_size, beta_mrg, n,
        seed=seed, cuda_device=cuda_device,
        cuda_bucket_size=cuda_bucket_size,
        cuda_streams=cuda_streams,
        profile=profile,
    )
    print('... beta backend: %s ...' % beta_sampler.describe())
    psi_sampler = make_psi_backend(
        backend, p, seed=seed, cuda_device=cuda_device,
        cuda_gig_max_rounds=cuda_gig_max_rounds,
    )
    print('... psi backend: %s ...' % psi_sampler.describe())
    profile = str(profile).upper() == 'TRUE'
    profile_beta = 0.0
    profile_psi = 0.0
    profile_total = 0.0
    profile_iterations = 0
    sigma_floor_count = 0
    sigma_floor_material_count = 0
    sigma_floor_first_iteration = None
    sigma_floor_worst_relative = 0.0

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        iteration_start = time.perf_counter()
        if itr % 100 == 0:
            print('--- iter-' + str(itr) + ' ---')

        beta_start = time.perf_counter()
        beta, quad = beta_sampler.sample(psi, sigma)
        beta_elapsed = time.perf_counter() - beta_start

        s1 = float((beta * beta_mrg).sum())
        s2 = float((beta**2 / psi).sum())
        e1 = float(n/2.0*(1.0 - 2.0*s1 + quad))
        e2 = float(n/2.0*s2)
        relative_deficit = _sigma_floor_relative_deficit(e1, e2, itr)
        if relative_deficit is not None:
            sigma_floor_count += 1
            if sigma_floor_first_iteration is None:
                sigma_floor_first_iteration = itr
            sigma_floor_worst_relative = max(
                sigma_floor_worst_relative, relative_deficit
            )
            if relative_deficit > _SIGMA_FLOOR_RELATIVE_TOLERANCE:
                sigma_floor_material_count += 1
        err = max(e1, e2)
        if err <= 0.0:
            raise FloatingPointError(
                'non-positive sigma rate at iteration %d: %r' % (itr, err)
            )

        # force sigma to be a Python float (not a 0-d array)
        sigma = float(1.0/np.random.gamma((n+p)/2.0, 1.0/err))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise FloatingPointError(
                'invalid sigma draw at iteration %d: %r' % (itr, sigma)
            )

        if hasattr(psi_sampler, 'sample_joint'):
            psi_start = time.perf_counter()
            delta_sum = psi_sampler.sample_joint(
                psi[:, 0],
                float(a - 0.5),
                float(a + b),
                psi[:, 0],
                float(phi),
                beta[:, 0],
                float(sigma),
                int(n),
                need_delta_sum=phi_updt,
            )
        else:
            delta = np.random.gamma(a+b, 1.0/(psi+phi))
            psi_start = time.perf_counter()
            psi_sampler.sample(
                psi[:, 0],
                float(a - 0.5),
                delta[:, 0],
                beta[:, 0],
                float(sigma),
                int(n),
            )
            delta_sum = float(delta.sum()) if phi_updt else None
        psi_elapsed = time.perf_counter() - psi_start
        
        psi[psi>1] = 1.0

        if phi_updt == True:
            w = np.random.gamma(1.0, 1.0/(phi+1.0))
            phi = np.random.gamma(p*b+0.5, 1.0/(delta_sum+w))

        # posterior
        if itr in posterior_iterations:
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

        iteration_elapsed = time.perf_counter() - iteration_start
        if profile:
            if itr > 1:
                profile_beta += beta_elapsed
                profile_psi += psi_elapsed
                profile_total += iteration_elapsed
                profile_iterations += 1
            if itr == 1:
                print(
                    '[PROFILE %s] iter 1 warm-up: beta %.4fs, psi %.4fs, '
                    'total %.4fs' %
                    (profile_label, beta_elapsed, psi_elapsed,
                     iteration_elapsed)
                )
            elif itr % 10 == 0 or itr == n_iter:
                other = profile_total - profile_beta - profile_psi
                print(
                    '[PROFILE %s] steady-state mean over %d iter: beta '
                    '%.4fs, psi %.4fs, other %.4fs, total %.4fs' %
                    (profile_label, profile_iterations,
                     profile_beta/profile_iterations,
                     profile_psi/profile_iterations,
                     other/profile_iterations,
                     profile_total/profile_iterations)
                )

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

    if sigma_floor_count:
        print(
            '... WARNING: sigma residual safeguard: %d/%d activations '
            '(%d material; first iteration %d; worst relative deficit '
            '%.3e) ...' % (
                sigma_floor_count, n_iter, sigma_floor_material_count,
                sigma_floor_first_iteration, sigma_floor_worst_relative,
            )
        )
    else:
        print(
            '... sigma residual safeguard: 0/%d activations ...' % n_iter
        )

    if profile and hasattr(beta_sampler, 'profile_summary'):
        print('[PROFILE %s] %s' %
              (profile_label, beta_sampler.profile_summary()))
    if profile and hasattr(psi_sampler, 'profile_summary'):
        print('[PROFILE %s] %s' %
              (profile_label, psi_sampler.profile_summary()))

    print('... Done ...')
