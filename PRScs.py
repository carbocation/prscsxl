#!/usr/bin/env python3

"""
PRS-CS: a polygenic prediction method that infers posterior SNP effect sizes under continuous shrinkage (CS) priors
using GWAS summary statistics and an external LD reference panel.

Reference: T Ge, CY Chen, Y Ni, YCA Feng, JW Smoller. Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors.
           Nature Communications, 10:1776, 2019.


Usage:
python PRScs.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --out_dir=OUTPUT_DIR
                [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR
                 --chrom=CHROM --joint_chromosomes=TRUE|FALSE --ld_cache_dir=PATH
                 --write_psi=WRITE_PSI --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED]

"""


import os
import sys
import getopt
import time

import parse_genet
import mcmc_gtb


def parse_param():
    long_opts_list = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=',
                      'n_iter=', 'n_burnin=', 'thin=', 'out_dir=', 'chrom=', 'joint_chromosomes=', 'ld_cache_dir=', 'beta_std=', 'write_psi=', 'write_pst=', 'seed=', 'help']

    param_dict = {'ref_dir': None, 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': None, 'n_gwas': None,
                  'n_iter': 1000, 'n_burnin': 500, 'thin': 5, 'out_dir': None, 'chrom': range(1,23),
                  'joint_chromosomes': 'FALSE',
                  'ld_cache_dir': None,
                  'beta_std': 'FALSE', 'write_psi': 'FALSE', 'write_pst': 'FALSE', 'seed': None}

    print('\n')

    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "h", long_opts_list)          
        except:
            print('Option not recognized.')
            print('Use --help for usage information.\n')
            sys.exit(2)

        for opt, arg in opts:
            if opt == "-h" or opt == "--help":
                print(__doc__)
                sys.exit(0)
            elif opt == "--ref_dir": param_dict['ref_dir'] = arg
            elif opt == "--bim_prefix": param_dict['bim_prefix'] = arg
            elif opt == "--sst_file": param_dict['sst_file'] = arg
            elif opt == "--a": param_dict['a'] = float(arg)
            elif opt == "--b": param_dict['b'] = float(arg)
            elif opt == "--phi": param_dict['phi'] = float(arg)
            elif opt == "--n_gwas": param_dict['n_gwas'] = int(arg)
            elif opt == "--n_iter": param_dict['n_iter'] = int(arg)
            elif opt == "--n_burnin": param_dict['n_burnin'] = int(arg)
            elif opt == "--thin": param_dict['thin'] = int(arg)
            elif opt == "--out_dir": param_dict['out_dir'] = arg
            elif opt == "--chrom": param_dict['chrom'] = arg.split(',')
            elif opt == "--joint_chromosomes": param_dict['joint_chromosomes'] = arg.upper()
            elif opt == "--ld_cache_dir": param_dict['ld_cache_dir'] = arg
            elif opt == "--beta_std": param_dict['beta_std'] = arg.upper()
            elif opt == "--write_psi": param_dict['write_psi'] = arg.upper()
            elif opt == "--write_pst": param_dict['write_pst'] = arg.upper()
            elif opt == "--seed": param_dict['seed'] = int(arg)
    else:
        print(__doc__)
        sys.exit(0)

    if param_dict['ref_dir'] == None:
        print('* Please specify the directory to the reference panel using --ref_dir\n')
        sys.exit(2)
    elif param_dict['bim_prefix'] == None:
        print('* Please specify the directory and prefix of the bim file for the target dataset using --bim_prefix\n')
        sys.exit(2)
    elif param_dict['sst_file'] == None:
        print('* Please specify the summary statistics file using --sst_file\n')
        sys.exit(2)
    elif param_dict['n_gwas'] == None:
        print('* Please specify the sample size of the GWAS using --n_gwas\n')
        sys.exit(2)
    elif param_dict['out_dir'] == None:
        print('* Please specify the output directory using --out_dir\n')
        sys.exit(2)
    elif param_dict['joint_chromosomes'] not in ('TRUE', 'FALSE'):
        print('* --joint_chromosomes must be True or False\n')
        sys.exit(2)

    for key in param_dict:
        print('--%s=%s' % (key, param_dict[key]))

    print('\n')
    return param_dict


def _reference_file(param_dict):
    if '1kg' in os.path.basename(param_dict['ref_dir']):
        return param_dict['ref_dir'] + '/snpinfo_1kg_hm3'
    if 'ukbb' in os.path.basename(param_dict['ref_dir']):
        return param_dict['ref_dir'] + '/snpinfo_ukbb_hm3'
    raise ValueError(
        'reference directory name must contain either 1kg or ukbb'
    )


def _load_chromosome(param_dict, chrom):
    ref_dict = parse_genet.parse_ref(_reference_file(param_dict), chrom)
    vld_dict = parse_genet.parse_bim(param_dict['bim_prefix'], chrom)
    sst_dict = parse_genet.parse_sumstats(
        ref_dict, vld_dict, param_dict['sst_file'], param_dict['n_gwas']
    )
    ld_blk, blk_size = parse_genet.parse_ldblk(
        param_dict['ref_dir'], sst_dict, chrom,
        cache_dir=param_dict['ld_cache_dir'],
    )
    return {
        'chrom': chrom,
        'sst_dict': sst_dict,
        'ld_blk': ld_blk,
        'blk_size': blk_size,
    }


def _load_joint_chromosomes(param_dict, chromosomes):
    """Load selected chromosomes without repeatedly scanning text inputs."""
    total_started = time.perf_counter()
    text_started = time.perf_counter()
    ref_dicts = parse_genet.parse_ref_chromosomes(
        _reference_file(param_dict), chromosomes
    )
    vld_dict = parse_genet.parse_bim_chromosomes(
        param_dict['bim_prefix'], chromosomes
    )
    sst_dicts = parse_genet.parse_sumstats_chromosomes(
        ref_dicts, vld_dict, param_dict['sst_file'], param_dict['n_gwas']
    )
    print('[LOAD joint] text inputs %.3fs' %
          (time.perf_counter() - text_started))

    chromosome_inputs = []
    for chromosome in chromosomes:
        print('##### load chromosome %d LD #####' % chromosome)
        sst_dict = sst_dicts[chromosome]
        ld_blk, blk_size = parse_genet.parse_ldblk(
            param_dict['ref_dir'], sst_dict, chromosome,
            report_timing=True,
            cache_dir=param_dict['ld_cache_dir'],
        )
        chromosome_inputs.append({
            'chrom': chromosome,
            'sst_dict': sst_dict,
            'ld_blk': ld_blk,
            'blk_size': blk_size,
        })

    print('[LOAD joint] all inputs %.3fs' %
          (time.perf_counter() - total_started))
    return chromosome_inputs


def _combine_chromosomes(chromosome_inputs):
    if not chromosome_inputs:
        raise ValueError('at least one chromosome must be selected')

    sst_dict = {key: [] for key in chromosome_inputs[0]['sst_dict']}
    ld_blk = []
    blk_size = []
    chromosome_slices = []
    seen_chromosomes = set()
    start = 0

    for chromosome_input in chromosome_inputs:
        chromosome = int(chromosome_input['chrom'])
        if chromosome in seen_chromosomes:
            raise ValueError(
                'joint chromosome selection contains chromosome %d twice' %
                chromosome
            )
        seen_chromosomes.add(chromosome)

        chromosome_sst = chromosome_input['sst_dict']
        if set(chromosome_sst) != set(sst_dict):
            raise ValueError('all chromosome summary dictionaries must match')
        if not chromosome_sst['SNP']:
            raise ValueError(
                'no common SNPs found for selected chromosome %d; '
                'joint chromosome sampling requires data for every '
                'selected chromosome' % chromosome
            )
        if len(chromosome_input['ld_blk']) != len(
                chromosome_input['blk_size']):
            raise ValueError(
                'LD blocks and sizes must match on chromosome %d' % chromosome
            )
        if sum(chromosome_input['blk_size']) != len(chromosome_sst['SNP']):
            raise ValueError(
                'LD blocks do not cover every SNP on chromosome %d' %
                chromosome
            )

        for key in sst_dict:
            sst_dict[key].extend(chromosome_sst[key])
        ld_blk.extend(chromosome_input['ld_blk'])
        blk_size.extend(chromosome_input['blk_size'])

        stop = start + len(chromosome_sst['SNP'])
        chromosome_slices.append((chromosome, start, stop))
        start = stop

    return {
        'sst_dict': sst_dict,
        'ld_blk': ld_blk,
        'blk_size': blk_size,
        'chromosome_slices': chromosome_slices,
    }


def _run_mcmc(param_dict, chromosome_input, chrom,
              chromosome_slices=None):
    mcmc_gtb.mcmc(
        param_dict['a'], param_dict['b'], param_dict['phi'],
        chromosome_input['sst_dict'], param_dict['n_gwas'],
        chromosome_input['ld_blk'], chromosome_input['blk_size'],
        param_dict['n_iter'], param_dict['n_burnin'], param_dict['thin'],
        chrom, param_dict['out_dir'], param_dict['beta_std'],
        param_dict['write_psi'], param_dict['write_pst'],
        param_dict['seed'], chromosome_slices=chromosome_slices,
    )


def main():
    param_dict = parse_param()
    chromosomes = [int(chrom) for chrom in param_dict['chrom']]

    if param_dict['joint_chromosomes'] == 'TRUE':
        print(
            '##### jointly process chromosomes %s #####' %
            ','.join(str(chrom) for chrom in chromosomes)
        )
        joint_input = _combine_chromosomes(
            _load_joint_chromosomes(param_dict, chromosomes)
        )
        _run_mcmc(
            param_dict, joint_input, chromosomes,
            chromosome_slices=joint_input['chromosome_slices'],
        )
        print('\n')
        return

    for chrom in chromosomes:
        print('##### process chromosome %d #####' % chrom)
        chromosome_input = _load_chromosome(param_dict, chrom)
        _run_mcmc(param_dict, chromosome_input, chrom)
        print('\n')


if __name__ == '__main__':
    main()
