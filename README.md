# PRS-CS

**PRS-CS** is a Python based command line tool that infers posterior SNP effect sizes under continuous shrinkage (CS) priors
using GWAS summary statistics and an external LD reference panel.

- Details of the development and evaluation of PRS-CS are described in: \
  T Ge, CY Chen, Y Ni, YCA Feng, JW Smoller. Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors. *Nature Communications*, 10:1776, 2019.

- An extension of PRS-CS to PRS-CSx for cross-population polygenic prediction is available at https://github.com/getian107/PRScsx and described in: \
  Y Ruan, YF Lin, YCA Feng, CY Chen, M Lam, Z Guo, Stanley Global Asia Initiatives, L He, A Sawa, AR Martin, S Qin, H Huang, T Ge. Improving polygenic prediction in ancestrally diverse populations. *Nature Genetics*, 54:573-580, 2022.

- A review of the methods and best practices for cross-ancestry polygenic prediction is available at: \
  L Kachuri, N Chatterjee, J Hirbo, DJ Schaid, I Martin, IJ Kullo, EE Kenny, B Pasaniuc, JS Witte, T Ge. Principles and methods for transferring polygenic risk scores across global populations. *Nature Reviews Genetics*, 25:8-25, 2024.


## Version History

**May 14, 2024**: Replaced some scipy functions with numpy due to changes in the latest scipy version.

**Apr 9, 2024**: Allowed for the output of all posterior samples, which can be used to estimate the uncertainty of individualized PRS.

🔴
**Aug 10, 2023**: Added BETA/OR + SE as a new input format (see the format of GWAS summary statistics below), which is now the recommended input data. When using BETA/OR + P as the input, p-values smaller than 1e-323 are truncated, which may reduce the prediction accuracy for traits that have highly significant loci.

**Aug 10, 2023**: Allowed for the output of variant-specific shrinkage estimates.

**Nov 3, 2022**: Import random module from numpy instead of scipy.

**Jun 4, 2021**: Expanded reference panels to five populations.

**May 26, 2021**: Added suggestions for limiting the number of threads in scipy when running PRS-CS (see Computational Efficiency section below).

**Apr 6, 2021**: Added projection of the LD matrix to its nearest non-negative definite matrix.

**Mar 4, 2021**: LD reference panels constructed using the UK Biobank data are now available. 

**Jan 4, 2021**: Improved the accuracy and robustness of random sampling from the generalized inverse Gaussian distribution. Prediction accuracy will probably slightly improve over previous versions.

**Sept 10, 2020**: Fixed a bug in strand flip when there are non-ATGC alleles (e.g., indels) in the GWAS summary statistics. Previous versions erroneously remove variants that can be matched across GWAS summary statistics, the reference panel and the validation bim file via strand flip, which reduces the number of SNPs used in prediction and may slightly affect prediction accuracy. 

**Apr 24, 2020**: Accounted for a rare ZeroDivisionError in MCMC sampling.

**Apr 20, 2020**: Added non-ATGC allele check.

**Apr 11, 2020**: Added strand flip check.

**Mar 25, 2020**: Minor changes to make the software Python 2 and 3 compatible.

**Oct 20, 2019**: Added `--seed`, which can be used to seed the random number generator using a non-negative integer.

**Jun 6, 2019**: Fixed a bug in `--beta_std`. If you explicitly specified `--beta_std=False`, the output was actually standardized beta (in contrast to desired per-allele beta) and we recommend rerunning the analysis. If you left `--beta_std` as default or used `--beta_std=True`, the results were not affected.


## Getting Started

- Clone this repository using the following git command:
   
    `git clone https://github.com/getian107/PRScs.git`

    Alternatively, download the source files from the github website (`https://github.com/getian107/PRScs`)

- Download the LD reference panels and extract files:

    LD reference panels constructed using the 1000 Genomes Project phase 3 samples:
    
     [AFR reference](https://www.dropbox.com/s/mq94h1q9uuhun1h/ldblk_1kg_afr.tar.gz?dl=0 "AFR reference") (~4.44G);
     `tar -zxvf ldblk_1kg_afr.tar.gz`
     
     [AMR reference](https://www.dropbox.com/s/uv5ydr4uv528lca/ldblk_1kg_amr.tar.gz?dl=0 "AMR reference") (~3.84G);
     `tar -zxvf ldblk_1kg_amr.tar.gz`
        
     [EAS reference](https://www.dropbox.com/s/7ek4lwwf2b7f749/ldblk_1kg_eas.tar.gz?dl=0 "EAS reference") (~4.33G);
     `tar -zxvf ldblk_1kg_eas.tar.gz`
        
     [EUR reference](https://www.dropbox.com/s/mt6var0z96vb6fv/ldblk_1kg_eur.tar.gz?dl=0 "EUR reference") (~4.56G);
     `tar -zxvf ldblk_1kg_eur.tar.gz`
     
     [SAS reference](https://www.dropbox.com/s/hsm0qwgyixswdcv/ldblk_1kg_sas.tar.gz?dl=0 "SAS reference") (~5.60G);
     `tar -zxvf ldblk_1kg_sas.tar.gz`
    
    LD reference panels constructed using the UK Biobank data ([Notes](https://www.dropbox.com/s/y3hsc15kwjxwjtd/UKBB_ref.txt?dl=0 "Notes")):
    
     [AFR reference](https://www.dropbox.com/s/dtccsidwlb6pbtv/ldblk_ukbb_afr.tar.gz?dl=0 "AFR reference") (~4.93G);
     `tar -zxvf ldblk_ukbb_afr.tar.gz`
     
     [AMR reference](https://www.dropbox.com/s/y7ruj364buprkl6/ldblk_ukbb_amr.tar.gz?dl=0 "AMR reference") (~4.10G);
     `tar -zxvf ldblk_ukbb_amr.tar.gz`
    
     [EAS reference](https://www.dropbox.com/s/fz0y3tb9kayw8oq/ldblk_ukbb_eas.tar.gz?dl=0 "EAS reference") (~5.80G);
     `tar -zxvf ldblk_ukbb_eas.tar.gz`
    
     [EUR reference](https://www.dropbox.com/s/t9opx2ty6ucrpib/ldblk_ukbb_eur.tar.gz?dl=0 "EUR reference") (~6.25G);
     `tar -zxvf ldblk_ukbb_eur.tar.gz`
    
     [SAS reference](https://www.dropbox.com/s/nto6gdajq8qfhh0/ldblk_ukbb_sas.tar.gz?dl=0 "SAS reference") (~7.37G);
     `tar -zxvf ldblk_ukbb_sas.tar.gz`
     
    For regions that don't have access to Dropbox, reference panels can be downloaded from the
    [alternative download site](https://personal.broadinstitute.org/hhuang//public//PRS-CSx/Reference).

- PRScs requires Python packages **scipy** (https://www.scipy.org/), **h5py** (https://www.h5py.org/) and **numba** (https://numba.pydata.org/) installed.

- The optional CUDA backend requires **CuPy 14.1 or newer**, installed using
  the CuPy package that matches the machine's existing CUDA runtime. It does
  not require changing the NVIDIA driver or CUDA installation.
 
- Once Python and its dependencies have been installed, running

    `./PRScs.py --help` or `./PRScs.py -h`

    will print a list of command-line options.


## Using PRS-CS

`
python PRScs.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --out_dir=OUTPUT_DIR [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR --chrom=CHROM --joint_chromosomes=TRUE|FALSE --ld_cache_dir=PATH --beta_std=BETA_STD --write_psi=WRITE_PSI --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED --backend=cpu|cuda --cuda_device=DEVICE --cuda_bucket_size=SIZE --psi_backend=cpu|cuda --cuda_gig_max_rounds=ROUNDS --ld_diagnostics=TRUE|FALSE --ld_rank_tol=TOL --profile=TRUE|FALSE]
`
 - PATH_TO_REFERENCE (required): Full path (including folder name) to the directory that contains information on the LD reference panel (the snpinfo file and hdf5 files). If the 1000 Genomes reference panel is used, folder name would be `ldblk_1kg_afr`, `ldblk_1kg_amr`, `ldblk_1kg_eas`, `ldblk_1kg_eur` or `ldblk_1kg_sas`; if the UK Biobank reference panel is used, folder name would be `ldblk_ukbb_afr`, `ldblk_ukbb_amr`, `ldblk_ukbb_eas`, `ldblk_ukbb_eur` or `ldblk_ukbb_sas`. Note that the reference panel should match the ancestry of the GWAS sample (not the target sample).

 - VALIDATION_BIM_PREFIX (required): Full path and the prefix of the bim file for the target (validation/testing) dataset. This file is used to provide a list of SNPs that are available in the target dataset.

 - SUM_STATS_FILE (required): Full path and the file name of the GWAS summary statistics. The summary statistics file must include either BETA/OR + SE or BETA/OR + P. When using BETA/OR + SE as the input, the file must have the following format (including the header line):

```
    SNP          A1   A2   BETA      SE
    rs4970383    C    A    -0.0064   0.0090
    rs4475691    C    T    -0.0145   0.0094
    rs13302982   A    G    -0.0232   0.0199
    ...
```
Or:
```
    SNP          A1   A2   OR        SE
    rs4970383    A    C    0.9825    0.0314                 
    rs4475691    T    C    0.9436    0.0319
    rs13302982   A    G    1.1337    0.0543
    ...
```
where SNP is the rs ID, A1 is the effect allele, A2 is the alternative allele, BETA/OR is the effect/odds ratio of the A1 allele, SE is the standard error of the effect. Note that when OR is used, SE corresponds to the standard error of logOR.

When using BETA/OR + P as the input, the file must have the following format (including the header line):

```
    SNP          A1   A2   BETA      P
    rs4970383    C    A    -0.0064   0.4778
    rs4475691    C    T    -0.0145   0.1245
    rs13302982   A    G    -0.0232   0.2429
    ...
```
Or:
```
    SNP          A1   A2   OR        P
    rs4970383    A    C    0.9825    0.5737                 
    rs4475691    T    C    0.9436    0.0691
    rs13302982   A    G    1.1337    0.0209
    ...
```
where SNP is the rs ID, A1 is the effect allele, A2 is the alternative allele, BETA/OR is the effect/odds ratio of the A1 allele, P is the p-value of the effect. Here, a standardized effect size is calculated using the p-value while BETA/OR is only used to determine the direction of an association. Therefore if z-scores or even +1/-1 indicating effect directions are presented in the BETA column, the algorithm should still work properly.

 - GWAS_SAMPLE_SIZE (required): Sample size of the GWAS.

 - OUTPUT_DIR (required): Output directory and output filename prefix of the posterior effect size estimates.

 - PARAM_A (optional): Parameter a in the gamma-gamma prior. Default is 1.

 - PARAM_B (optional): Parameter b in the gamma-gamma prior. Default is 0.5.

 - PARAM_PHI (optional): Global shrinkage parameter phi. If phi is not specified, it will be learnt from the data using a fully Bayesian approach. This usually works well for polygenic traits with large GWAS sample sizes (hundreds of thousands of subjects). For GWAS with limited sample sizes (including most of the current disease GWAS), fixing phi to 1e-2 (for highly polygenic traits) or 1e-4 (for less polygenic traits), or doing a small-scale grid search (e.g., phi=1e-6, 1e-4, 1e-2, 1) to find the optimal phi value in the validation dataset often improves perdictive performance.

 - MCMC_ITERATIONS (optional): Total number of MCMC iterations. Default is 1,000.

 - MCMC_BURNIN (optional): Number of burnin iterations. Default is 500.

 - MCMC_THINNING_FACTOR (optional): Thinning factor of the Markov chain. Default is 5.

 - CHROM (optional): The chromosome(s) on which the model is fitted, separated by comma, e.g., `--chrom=1,3,5`. Parallel computation for the 22 autosomes is recommended. Default is iterating through 22 autosomes (can be time-consuming).

- JOINT_CHROMOSOMES (optional): If True, fit the selected chromosomes in one MCMC chain with shared global parameters. Default is False.

- LD_CACHE_DIR (optional): Directory for persistent filtered and PSD-projected
LD caches. Entries are keyed by the source LD file metadata and the exact
ordered SNP and allele-flip selection, so changing the reference, selected
variants, or allele orientation produces a cache miss. The first matching run
writes a potentially large uncompressed cache; later runs skip the original LD
read, filtering, and eigendecomposition. No cache is read or written by
default.

- BACKEND (optional): `cpu` uses the exact FP64 LAPACK/BLAS implementation
and is the default. `cuda` uses exact FP64 batched Cholesky and triangular
solves through preallocated cuSOLVER and cuBLAS workspaces.

- CUDA_DEVICE (optional): Zero-based CUDA device index. Default is 0.

- CUDA_BUCKET_SIZE (optional): LD block sizes are rounded up to this interval
for CUDA batching. Smaller values reduce padding while larger values may form
denser batches. Default is 32.

- PSI_BACKEND (optional): `cpu` uses the fused Numba GIG sampler and is the
default. `cuda` generates each gamma-distributed `delta` and its dependent GIG
`psi` draw in one CUDA kernel per MCMC iteration using independent Philox
4x32-10 streams. Seeded runs are reproducible within a fixed backend, but
backend random streams differ.

- CUDA_GIG_MAX_ROUNDS (optional): Maximum vector rejection rounds for the
CUDA GIG sampler. The sampler fails explicitly instead of returning incomplete
draws if this bound is reached. Default is 1000.

- LD_DIAGNOSTICS (optional): If True, report LD block sizes, CUDA padding,
numerical rank, and retained condition estimates before sampling. Default is
False.

- LD_RANK_TOL (optional): Relative eigenvalue threshold used for the optional
LD rank diagnostic. It does not truncate the exact Cholesky computation.
Default is `1e-8`.

- PROFILE (optional): If True, report warm-up and steady-state time spent in
the beta, psi, and remaining MCMC stages. Default is False.

- BETA_STD (optional): If True, return standardized posterior SNP effect sizes (i.e., effect sizes corresponding to standardized genotypes with zero mean and unit variance across subjects). If False, return per-allele posterior SNP effect sizes, calculated by properly weighting the posterior standardized effect sizes using allele frequencies estimated from the reference panel. Default is False.

- WRITE_PSI (optional): If True, write variant-specific shrinkage estimates. Default is False.

- WRITE_POSTERIOR_SAMPLES (optional): If True, write all posterior samples of SNP effect sizes after thinning. Default is False.

- SEED (optional): Non-negative integer which seeds the random number generator.


## Output

PRS-CS writes posterior SNP effect size estimates for each chromosome to the user-specified directory. The output file contains chromosome, rs ID, base position, A1, A2 and posterior effect size estimate for each SNP. An individual-level polygenic score can be produced by concatenating output files from all chromosomes and then using `PLINK`'s `--score` command (https://www.cog-genomics.org/plink/1.9/score). If polygenic scores are generated by chromosome, use the 'sum' modifier so that they can be combined into a genome-wide score.


### Experimental joint chromosome sampling

`--joint_chromosomes=True` changes the sampling model rather than merely
scheduling independent chromosome jobs. The selected chromosomes are
concatenated as block-diagonal LD blocks and updated in one chain, so automatic
`phi` and `sigma` inference use sufficient statistics from every selected
chromosome. It never constructs a dense genome-wide LD matrix. For example:

```
python PRScs.py ... --chrom=1,2,3 --joint_chromosomes=True
```

Joint sampling preserves the normal `_chrN.txt` output files. Its memory
requirements are approximately the sum of the chromosome-wise requirements
because every selected chromosome's LD state is resident at once. The default
remains False for compatibility with established PRS-CS results. In
particular, an automatic-`phi` joint run is not expected to reproduce
independently fitted chromosome chains.

Joint mode scans the reference and BIM text files once and the summary-
statistics file twice for all selected chromosomes, rather than repeating
those whole-file scans for each chromosome. It also reports text-input and
per-chromosome LD loading timings. The default chromosome-wise mode retains
the original text-parsing path.


## Computational Efficiency

PRS-CS relies on numpy packages, which automatically use all available cores on a compute node. This can be problematic when running PRS-CS on a compute cluster; PRS-CS jobs may interfere with other jobs running on the same node, reducing computational efficiency. To resolve this issue, including the following code in the script to specify the number of threads in scipy:

```
export MKL_NUM_THREADS=$N_THREADS
export NUMEXPR_NUM_THREADS=$N_THREADS
export OMP_NUM_THREADS=$N_THREADS
```
For example, to use a single thread for the computation, set `N_THREADS=1`.

The CPU beta update uses preallocated column-major workspaces and calls the
underlying FP64 LAPACK Cholesky and BLAS triangular-solve routines directly.
This preserves the original conditional Gaussian update while avoiding
repeated temporary matrices and generic SciPy wrapper overhead.

The CPU local-shrinkage update fills the complete `psi` vector inside one
Numba-compiled loop rather than making a Python call for every variant.
Seeded runs explicitly seed Numba's independent random-number stream, and the
compiled vector entry point is cached for reuse by later Python processes.
Set `NUMBA_CACHE_DIR` to override Numba's cache location.

### Experimental CUDA backend

`--backend=cuda` keeps LD blocks resident on the selected GPU and performs the
same FP64 conditional Gaussian beta update using batched Cholesky and
triangular solves. Per iteration, only the O(number of variants) `psi` vector
is copied to the device and the sampled beta vector plus its quadratic form is
copied back. Chromosomes remain independent jobs unless
`--joint_chromosomes=True` is also selected.

The CUDA backend builds factorization and right-hand-side workspaces and
device pointer arrays once, then invokes FP64 `potrfBatched` and
`trsmBatched` directly. This avoids generic-wrapper copies without changing
the conditional Gaussian calculation.

Real LD panels can have only one or two blocks in most padded-size buckets.
For buckets containing fewer than eight matrices, the CUDA backend uses
ordinary `potrf` and `trsm` calls; denser buckets retain the batched routines.
This occupancy-aware dispatch preserves the exact FP64 transition.

For a quick synthetic CPU/CUDA comparison:

```
python3 benchmark_gpu.py --backends=cpu,cuda --n-iter=100
```

Add `--psi-backend=cuda` to fuse both local-shrinkage draws in one CUDA kernel
and avoid the intermediate host `delta` vector.


## Test Data

The test data contains GWAS summary statistics and a bim file for 1,000 SNPs on chromosome 22.
An example to use the test data:

`
python PRScs.py --ref_dir=path_to_ref/ldblk_1kg_eur --bim_prefix=path_to_bim/test --sst_file=path_to_sumstats/sumstats_se.txt --n_gwas=200000 --chrom=22 --phi=1e-2 --out_dir=path_to_output/eur
`


## Support

Please direct questions or bug reports to Tian Ge (tge1@mgh.harvard.edu).
