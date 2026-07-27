# GhostBuster

## About the code
* `ghost_buster.py`: Main script to run the EM and find ghost populations
* `src/`: Supporting Python/R code (`plotting/`, `helper/`, and utility modules)
* `real_data_mask/`: Genome-build mask files used for filtering

Current support: GhostBuster currently only works with Relate-inferred trees.

## Installation (~5-10mins)
```
conda create -n gb python=3.13 -y
conda activate gb
conda install -c conda-forge gsl
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Examples (~10mins)
```
mkdir output/

python ghost_buster.py \
    --genome_build hg37 \
    --trees example/relate_chr \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 0 1 2 3 \
    --chrs 22 \
    --output output/relate
```

## Running GhostBuster on a SLURM cluster
You can split the EM inference by chromosome (e.g. via a SLURM array) and then merge all the pickles in one final pass.
```
#SBATCH --array=1-10
python ghost_buster.py \
    --genome_build hg37 \
    --trees example/relate_chr \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 0 1 2 3 \
    --chrs ${SLURM_ARRAY_TASK_ID} \
    --output output/relate \
    --only_make_pickle_files
```

Once every chromosome has its pickle in `output/relate`, rerun GhostBuster without `--trees` to aggregate:
```
python ghost_buster.py \
    --genome_build hg37 \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 0 1 2 3 \
    --chrs 1,2,3,4,5,6,7,8,9,10 \
    --output output/relate
```

## GhostBuster output files
For an output prefix `--output output/relate` and sample IDs `0 1 2 3` (label `0_1_2_3`), GhostBuster writes:

- `output/relate_tree_stats_chr<chr>.pkl`: cached tree statistics per chromosome.
- `output/relate_fixed_params_chr<chr>_sample<sample>.pkl`: cached fixed parameters.
- `output/relate_branch_persistence_chr<chr>_sample<sample>.pkl`: cached branch-persistence statistics.
- `output/relate_overall_membership_0_1_2_3.npy`: full posterior array.
- `output/relate_overall_membership_<sample_name>_sample_id_<sample_id>.csv`: local ancestry by window with `chr`, `pos`, `genpos`, `prob_*`.
- `output/relate_0_1_2_3.coal` (or `.coal.all` for many references): component-wise coalescence-rate table.
- `output/relate_gamma_0_1_2_3.npy`, `output/relate_props_0_1_2_3.npy`, `output/relate_tadmix_0_1_2_3.npy`: inferred EM parameters.
- `output/relate_0_1_2_3.logl`, `output/relate_0_1_2_3.tau`: EM trajectory logs.
- `output/relate_nohmm_overall_membership_<sample_name>_sample_id_<sample_id>.csv`, `output/relate_nohmm_0_1_2_3.coal`, `output/relate_props_nohmm_0_1_2_3.npy`: no-HMM outputs.

## Visualize `.coal` output
You can plot coalescence-rate curves using:

```bash
Rscript src/plotting/visualize_gamma.R output/relate 0_1_2_3
```

This reads `output/relate_nohmm_0_1_2_3.coal` (or `.coal.all`) and `output/relate_props_nohmm_0_1_2_3.npy`, and writes per-component SVGs such as:

- `output/relate_visual1.svg`
- `output/relate_gw_visual1.svg`
- `output/relate_gw.svg`

Required R packages are:

`install.packages(c("ggplot2", "cowplot", "dplyr", "tidyr", "pammtools", "ggplotify", "nls2", "ggthemes", "reticulate"))`

and

`remotes::install_github("leospeidel/relater")`

## LD dating with `ld_curve_dating.py`
`ld_curve_dating.py` assumes diploid samples by default and is most reliable when you have at least 2 diploid individuals (4 haplotypes) in the GhostBuster output.

Run LD-curve dating on GhostBuster local-ancestry CSV outputs:

```bash
python ld_curve_dating.py \
    --output output/relate \
    --genome_build hg37 \
    --mode 1date
```

Or run all supported models (`1date`, `2date`, `continuous`) by omitting `--mode`:

```bash
python ld_curve_dating.py \
    --output output/relate \
    --genome_build hg37
```

### LD dating outputs and interpretation
- `1date` console output reports `t_admix1` (single pulse admixture time, in generations) and fit score (`negloglike`/log MSE).
- `2date` console output reports `t_admix1`, `t_admix2`, and `weight` (mixture fraction of the first pulse).
- `continuous` console output reports `t_admix1`, `t_admix2`, and `mu` (continuous-admixture shape parameter in the model).
- `<output>_<mode>ld_curve.pdf` stores the matrix of co-ancestry curves with fitted model curves.
- `<output>_<mode>ld_curve_comp1.svg` stores the component-1 focused LD curve.
- Refit variants are written with `_refit`, for example `<output>_<mode>_refitld_curve.pdf`.

## Estimating population divergence times with `estimate_divergence.py`
`estimate_divergence.py` estimates the divergence time between two populations from coalescence rates along Relate-inferred (or simulated) trees. It counts within-population (A–A, B–B) and cross-population (A–B) coalescences per local tree, forms the coalescence rates in log-spaced time epochs, and reports the time at which the relative cross-coalescence rate `rCCR(t) = R_AB / (0.5 * (R_AA + R_BB))` rises through `--rccr_threshold` (default 0.9). A 95% CI is obtained by a genomic block bootstrap.

Run it on the bundled example trees (groups `focal` and `B` in `example/poplabels.txt`):

```bash
python estimate_divergence.py \
    --prefix example/relate_chr --chrs 22 \
    --poplabels example/poplabels.txt --pops focal B \
    --output output/div_focal_B
```

This reads `example/relate_chr22.trees` and prints:

```
Divergence(focal,B) = 21,796 years (mean of bootstrap)  rCCR threshold in [0.9, 0.9]  95% CI [14,550, 29,315]  (n=1000/1000)
```

`--epoch_start` / `--epoch_end` set the evaluated time window in log10 years (defaults `4` and `6`); the crossing must fall inside it. `--pops` takes two `GROUP` labels from the poplabels file. Optional flags: `--mask1` / `--mask2` restrict a population to sites where each sample's local ancestry (from GhostBuster `*_overall_membership_*` CSVs) exceeds `--local_ancestry_threshold`; `--recomb_map` restricts to low-recombination trees; `--dump_rates` writes the per-bootstrap coalescence-rate curves. Run `python estimate_divergence.py --help` for the full list.

## Checking component separation with `pca_silhouette.py`
`pca_silhouette.py` measures how well-separated GhostBuster's two inferred ancestry components are. Using the per-site coalescence-count matrix from a GhostBuster run, it labels each site by its `k=2` posterior argmax and reports a silhouette score (≈0 = the components are indistinguishable / noise; higher = genuine structure), plus a PCA density plot.

Run it on the output prefix of a GhostBuster run (see [Examples](#examples-10mins) above); the plot path is optional (defaults to `<prefix>_pca.svg`):

```bash
python pca_silhouette.py output/relate output/relate_pca.svg
```

This reads `output/relate_fixed_params_chr*_sample*.pkl` and `output/relate_overall_membership_*_sample_id_*.csv` and prints:

```
SILHOUETTE_NUM 0.0363
SILHOUETTE_X 0.1560
NSITES 13164
NFEATURES 4
wrote output/relate_pca.svg
```

`SILHOUETTE_NUM` scores the coalescence-count matrix, `SILHOUETTE_X` the full standardized feature matrix. The low value here is expected for the example, whose haplotypes carry no real second ancestry component.

## List of options
```
python ghost_buster.py --help
```

The table below lists commonly used options. For the full list, run `python ghost_buster.py --help`.

| Option                                  | Datatype            | Default        | Description                                                                                                           |
|-----------------------------------------|---------------------|----------------|-----------------------------------------------------------------------------------------------------------------------|
| `--sample_id`                           | list<int>           | `required`     | Enter space-separated list of the indices of haplotypes to analyze local ancestry for                                 |
| `--trees`                               | string              | `None`         | Directory of tskit trees                                                                                              |
| `--poplabels`                           | string              | `required`     | Population labels file                                                                                                |
| `--rec`                                 | string              | `required`     | Filename of recombination map                                                                                         |
| `--chrs`                                | list<int>           | `"1,2"`        | Comma-separated list of chromosomes to process                                                                        |
| `--output`                              | string              | `""`           | Output prefix                                                                                                         |
| `--num_clusters`                        | int                 | `2`            | Number of clusters to infer by EM                                                                                     |
| `--mut_scaling`                         | boolean             | `True`         | Scale likelihood by presence of mutation on lineage (improves robustness to bottlenecks)                              |
| `--ypg`                                 | float               | `28`           | Years per generation                                                                                                  |
| `--node_persist_thresh`                 | float               | `0.5`          | Correlation threshold above which nodes are considered equivalent                                                     |
| `--cm_grid`                             | float               | `None`         | Store local-ancestry information per centiMorgan                                                                      |
| `--masking_threshold`                   | float               | `0.5`          | Remove top 100*x percentile of high-recombination regions                                                                 |
| `--t_admix_guess`                       | string              | `None`         | Initial guess for time of admixture                                                                                   |
| `--start_time`                          | float               | `4.5`          | Start time for population-size plots (log-scale)                                                                      |
| `--end_time`                            | float               | `6`            | End time for population-size plots (log-scale)                                                                        |
| `--num_epochs`                          | int                 | `9`            | Number of epochs                                                                                                      |
| `--seed`                                | int                 | `2`            | Random seed for reproducibility                                                                                        |
| `--force_build`                         | int                 | `10000`        | Force build size to subsample the trees (in bp)                                                                       |
| `--num_iters`                           | int                 | `200`          | Number of EM iterations                                                                                                |
| `--n_repeats`                           | int                 | `20`           | Number of EM restarts when randomly initializing local ancestry; best run chosen                                      |
| `--sweep_num_iters`                     | int                 | `10`           | Number of EM iterations per random sweep                                                                               |
| `--only_make_pickle_files`              | flag                | `False`        | Build tree/fixed/branch pickle files only; skips EM and forces `--num_iters 1 --n_repeats 1 --sweep_num_iters 1`     |
| `--groups`                              | list<string>        | `None`         | Space-separated list of source groups (e.g. `Nea CHB`)                                                                 |
| `--load_gamma`                          | string              | `None`         | Starting γ values from file                                                                                           |
| `--load_mask`                           | string              | `None`         | Load mask CSV file (columns: chr, tree_position_left)                                                                 |
| `--load_props`                          | string              | `None`         | Starting τ values from file or space-separated list                                                                   |
| `--tree_stats_file_prefix`              | string              | `None`         | Prefix for output tree-statistics file                                                                                 |
| `--branch_persistence_file_prefix`      | string              | `None`         | Prefix for output branch-persistence file                                                                              |
| `--fixed_params_file_prefix`            | string              | `None`         | Prefix for output fixed-parameters file                                                                                |
| `--load_membership`                     | string              | `None`         | Load membership array from `.npy` file                                                                                 |
| `--genome_build`                        | string              | `None`         | Genome build for filtering (e.g. `hg38`, `hg37`, or `None`)                                                           |
| `--gt_ref`                              | string              | `None`         | Local ancestry of the reference panel                                                                                  |
| `--ignore_coal_between_targets`         | boolean             | `False`        | Ignore coalescence events between target samples                                                                       |
| `--hmm`                                 | boolean             | `True`         | Run HMM (`True`) or treat windows as independent (`False`)                                                            |
| `--ignore_first_epoch`                  | boolean             | `True`         | Ignore first epoch when calculating local ancestry in EM                                                              |
| `--ignore_last_epoch`                   | boolean             | `True`         | Ignore last epoch when calculating local ancestry in EM                                                               |


## Citation
Loya et al. under review, "Genome-wide genealogies reveal deep admixtures forming modern humans"

## Contact
loyahrushikesh@gmail.com
