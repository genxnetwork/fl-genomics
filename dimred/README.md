# Dimensionality Reduction

## GWAS

### Input

1. plink .pgen files for each node in split.
2. Phenotype .tsv files with covariates for each node in split.

### Output

1. .tsv file with p-values for all SNPs in .pgen file.
2. QQ and manhattan plots for each GWAS.
3. .pgen file with maximum possible number of significant SNPs: 10K for each node in split.
4. Union lengths statistics plot.

### How to Run

1. Split genotype and phenotype data into 10 cv folds. Each fold contains 80% train, 10% val and 10% test samples.
```
    sbatch src/gwas/train_test_split.sh split_dir=<dir with split> node_count=<number of split parts>
```
2. Run a GWAS on each node index
```
    sbatch src/gwas/gwas.sh split_dir=<dir with split> node_index=0
```
3. Visualise GWAS results.
Analysis script will produce the following plots in `<split_dir>/gwas`:
    - `log10(p-value)` distribution in `split<split_index>.hist.png`
    - qq-plot in `split<split_index>.qqplot.png`
    - manhattan plot in `split<split_index>.manhattan.png`

```
    sbatch src/gwas/analysis.sh split_dir=<dir with split> split_index=0
```
4. Construct node-based datasets with the same SNPs (union from node-based GWASes). 
First, intesection of all SNPs from all nodes will be constructed.
Second, `max_snp_count` most significant SNPs will be taken from each GWAS.
Last, union of SNPs from the second step will be intersected with SNPs from the first step.
```
    sbatch src/gwas/union.sh split_dir=<dir with split> max_snp_count=1000 split_count=<number of split parts>
```
