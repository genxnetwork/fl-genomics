### How to run snakemake on SLURM cluster

**test example**

```
    export PYTHONPATH=`pwd`/src
    srun mamba run -n fl snakemake --profile src/zhores --snakefile src/test_Snakefile
```

**ethnic split example from preprocessing to GWAS**

```
    module load python/mambaforge3
    export PYTHONPATH=`pwd`/src
    srun mamba run -n fl snakemake --snakefile src/gwas_Snakefile --directory /gpfs/gpfs0/ukb_data/test/ethnic_split/ --profile src/zhores --configfile `pwd`/src/gwas.yaml
```

### How to train FL models on SLURM cluster

```
    module load python/mambaforge3
    export PYTHONPATH=`pwd/src`
    sbatch src/fl/runner.py
```

**training of all models in src/fl/configs folder on two nodes from uneven split using snakemake**

```
FL_NODE_COUNT=2 srun --time 00:22:00 -o logs/output.txt mamba run -n fl snakemake --snakefile src/Snakefile --directory /gpfs/gpfs0/ukb_data/test/uneven_split/ --jobs 3 --configfile `pwd`/src/gwas_uneven.yaml --profile src/zhores --config snp_counts=["2000"] ethnicities=["WB4","WB5"] nodes=[4,5]
```