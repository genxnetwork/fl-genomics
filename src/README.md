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