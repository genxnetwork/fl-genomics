### How to run snakemake on SLURM cluster

```
    export PYTHONPATH=`pwd`/src
    srun mamba run -n fl snakemake --profile src/zhores --snakefile src/test_Snakefile
```
