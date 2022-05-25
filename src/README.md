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

**Customize both cluster node options and training options**
```
    sbatch --time 00:40:00 --partition gpu --gpus 4 --cpus-per-task 24 --mem 240000 src/fl/runner.py node.model.hidden_size=1024
```



### Dataset statistics

**Uneven split**

```
wc -l node_{0,1,2,3,4,5,6,7}/fold_0_train.tsv
   6103 node_0/fold_0_train.tsv
   1199 node_1/fold_0_train.tsv
 171578 node_2/fold_0_train.tsv
  85813 node_3/fold_0_train.tsv
  42884 node_4/fold_0_train.tsv
  21444 node_5/fold_0_train.tsv
  10717 node_6/fold_0_train.tsv
   3432 node_7/fold_0_train.tsv
 343170 total
 ```

### [Deprecated] Old way

**training of all models in src/fl/configs folder on two nodes from uneven split using snakemake**

```
FL_NODE_COUNT=2 srun --time 00:22:00 -o logs/output.txt mamba run -n fl snakemake --snakefile src/Snakefile --directory /gpfs/gpfs0/ukb_data/test/uneven_split/ --jobs 3 --configfile `pwd`/src/gwas_uneven.yaml --profile src/zhores --config snp_counts=["2000"] ethnicities=["WB4","WB5"] nodes=[4,5]
```



