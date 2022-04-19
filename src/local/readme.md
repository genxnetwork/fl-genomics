### Individual experiments

To launch a local experiment with default configuration specified in `configs/default.yaml`, from `src` directory:

`python -m local.experiment`

To select models:

`python -m local.experiment +model=$MODEL_NAME`

To run an experiment with SNPs selected from GWAS on a different node,

`python -m local.experiment +experiment=different_node_gwas gwas_node_id=$GWAS_NODE_ID`

### Wrapper for launching multiple experiments

The wrapper is used to launch a series of experiments based on a config file with the parameters to be varied supplied as lists of values. Experiments will be run for every combination of those values.

A minimal default config is provided which can be modified, but it's better to define separate experiments. Configs for grid experiments are in `src/local/configs/grid/experiments`. A grid experiment can be run with:

`python -m local.wrapper +experiment=$CONFIG_NAME`

A grid config requires parameters:

* model.name (Selects the local model to launch)
* node_index
* experiment.snp_count
* split_dir

for knowing how many resources to request from the Slurm scheduler.