To launch a local experiment with default configuration specified in `configs/default.yaml`, from `src` directory:

`python -m local.experiment`

To select models:

`python -m local.experiment +model=$MODEL_NAME`

To run an experiment with SNPs selected from GWAS on a different node,

`python -m local.experiment +experiment=different_node_gwas gwas_node_id=$GWAS_NODE_ID`