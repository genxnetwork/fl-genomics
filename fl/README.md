# Federated Learning Module

## Installation

1. Clone this repository
2. Run 
`conda env create -f fl.yaml`
3. Run 
`pip install git+https://github.com/chrchang/plink-ng/#subdirectory=2.0/Python`
## Input

### Client

1. Path to genotype .pgen files in configs/client/{config_name}.yaml `data.{train,val,test}` fields
.pgen files should be already filtered.
2. Path to GWAS results from plink 2.0 in configs/client/{config_name}.yaml `data.gwas` field

### Server

### Training Procedure

1. Launch `server.py` script with `python -m train.server`
2. Launch `client.py` scripts with `python -m train.client`


**MLP Regressor Training**

```
# will use cpu partition by default
sbatch train/server.sh node_count=4 epochs=128
# will launch 4 total jobs
sbatch --array=0-3 train/client.sh model.name=mlp_regressor +model.hidden_size=512 model.scheduler.rounds=128 model.scheduler.epochs_in_round=4
```