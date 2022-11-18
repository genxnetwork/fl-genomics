# FedAvg on UKB data on SLURM cluster

4 Nodes

```sbatch --gpus 1 --partition gpu_devel --cpus-per-task 4 --mem 26000 --time 01:00:00 src/fl/runner.sh split=assessment data=meta_gwas strategy=fedavg strategy.active_nodes=[n2,n3,n4,n5] server.rounds=128 scheduler.epochs_in_round=1 model=lassonet_regressor model.hidden_size=256 model.batch_size=16 optimizer.weight_decay=0.0 optimizer.lr=5e-3 experiment.name=fl-ukb-scaffold-test experiment.pretrain_on_cov=weights experiment.snp_count=2000 +sample_weights=False server.port=8081 scheduler.gamma=0.99 split.nodes.n2.resources.gpus=1 +training.devices=1 +model.use_bn=False model.alpha_start=-3 model.alpha_end=1 model.init_limit=0.01 fold.index=6 data.phenotype.name=standing_height```


### Cancellation of all user jobs on SLURM cluster


```squeue --me -h -o "%i" | xargs scancel```