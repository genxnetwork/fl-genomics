import hydra
from omegaconf import DictConfig

import mlflow
import numpy as np
from pytorch_lightning import Trainer
import torch
from sklearn.metrics import r2_score

from local.experiment import LassoNetExperiment

@hydra.main(config_path='configs', config_name='test')
def test(cfg: DictConfig):
    print(cfg)
    
    experiment = LassoNetExperiment(cfg)
    experiment.load_sample_indices()
    experiment.load_data()
    
    print("Loading train experiment")
    train_experiment = mlflow.get_experiment_by_name('ethnic-split-all-snps-train')
    runs = mlflow.list_run_infos(train_experiment.experiment_id)
    df = mlflow.search_runs(experiment_ids=[train_experiment.experiment_id])

    assert(sum(df['tags.node_index'] == str(cfg.train_node_index)) == 1)
    mlflow_run_id = df.loc[(df['tags.node_index'] == str(cfg.train_node_index)) and (df['tags.snp_count'] == '200000'), 'run_id'].values[0]
    run = mlflow.get_run(mlflow_run_id)
    loaded = mlflow.pytorch.load_model(f"runs:/{mlflow_run_id}/lassonet-model")
    loaded.eval()
    
    print("Predicting")
    
    trainer = Trainer(gpus=0)
    train_preds, val_preds, test_preds = trainer.predict(loaded, experiment.data_module)

    best_col = int(run.data.metrics['best_alpha_index'])
    best_val_r2 = r2_score(experiment.y_val, torch.cat(val_preds)[:, best_col].numpy())
    best_train_r2 = r2_score(experiment.y_train, torch.cat(train_preds)[:, best_col].numpy())
    best_test_r2 = r2_score(experiment.y_test, torch.cat(test_preds)[:, best_col].numpy())

    print("Logging")
    
    mlflow.set_experiment(experiment.cfg.experiment.name)
    with mlflow.start_run(tags={'test_node_index': str(experiment.cfg.node_index),
                                'train_node_index': str(experiment.cfg.train_node_index)}):
        print(f"Train r2: {best_train_r2}")
        mlflow.log_metric('train_r2', best_train_r2)
        print(f"Val r2: {best_val_r2}")
        mlflow.log_metric('val_r2', best_val_r2)
        print(f"Test r2: {best_test_r2}")
        mlflow.log_metric('test_r2', best_test_r2)

if __name__ == '__main__':
    test()
    
