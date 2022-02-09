import os
import torch
from torch.utils.data import TensorDataset

from omegaconf import DictConfig
import hydra
import mlflow

from model.models import EnsembleLASSO
from datasets.memory import load_from_pgen, load_phenotype, get_sample_indices
from datasets.lightning import prepare_trainer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

@hydra.main(config_path='../configs/local', config_name='default')
def local_experiment(cfg: DictConfig):
    sample_indices_train=get_sample_indices(cfg.data.genotype.train, cfg.data.phenotype.train)
    sample_indices_val=get_sample_indices(cfg.data.genotype.val, cfg.data.phenotype.val)
    
    X = load_from_pgen(cfg.data.genotype.train, cfg.data.gwas, snp_count=cfg.experiment.snp_count, sample_indices=sample_indices_train)
    X_test = load_from_pgen(cfg.data.genotype.val, cfg.data.gwas, snp_count=cfg.experiment.snp_count, sample_indices=sample_indices_val)
    y = load_phenotype(cfg.data.phenotype.train)
    y_test = load_phenotype(cfg.data.phenotype.val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=cfg.experiment.random_state)

    train_dataset = TensorDataset(
                                torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )

    val_dataset = TensorDataset(
                                torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    test_dataset = TensorDataset(
                                torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )

    input_size = X.shape[1]
    
    with mlflow.start_run(tags={
                            'name': 'lassonet',
                            'type': 'local',
                            'phenotype': cfg.phenotype.name,
                            'node_index': str(cfg.node_index),
                            'snp_count': str(cfg.experiment.snp_count)
                            }
                        ) as run:
        model = EnsembleLASSO(train_dataset, val_dataset, test_dataset=test_dataset, alpha_start=cfg.model.alpha_start, alpha_end=cfg.model.alpha_end, input_size=input_size, batch_size=cfg.model.batch_size,
                              hidden_size=cfg.model.hidden_size)
        trainer = prepare_trainer('models', 'logs', f'ensemble_lasso/{cfg.phenotype.name}}', f'run{run.info.run_id}', gpus=cfg.experiment.gpus, precision=cfg.model.precision,
                                    max_epochs=cfg.model.max_epochs, weights_summary='full', patience=10, log_every_n_steps=5)
        trainer.fit(model)

        best_model = EnsembleLASSO.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                            train_dataset=train_dataset, val_dataset=val_dataset, 
                                            input_size=input_size, batch_size=cfg.model.batch_size,
                                            hidden_size=cfg.model.hidden_size, alpha_start=cfg.model.alpha_start, 
                                            alpha_end=cfg.model.alpha_end, num_workers=1,
                                            total_steps=cfg.model.max_epochs)

        best_model.eval()

        preds = best_model.predict(model.train_dataloader()).cpu().numpy()
        val_preds = best_model.predict(model.val_dataloader()).cpu().numpy()
        test_preds = best_model.predict(model.test_dataloader()).cpu().numpy()
        max_val_r2 = 0.0
        best_col = 0

        ln_train_r2s = []
        ln_val_r2s = []
        for col in range(cfg.model.hidden_size):
            train_r2 = r2_score(y_train, preds[:, col])
            val_r2 = r2_score(y_val, val_preds[:, col])
            ln_train_r2s.append(train_r2)
            ln_val_r2s.append(val_r2)
            if val_r2 > max_val_r2:
                max_val_r2 = val_r2
                best_col = col
            print(f'for alpha {best_model.alphas[col]:.4f} train_r2 is {train_r2:.4f}, val_r2 is {val_r2:.4f}')

        print(f'test r2 for best alpha: {r2_score(y_test, test_preds[:, best_col]):.4f}')
        mlflow.log_metric('train_r2', r2_score(y_train, preds[:, best_col]))
        mlflow.log_metric('val_r2', r2_score(y_val, val_preds[:, best_col]))
        mlflow.log_metric('test_r2', r2_score(y_test, test_preds[:, best_col]))
    

if __name__ == '__main__':
    local_experiment()
