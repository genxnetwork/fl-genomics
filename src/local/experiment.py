import hydra
import logging
from sys import stdout
from omegaconf import DictConfig
import mlflow
from mlflow.xgboost import autolog
from numpy import hstack
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import torch

from local.config import node_size_dict, node_name_dict
from fl.datasets.memory import load_covariates, load_phenotype, load_from_pgen, get_sample_indices
from nn.lightning import DataModule
from nn.train import prepare_trainer
from nn.models import MLPRegressor

class LocalExperiment():
    """
    Base class for experiments in a local setting
    """
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Configuration for experiments from hydra
        """
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO,
                        stream=stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
        self.logger = logging.getLogger()
            
    def start_mlflow_run(self):
        split = self.cfg.split_dir.split('/')[-1]
        num_samples = node_size_dict[split][self.cfg.node_index]
        mlflow.set_experiment(f'local-models')
        self.run = mlflow.start_run(tags={
            'model': self.cfg.model.name,
            'split': split,
            'phenotype': self.cfg.phenotype.name,
            'node_index': str(self.cfg.node_index),
            'snp_count': str(self.cfg.experiment.snp_count),
            'sample_count': str(round(num_samples, -2)),
            'sample_count_exact': str(num_samples),
            'dataset': f"{node_name_dict[split][self.cfg.node_index]}_{round(num_samples, -2)}",
            'different_node_gwas': str(int(self.cfg.experiment.different_node_gwas))}
        )

    def load_data(self):
        self.logger.info("Loading data")
        
        self.y_train = load_phenotype(self.cfg.data.phenotype.train)
        self.y_val = load_phenotype(self.cfg.data.phenotype.val)
        self.y_test = load_phenotype(self.cfg.data.phenotype.test)
        
        assert self.cfg.experiment.include_genotype or self.cfg.experiment.include_covariates
        
        if self.cfg.experiment.include_genotype and self.cfg.experiment.include_covariates:
            self.load_genotype_and_covariates_()    
        elif self.cfg.experiment.include_genotype:
            self.load_genotype_()
        else:
            self.load_covariates()
            
        self.logger.info(f"{self.X_train.shape[1]} features loaded")
        
    def load_sample_indices(self):
        self.logger.info("Loading sample indices")
        self.sample_indices_train = get_sample_indices(self.cfg.data.genotype.train,
                                                       self.cfg.data.phenotype.train)
        self.sample_indices_val = get_sample_indices(self.cfg.data.genotype.val,
                                                     self.cfg.data.phenotype.val)
        self.sample_indices_test = get_sample_indices(self.cfg.data.genotype.test,
                                                      self.cfg.data.phenotype.test)
        
    def load_genotype_and_covariates_(self):
        self.X_train = hstack((load_from_pgen(self.cfg.data.genotype.train,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count,
                                              sample_indices=self.sample_indices_train),
                               load_covariates(self.cfg.data.covariates.train)))
        self.X_val = hstack((load_from_pgen(self.cfg.data.genotype.val,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count,
                                              sample_indices=self.sample_indices_val),
                               load_covariates(self.cfg.data.covariates.val)))
        self.X_test = hstack((load_from_pgen(self.cfg.data.genotype.test,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count,
                                              sample_indices=self.sample_indices_test),
                               load_covariates(self.cfg.data.covariates.test)))
        
    def load_genotype_(self):
        self.X_train = load_from_pgen(self.cfg.data.genotype.train,
                                      self.cfg.data.gwas,
                                      snp_count=self.cfg.experiment.snp_count,
                                      sample_indices=self.sample_indices_train)
        self.X_val = load_from_pgen(self.cfg.data.genotype.val,
                                    self.cfg.data.gwas,
                                    snp_count=self.cfg.experiment.snp_count,
                                    sample_indices=self.sample_indices_val)
        self.X_test = load_from_pgen(self.cfg.data.genotype.test,
                                     self.cfg.data.gwas,
                                     snp_count=self.cfg.experiment.snp_count,
                                     sample_indices=self.sample_indices_test)
        
    def load_covariates_(self):
        self.X_train = load_covariates(self.cfg.data.covariates.train)
        self.X_val = load_covariates(self.cfg.data.covariates.val)
        self.X_test = load_covariates(self.cfg.data.covariates.test)
    
    def train(self):
        pass
        
    def eval_and_log(self):
        self.logger.info("Evaluating model")
        preds_train = self.model.predict(self.X_train)
        preds_val = self.model.predict(self.X_val)
        preds_test = self.model.predict(self.X_test)

        r2_train = r2_score(self.y_train, preds_train)
        r2_val = r2_score(self.y_val, preds_val)
        r2_test = r2_score(self.y_test, preds_test)

        print(f"Train r2: {r2_train}")
        mlflow.log_metric('train_r2', r2_train)
        print(f"Val r2: {r2_val}")
        mlflow.log_metric('val_r2', r2_val)
        print(f"Test r2: {r2_test}")
        mlflow.log_metric('test_r2', r2_test)
    
    def run(self):
        self.load_sample_indices()
        self.load_data()
        self.start_mlflow_run()
        self.train()
        self.eval_and_log()
    

def simple_estimator_factory(model):
    """Returns a SimpleEstimatorExperiment for a given model class, expected
    to have the same interface as scikit-learn estimators.
    
    Args:
        model: Model class
        model_kwargs_dict: Dictionary of parameters passed during model initialization
    """
    class SimpleEstimatorExperiment(LocalExperiment):
        def __init__(self, cfg):
            LocalExperiment.__init__(self, cfg)
            self.model = model(**self.cfg.model.params)

        def train(self):
            self.logger.info("Training")
            self.model.fit(self.X_train, self.y_train)
       
    return SimpleEstimatorExperiment

class XGBExperiment(LocalExperiment):
    def __init__(self, cfg):
        LocalExperiment.__init__(self, cfg)
        self.model = XGBRegressor(**self.cfg.model.params)

    def train(self):
        self.logger.info("Training")
        autolog()
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)],
                       early_stopping_rounds=self.cfg.model.early_stopping_rounds, verbose=True)

class NNExperiment(LocalExperiment):
    def __init__(self, cfg):
        LocalExperiment.__init__(self, cfg)

    def load_data(self):
        LocalExperiment.load_data(self)
        self.data_module = DataModule(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, batch_size=self.cfg.model.batch_size)
        
    def train(self):
        mlflow.log_params({'model': self.cfg.model})
        mlflow.log_params({'optimizer': self.cfg.experiment.optimizer})
        mlflow.log_params({'scheduler': self.cfg.experiment.scheduler})
        
        input_size = self.X_train.shape[1]
        self.model = MLPRegressor(input_size=input_size,
                                  hidden_size=self.cfg.model.hidden_size,
                                  l1=self.cfg.model.alpha,
                                  optim_params=self.cfg.experiment.optimizer,
                                  scheduler_params=self.cfg.experiment.scheduler
                                 )
        self.trainer = prepare_trainer('models', 'logs', f'{self.cfg.model.name}/{self.cfg.phenotype.name}', f'run{self.run.info.run_id}', gpus=self.cfg.experiment.gpus, precision=self.cfg.model.precision,
                                    max_epochs=self.cfg.model.max_epochs, weights_summary='full', patience=10, log_every_n_steps=5)
        
        print("Fitting")
        self.trainer.fit(self.model, self.data_module)
        print("Fitted")
        self.load_best_model()
        print(f'Loaded best model {self.trainer.checkpoint_callback.best_model_path}')

    def load_best_model(self):
        self.model = MLPRegressor.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path,
            input_size=self.X_train.shape[1],
            hidden_size=self.cfg.model.hidden_size,
            l1=self.cfg.model.alpha,
            optim_params=self.cfg.experiment.optimizer,
            scheduler_params=self.cfg.experiment.scheduler
        )
        
    def eval_and_log(self):
        self.model.eval()
        train_preds, val_preds, test_preds = self.trainer.predict(self.model, self.data_module)
        
        train_preds = torch.cat(train_preds).squeeze().cpu().numpy()
        val_preds = torch.cat(val_preds).squeeze().cpu().numpy()
        test_preds = torch.cat(test_preds).squeeze().cpu().numpy()
                
        r2_train = r2_score(self.y_train, train_preds)
        r2_val = r2_score(self.y_val, val_preds)
        r2_test = r2_score(self.y_test, test_preds)
        
        print(f"Train r2: {r2_train}")
        mlflow.log_metric('train_r2', r2_train)
        print(f"Val r2: {r2_val}")
        mlflow.log_metric('val_r2', r2_val)
        print(f"Test r2: {r2_test}")
        mlflow.log_metric('test_r2', r2_test)
        
# Dict of possible experiment types and their corresponding classes
experiment_dict = {
    'lasso': simple_estimator_factory(LassoCV),
    'xgboost': XGBExperiment,
    'mlp': NNExperiment
}

            
@hydra.main(config_path='configs', config_name='default')
def local_experiment(cfg: DictConfig):
    assert cfg.model.name in experiment_dict.keys()
    experiment = experiment_dict[cfg.model.name](cfg)
    experiment.run()   
    
if __name__ == '__main__':
    local_experiment()
