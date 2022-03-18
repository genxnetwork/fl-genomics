import hydra
from omegaconf import DictConfig
import mlflow
from numpy import hstack
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


from fl.datasets.memory import load_covariates, load_phenotype, load_from_pgen

class LocalExperiment():
    """
    Base class for experiments in a local setting
    
    Args:
        cfg: Configuration for experiments from hydra
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    def start_mlflow_run(self):
        mlflow.set_experiment('local')
        self.run = mlflow.start_run(tags={
                            'name': self.cfg.experiment.model,
                            'split': self.cfg.split_dir.split('/')[-1],
                            'phenotype': self.cfg.phenotype.name,
                            'node_index': str(self.cfg.node_index),
                            'snp_count': str(self.cfg.experiment.snp_count),
                            'gwas_path': self.cfg.data.gwas}
                            )

    def load_data(self):
        print("Loading data")
        
        self.y_train = load_phenotype(self.cfg.data.phenotype.train)
        self.y_val = load_phenotype(self.cfg.data.phenotype.val)
        self.y_test = load_phenotype(self.cfg.data.phenotype.test)
        
        assert self.cfg.experiment.include_genotype or self.cfg.experiment.include_covariates
        
        if self.cfg.experiment.include_genotype and self.cfg.experiment.include_covariates:
            print("Loading genotypes and covariates")
            self.load_genotype_and_covariates_()    
        elif self.cfg.experiment.include_genotype:
            print("Loading genotypes")
            self.load_genotype_()
        else:
            print("Loading covariates")
            self.load_covariates()
        
    def load_genotype_and_covariates_(self):
        self.X_train = hstack((load_from_pgen(self.cfg.data.genotype.train,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count),
                               load_covariates(self.cfg.data.covariates.train)))
        self.X_val = hstack((load_from_pgen(self.cfg.data.genotype.val,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count),
                               load_covariates(self.cfg.data.covariates.val)))
        self.X_test = hstack((load_from_pgen(self.cfg.data.genotype.test,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count),
                               load_covariates(self.cfg.data.covariates.test)))
        
    def load_genotype_(self):
        self.X_train = load_from_pgen(self.cfg.data.genotype.train,
                                          self.cfg.data.gwas,
                                          snp_count=self.cfg.experiment.snp_count)
        self.X_val = load_from_pgen(self.cfg.data.genotype.val,
                                    self.cfg.data.gwas,
                                    snp_count=self.cfg.experiment.snp_count)
        self.X_test = load_from_pgen(self.cfg.data.genotype.test,
                                     self.cfg.data.gwas,
                                     snp_count=self.cfg.experiment.snp_count)
    def load_covariates_(self):
        self.X_train = load_covariates(self.cfg.data.covariates.train)
        self.X_val = load_covariates(self.cfg.data.covariates.val)
        self.X_test = load_covariates(self.cfg.data.covariates.test)
    
    def train(self):
        pass
        
    def eval_and_log(self):
        pass
    
    def run(self):
        pass

def simple_estimator_decorator(model, model_kwargs_dict: dict=dict()):
    """Returns a SimpleEstimatorExperiment for a given model class, expected
    to have the same interface as scikit-learn estimators.
    
    Args:
        model: Model class
        model_kwargs_dict: Dictionary of parameters passed during model initialization
    """
    class SimpleEstimatorExperiment(LocalExperiment):
        def __init__(self, cfg):
            LocalExperiment.__init__(self, cfg)
            self.model = model(**model_kwargs_dict)

        def train(self):
            print("Training")
            self.model.fit(self.X_train, self.y_train)

        def eval_and_log(self):
            print("Eval")
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
            self.load_data()
            self.start_mlflow_run()
            self.train()
            self.eval_and_log() 
    return SimpleEstimatorExperiment

def xgboost_decorator(xgb_kwargs_dict: dict=dict()):
    """Returns XGBExperiment class containing a regressor with model parameters specified in
    a given dictionary
    
    Args:
        xgb_kwargs_dict: Dictionary of parameters passed during model initialization
    """
    class XGBExperiment(LocalExperiment):
        def __init__(self, cfg):
            LocalExperiment.__init__(self, cfg)
            self.model = XGBRegressor(**xgb_kwargs_dict)

        def train(self):
            print("Training")
            self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], early_stopping_rounds=20, verbose=True)

        def eval_and_log(self):
            print("Eval")
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
            self.load_data()
            self.start_mlflow_run()
            self.train()
            self.eval_and_log()
            
    return XGBExperiment

# todo: model configs from hydra?            
xgb_kwargs_dict = {
    'max_depth': 3,
    'alpha': 0.5,
    'n_estimators': 1000
}
        
# Dict of possible experiment types and their corresponding classes
experiment_dict = {
    'lasso': simple_estimator_decorator(LassoCV),
    'xgboost': xgboost_decorator(xgb_kwargs_dict)
}

            
@hydra.main(config_path='configs', config_name='default')
def local_experiment(cfg: DictConfig):
    assert cfg.experiment.model in experiment_dict.keys()
    experiment = experiment_dict[cfg.experiment.model](cfg)
    experiment.run()   
    
if __name__ == '__main__':
    local_experiment()