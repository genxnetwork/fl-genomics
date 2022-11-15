from omegaconf import DictConfig, OmegaConf
import mlflow
import hydra

from local.experiment import LassoNetExperiment


@hydra.main(config_path='configs', config_name='default')
def cross_node_model_eval(cfg: DictConfig):
    """
    Loads a trained model from a specified experiment/parameters and tests it on
    another node, logging the results.
    """
    print(OmegaConf.to_yaml(cfg))
    experiment = LassoNetExperiment(cfg)
    print("Loading data")
    experiment.load_data()
    
    print("Getting mlflow run")
    train_experiment = mlflow.get_experiment_by_name(cfg.train_experiment_name)
    runs = mlflow.list_run_infos(train_experiment.experiment_id)
    df = mlflow.search_runs(experiment_ids=[train_experiment.experiment_id])
    
    matching_runs = (df['tags.node_index'] == str(cfg.train_node_index))\
                   &(df['tags.snp_count'] == str(cfg.experiment.snp_count))\
                   &(df['tags.fold_index'] == str(cfg.fold_index))\
                   &(df['tags.phenotype'] == str(cfg.data.phenotype.name))
    
    assert sum(matching_runs) == 1
    
    mlflow_run_id = df.loc[matching_runs, 'run_id'].values[0]
    run = mlflow.get_run(mlflow_run_id)
    
    print("Loading model and predicting")
    loaded = mlflow.pytorch.load_model(f"runs:/{mlflow_run_id}/lassonet-model")
    loaded.eval()
    metrics = loaded.predict_and_eval(experiment.data_module, test=True)
    best_alpha_index = int(run.data.metrics['best_alpha_index'])
    
    print("Logging")
    
    mlflow.set_experiment(experiment.cfg.experiment.name)
    with mlflow.start_run(tags={'test_node_index': str(experiment.cfg.node_index),
                                'train_node_index': str(experiment.cfg.train_node_index),
                                'fold_index': str(experiment.cfg.fold_index),
                                'phenotype': experiment.cfg.data.phenotype.name}):
        mlflow.log_metric('train_r2', metrics.train[best_alpha_index].r2)
        mlflow.log_metric('val_r2', metrics.val[best_alpha_index].r2)
        mlflow.log_metric('test_r2', metrics.test[best_alpha_index].r2)

if __name__ == '__main__':
    cross_node_model_eval()
