from collections import namedtuple
import logging
import os
from omegaconf import OmegaConf, DictConfig
import mlflow
from socket import gethostname
from flwr.server import start_server
from flwr.server.strategy import FedAvg
from federation.strategy import Checkpointer, MCFedAvg, MCFedAdagrad, MCFedAdam, MCQFedAvg, MlflowLogger, fit_round


def get_strategy(strategy_params: DictConfig, epochs_in_round: int, checkpoint_dir: str) -> FedAvg:
    """Creates flwr Strategy from strategy config entry

    Args:
        strategy_params (DictConfig): Strategy params with name, node_count and optional args dict
        checkpoint_dir (str): Directory for saving model checkpoints

    Raises:
        ValueError: If strategy_params.name is unknown

    Returns:
        FedAvg: flwr Strategy with checkpointing and mlflow logging capabilities
    """    
    default_args = OmegaConf.create({
        "fraction_fit": 0.99,
        "fraction_eval": 0.99,
        "min_fit_clients": strategy_params.node_count,
        "min_eval_clients": strategy_params.node_count,
        "min_available_clients": strategy_params.node_count
    })
    if 'args' in strategy_params:
        args = OmegaConf.merge(default_args, strategy_params.args)
    else:
        args = default_args

    mlflow_logger = MlflowLogger(epochs_in_round)
    checkpointer = Checkpointer(checkpoint_dir)
    if strategy_params.name == 'fedavg':
        return MCFedAvg(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name == 'qfedavg':
        return MCQFedAvg(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name ==  'fedadam':
        return MCFedAdam(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name == 'fedadagrad':
        return MCFedAdagrad(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    else:
        raise ValueError(f'Strategy name {strategy_params.name} should be one of the ["fedavg", "qfedavg", "fedadam", "fedadagrad"]')


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'resources', 'log'])
        snakemake = Snakemake(
            input={'name': 'test.input'},
            output={'name': 'test.output'},
            params={'parameter': 'value'},
            resources={'resource_name': 'value'},
            log=['test.log']
        )

    config_path = snakemake.input['config'][0]
    params_hash = snakemake.wildcards['params_hash']
    checkpoint_dir = snakemake.params['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')

    cfg = OmegaConf.load(config_path)
    
    experiment = mlflow.set_experiment(cfg.experiment.name)
    strategy = get_strategy(cfg.server.strategy, cfg.node.scheduler.epochs_in_round, checkpoint_dir)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        tags={
            'description': cfg.experiment.description,
            'params_hash': params_hash,
            'hostname': gethostname()
        }
    ) as run:
        mlflow.log_params(cfg.server)
        logging.info(f'starting server with min clients {cfg.server.strategy.node_count} and {cfg.server.rounds} max rounds')
        start_server(
                    server_address="[::]:8080",
                    strategy=strategy,
                    config={"num_rounds": cfg.server.rounds}
        )
    
    best_model_path = snakemake.output[0]
    strategy.checkpointer.copy_best_model(best_model_path)
        