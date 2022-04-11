from multiprocessing import Process, Queue
import os

from omegaconf import DictConfig, OmegaConf
import numpy
from flwr.server import start_server
from flwr.server.strategy import FedAvg

from fl.federation.strategy import Checkpointer, MCFedAvg, MCFedAdagrad, MCFedAdam, MCQFedAvg, MlflowLogger, fit_round


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


class Server(Process):
    def __init__(self, queue: Queue, params_hash: str, cfg_path: str, **kwargs):
        Process.__init__(self, **kwargs)
        self.queue = queue
        self.params_hash = params_hash
        null_node = OmegaConf.from_dotlist([f'node.index=null'])
        self.cfg = OmegaConf.merge(null_node, OmegaConf.load(cfg_path))

    def run(self) -> None:
        strategy = get_strategy(self.cfg.server.strategy, 
                                self.cfg.node.scheduler.epochs_in_round, 
                                os.path.join(self.cfg.server.checkpoint_dir, self.params_hash))
        start_server(
                    server_address="[::]:8080",
                    strategy=strategy,
                    config={"num_rounds": self.cfg.server.rounds}
        )
    
        strategy.checkpointer.copy_best_model(os.path.join(self.cfg.server.checkpoint_dir, self.params_hash, 'best_model.ckpt'))