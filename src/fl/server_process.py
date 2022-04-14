from multiprocessing import Process, Queue
import os
import logging

from omegaconf import DictConfig, OmegaConf
import numpy
from flwr.server import start_server
from flwr.server.strategy import FedAvg

from fl.federation.strategy import Checkpointer, MCFedAvg, MCFedAdagrad, MCFedAdam, MCQFedAvg, MlflowLogger, fit_round, on_evaluate_config_fn


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
        "min_fit_clients": len(strategy_params.nodes),
        "min_eval_clients": len(strategy_params.nodes),
        "min_available_clients": len(strategy_params.nodes)
    })
    if 'args' in strategy_params:
        args = OmegaConf.merge(default_args, strategy_params.args)
    else:
        args = default_args

    mlflow_logger = MlflowLogger(epochs_in_round)
    checkpointer = Checkpointer(checkpoint_dir)
    if strategy_params.name == 'fedavg':
        return MCFedAvg(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, on_evaluate_config_fn=on_evaluate_config_fn, **args)
    elif strategy_params.name == 'qfedavg':
        return MCQFedAvg(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, on_evaluate_config_fn=on_evaluate_config_fn, **args)
    elif strategy_params.name ==  'fedadam':
        return MCFedAdam(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, on_evaluate_config_fn=on_evaluate_config_fn, **args)
    elif strategy_params.name == 'fedadagrad':
        return MCFedAdagrad(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, on_evaluate_config_fn=on_evaluate_config_fn, **args)
    else:
        raise ValueError(f'Strategy name {strategy_params.name} should be one of the ["fedavg", "qfedavg", "fedadam", "fedadagrad"]')


class Server(Process):
    def __init__(self, log_dir: str, queue: Queue, params_hash: str, cfg_path: str, **kwargs):
        """Process for running flower server

        Args:
            log_dir (str): Directory where server.log will be created
            queue (Queue): Queue for communication between processes
            params_hash (str): Parameters hash for choosing the directory for saving model checkpoints
            cfg_path (str): Path to full yaml config file
        """        
        Process.__init__(self, **kwargs)
        self.log_dir = log_dir
        self.queue = queue
        self.params_hash = params_hash
        null_node = OmegaConf.from_dotlist([f'node.index=null'])
        self.cfg = OmegaConf.merge(null_node, OmegaConf.load(cfg_path))

    def _configure_logging(self):
        logging.basicConfig(filename=os.path.join(self.log_dir, f'server.log'), level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')

    def run(self) -> None:
        """Runs flower server federated learning training
        """        
        self._configure_logging()
        strategy = get_strategy(self.cfg.server.strategy, 
                                self.cfg.node.scheduler.epochs_in_round, 
                                os.path.join(self.cfg.server.checkpoint_dir, self.params_hash))
        start_server(
                    server_address="[::]:8080",
                    strategy=strategy,
                    config={"num_rounds": self.cfg.server.rounds},
                    force_final_distributed_eval=True
        )
    
        strategy.checkpointer.copy_best_model(os.path.join(self.cfg.server.checkpoint_dir, self.params_hash, 'best_model.ckpt'))