from multiprocessing import Process, Queue
import os
import logging

from omegaconf import DictConfig, OmegaConf
import numpy
from flwr.server import start_server
from flwr.server.strategy import FedAvg

from fl.federation.strategy import Checkpointer, MCFedAvg, MCFedAdagrad, MCFedAdam, MCQFedAvg, MCScaffold, MlflowLogger, fit_round, on_evaluate_config_fn


def get_strategy(strategy_params: DictConfig, epochs_in_round: int, node_count: int, checkpoint_dir: str, model_type: str) -> FedAvg:
    """Creates flwr Strategy from strategy configs entry

    Args:
        strategy_params (DictConfig): Strategy params with name, node_count and optional args dict
        epochs_in_round (int): Number of epochs in each federation round
        node_count (int): Number of nodes to federate
        checkpoint_dir (str): Directory for saving model checkpoints

    Raises:
        ValueError: If strategy_params.name is unknown

    Returns:
        FedAvg: flwr Strategy with checkpointing and mlflow logging capabilities
    """    
    default_args = OmegaConf.create({
        "fraction_fit": 0.9,
        "fraction_eval": 0.9,
        "min_fit_clients": 16,
        "min_eval_clients": 16,
        "min_available_clients": node_count
    })
    if 'args' in strategy_params:
        args = OmegaConf.merge(default_args, strategy_params.args)
    else:
        args = default_args

    logging.info(f'strategy args: {args}')

    mlflow_logger = MlflowLogger(epochs_in_round, model_type)
    checkpointer = Checkpointer(checkpoint_dir)
    if strategy_params.name == 'fedavg':
        return MCFedAvg(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name == 'qfedavg':
        return MCQFedAvg(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name ==  'fedadam':
        return MCFedAdam(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name == 'fedadagrad':
        return MCFedAdagrad(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    elif strategy_params.name == 'scaffold':
        return MCScaffold(mlflow_logger, checkpointer, on_fit_config_fn=fit_round, **args)
    else:
        raise ValueError(f'Strategy name {strategy_params.name} should be one of the ["fedavg", "qfedavg", "fedadam", "fedadagrad", "scaffold"]')


class Server(Process):
    def __init__(self, log_dir: str, queue: Queue, params_hash: str, cfg: DictConfig, **kwargs):
        """Process for running flower server

        Args:
            log_dir (str): Directory where server.log will be created
            queue (Queue): Queue for communication between processes
            params_hash (str): Parameters hash for choosing the directory for saving model checkpoints
            cfg_path (str): Path to full yaml configs file
        """        
        Process.__init__(self, **kwargs)
        self.log_dir = log_dir
        self.queue = queue
        self.params_hash = params_hash
        null_node = OmegaConf.from_dotlist([f'node.index=null'])
        self.cfg = OmegaConf.merge(null_node, cfg)

    def _configure_logging(self):
        logging.basicConfig(filename=os.path.join(self.log_dir, f'server.log'), level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')

    def run(self) -> None:
        """Runs flower server federated learning training
        """        
        self._configure_logging()
        node_count = len(self.cfg.split.nodes) if self.cfg.strategy.active_nodes == 'all' else len(self.cfg.strategy.active_nodes)
        strategy = get_strategy(self.cfg.strategy, 
                                self.cfg.scheduler.epochs_in_round, 
                                node_count,
                                os.path.join(self.cfg.server.checkpoint_dir, self.params_hash),
                                self.cfg.model.name)
        print(f'SERVER LOGGING CONFIGURED')
        logging.info(f'SERVER LOGGING CONFIGURED')
        start_server(
                    server_address=f"[::]:{self.cfg.server.port}",
                    strategy=strategy,
                    config={"num_rounds": self.cfg.server.rounds},
                    force_final_distributed_eval=True
        )
    
        strategy.checkpointer.copy_best_model(os.path.join(self.cfg.server.checkpoint_dir, self.params_hash, 'best_model.ckpt'))