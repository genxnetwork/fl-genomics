from collections import namedtuple
import logging
import os
from omegaconf import OmegaConf
import mlflow
from socket import gethostname
from flwr.server import start_server

from federation.strategy import MCFedAvg, fit_round


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
    
    strategy = MCFedAvg(
        checkpoint_dir,
        fraction_fit=0.99,
        fraction_eval=0.99,
        min_fit_clients = cfg.server.node_count,
        min_eval_clients = cfg.server.node_count,
        min_available_clients = cfg.server.node_count,
        on_fit_config_fn=fit_round
    )

    experiment = mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        tags={
            'description': cfg.experiment.description,
            'params_hash': params_hash,
            'hostname': gethostname()
        }
    ) as run:
        mlflow.log_params(cfg.server)
        logging.info(f'starting server with min clients {cfg.server.node_count} and {cfg.server.rounds} max rounds')
        start_server(
                    server_address="[::]:8080",
                    strategy=strategy,
                    config={"num_rounds": cfg.server.rounds}
        )
    
    best_model_path = snakemake.output[0]
    strategy.checkpointer.copy_best_model(best_model_path)
        