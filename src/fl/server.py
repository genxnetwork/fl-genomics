from collections import namedtuple
import logging
import os
from omegaconf import OmegaConf
import mlflow
from socket import gethostname
from flwr.server import start_server

from federation.strategy import MlflowStrategy, fit_round


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
    
    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    cfg = OmegaConf.load(config_path)
    
    strategy = MlflowStrategy(
        fraction_fit=0.99,
        fraction_eval=0.99,
        min_fit_clients = cfg.server.node_count,
        min_eval_clients = cfg.server.node_count,
        min_available_clients = cfg.server.node_count,
        on_fit_config_fn=fit_round
    )

    logging.info(f'starting to print os environ')
    for key, value in os.environ.items():
        logging.info(f'{key}: {value};')
    experiment = mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        tags={
            'description': cfg.experiment.description,
            'params_hash': params_hash,
            'hostname': gethostname()
        }
    ) as run:

        start_server(
                    server_address="[::]:8080",
                    strategy=strategy,
                    config={"num_rounds": cfg.server.rounds}
        )
        