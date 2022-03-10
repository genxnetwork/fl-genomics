from collections import namedtuple
from time import sleep
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

    config_path = snakemake.input['config']
    params_hash = snakemake.wildcards['params_hash']
    
    cfg = OmegaConf.load(config_path)
    
    strategy = MlflowStrategy(
        fraction_fit=0.99,
        fraction_eval=0.99,
        min_fit_clients = cfg.server.node_count,
        min_eval_clients = cfg.server.node_count,
        min_available_clients = cfg.server.node_count,
        on_fit_config_fn=fit_round
    )

    with mlflow.start_run(
        tags={
            'description': cfg.experiment.description,
            'params_hash': params_hash,
            'hostname': gethostname()
        }
    ) as run:

        start_server(
                    server_address="[::]:8080",
                    strategy=strategy,
                    config={"num_rounds": cfg.rounds}
        )
        