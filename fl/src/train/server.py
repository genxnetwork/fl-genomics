from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
import mlflow
import flwr
import socket
 
from federation.strategy import MlflowStrategy, fit_round


def write_run_id(run_id: str):
    with open(to_absolute_path('.mlflow_parent_run_id'), 'w') as pri_file:
        pri_file.write(f'MLFLOW_PARENT_RUN_ID={run_id}')
        print(f'PARENT RUN ID was written to {to_absolute_path(".mlflow_parent_run_id")}')

# Function to display hostname and
# IP address
def write_hostname():
    host_name = socket.gethostname()
    with open(to_absolute_path('.server_hostname'), 'w') as hn_file:
        hn_file.write(f'FLWR_SERVER_HOSTNAME={host_name}')    


@hydra.main(config_path='../configs/server', config_name='default')
def main(cfg: DictConfig):
    # print(os.environ)
    strategy = MlflowStrategy(
        fraction_fit=0.99,
        fraction_eval=0.99,
        min_fit_clients = cfg.node_count,
        min_eval_clients = cfg.node_count,
        min_available_clients = cfg.node_count,
        on_fit_config_fn=fit_round
    )

    with mlflow.start_run(
        tags={
            'description': cfg.experiment.description
        }
    ) as run:

        print(run.info.run_id)
        write_run_id(run.info.run_id)
        write_hostname()

        flwr.server.start_server(
                        server_address="[::]:8080",
                        strategy=strategy,
                        config={"num_rounds": cfg.rounds}
        )


if __name__ == '__main__':
    main()