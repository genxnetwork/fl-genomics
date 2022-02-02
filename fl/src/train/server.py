from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Dict
import mlflow


@hydra.main(config_path='configs/server', config_name='default')
def main(cfg: DictConfig):

    with mlflow.start_run(
        tags={
            'description': cfg.description
        }
    ) as run:
        print(run.info.run_id)


if __name__ == '__main__':
    main()