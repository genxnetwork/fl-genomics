import pickle
import shutil
from typing import List, Tuple, Optional, Dict
import logging
import mlflow
import numpy
import os

from flwr.common import (
    EvaluateRes,
    EvaluateIns,
    Scalar,
    Weights,
    Parameters,
    FitRes,
    parameters_to_weights,
    weights_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, QFedAvg
from flwr.server.client_manager import ClientManager

from nn.utils import Metrics, RegFederatedMetrics


def fit_round(rnd: int):
    """Send round number to client."""
    return {'current_round': rnd}


def on_evaluate_config_fn(rnd: int):
    return {'current_round': rnd}


RESULTS = List[Tuple[ClientProxy, EvaluateRes]]


class MlflowLogger:
    def __init__(self, epochs_in_round: int, model_type: str) -> None:
        """Logs server-side per-round metrics to mlflow
        """        
        self.epochs_in_round = epochs_in_round
        if model_type not in ['lassonet_regressor', 'mlp_regressor']:
            raise ValueError(f'model_type should be one of the lassonet_regressor, mlp_regressor and not {model_type}')

        self.model_type = model_type

    def _get_reduction_type(self, rnd: int) -> str:
        if self.model_type == 'lassonet_regressor':
            return 'mean' if rnd == -1 else 'lassonet_best'
        else:
            return 'mean' 

    def log_losses(self, rnd: int, results: RESULTS) -> Metrics:
        """Logs val_loss, val and train r^2 to mlflow

        Args:
            rnd (int): current FL round
            results (RESULTS): Central model evaluation results from server

        Returns:
            Metrics: Metrics reduced over clients
        """       
        metric_list = [pickle.loads(r[1].metrics['metrics']) for r in results]
        fed_metrics = RegFederatedMetrics(metric_list, rnd*self.epochs_in_round)
        # LassoNetRegMetrics averaged by clients axis, i.e. one aggregated metric value for each alpha value
        # Other metrics are averaged by client axis but have only one value in total for train, val, test datasets
        avg_metrics = fed_metrics.reduce(reduction=self._get_reduction_type(rnd))
        # logging.info(avg_metrics)
        logging.info(f'round {rnd}\t' + str(avg_metrics))
        avg_metrics.log_to_mlflow()

        if rnd != -1:
            lassonet_best_metrics = avg_metrics.reduce('mean')
            logging.info(f'logging final centralized evaluation results')
            logging.info(f'round: {rnd}\t' + str(lassonet_best_metrics))

        return avg_metrics


class Checkpointer:
    def __init__(self, checkpoint_dir: str) -> None:
        """Saves best model checkpoints to {checkpoint_dir}

        Args:
            checkpoint_dir (str): Dir for saving model checkpoints
        """    
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.history = []

    def add_loss_to_history(self, loss: float) -> None:
        self.history.append(loss)

    def save_checkpoint(self, rnd: int, aggregated_parameters: Parameters) -> None:
        """Checks if current model has minimum loss and if so, saves it to 'best_temp_model.ckpt'

        Args:
            rnd (int): Current FL round
            aggregated_parameters (Parameters): Aggregated model parameters
        """        
        if aggregated_parameters is not None and len(self.history) > 0 and self.history[-1] == min(self.history):
            # Save aggregated_weights
            aggregated_weights = parameters_to_weights(aggregated_parameters)
            print(f"round {rnd}\tmin_val_loss: {self.history[-1]:.2f}\tsaving_checkpoint to {self.checkpoint_dir}")
            numpy.savez(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt'), *aggregated_weights)
        else:
            pass
    
    def load_best_parameters(self) -> Parameters:
        dct = numpy.load(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt.npz'))
        print(f'loading best parameters')
        for key, value in dct.items():
            print(key, value.shape)
        weights = list(numpy.load(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt.npz')).values())
        return weights_to_parameters(weights)
    
    def copy_best_model(self, best_model_path: str):
        shutil.copy2(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt.npz'), best_model_path)


class MCMixin:
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        """Mixin for strategies which can log metrics and save checkpoints

        Args:
            mlflow_logger (MlflowLogger): Mlflow Logger
            checkpointer (Checkpointer): Model checkpoint saver
        """        
        self.mlflow_logger = mlflow_logger
        self.checkpointer = checkpointer
        kwargs['on_evaluate_config_fn'] = self.on_evaluate_config_fn_closure
        super().__init__(**kwargs)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: RESULTS,
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None
        metrics = self.mlflow_logger.log_losses(rnd, results)
        reduced_metrics = metrics.reduce(self.mlflow_logger._get_reduction_type(rnd)) 
        self.checkpointer.add_loss_to_history(reduced_metrics.val_loss)
        self.checkpointer.last_metrics = metrics
        return super().aggregate_evaluate(rnd, results, failures)
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        self.checkpointer.save_checkpoint(rnd, aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if rnd == -1:
            print(f'loading best parameters for final evaluation')
            parameters = self.checkpointer.load_best_parameters()
        return super().configure_evaluate(rnd, parameters, client_manager)

    def on_evaluate_config_fn_closure(self, rnd: int):
        if rnd == -1:
            return {'current_round': rnd, 'best_col': self.checkpointer.last_metrics.best_col}
        else:
            return {'current_round': rnd}


class MCFedAvg(MCMixin,FedAvg):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        super().__init__(mlflow_logger, checkpointer, **kwargs)


class MCQFedAvg(MCMixin,QFedAvg):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        super().__init__(mlflow_logger, checkpointer, **kwargs)


class MCFedAdagrad(MCMixin,FedAdagrad):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        super().__init__(mlflow_logger, checkpointer, **kwargs)


class MCFedAdam(MCMixin,FedAdam):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        super().__init__(mlflow_logger, checkpointer, **kwargs)
