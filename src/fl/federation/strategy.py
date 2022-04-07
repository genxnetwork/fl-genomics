import shutil
from typing import List, Tuple, Optional, Dict
import logging
import mlflow
import numpy
import os

from flwr.common import (
    EvaluateRes,
    Scalar,
    Weights,
    Parameters,
    FitRes,
    parameters_to_weights
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, QFedAvg


def fit_round(rnd: int):
    """Send round number to client."""
    return {'current_round': rnd}


def on_evaluate_config_fn(rnd: int):
    return {'current_round': rnd}


RESULTS = List[Tuple[ClientProxy, EvaluateRes]]


class MlflowLogger:
    def __init__(self, epochs_in_round: int) -> None:
        """Logs server-side per-round metrics to mlflow
        """        
        self.epochs_in_round = epochs_in_round

    def _calculate_agg_metric(self, metric_name: str, results: RESULTS) -> float:
        """Calculates weighted average {metric_name} from {results}

        Args:
            metric_name (str): Name of metric in {results}
            results (RESULTS): List of client's client proxies and current round metrics

        Returns:
            float: averaged metric
        """        
        losses = [r.metrics[metric_name] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        return sum(losses) / sum(examples)

    def log_losses(self, rnd: int, results: RESULTS) -> float:
        """Logs val_loss, val and train r^2 to mlflow

        Args:
            rnd (int): current FL round
            results (RESULTS): Central model evaluation results from server

        Returns:
            float: Current validation loss
        """        
        train_loss = self._calculate_agg_metric('train_loss', results)
        val_loss = self._calculate_agg_metric('val_loss', results)
        train_r2 = self._calculate_agg_metric('train_r2', results)
        val_r2 = self._calculate_agg_metric('val_r2', results)
        logging.info(f"round {rnd}\ttrain_loss: {train_loss:.4f}\ttrain_r2: {train_r2:.4f}\tval_loss: {val_loss:.2f}\tval_r2: {val_r2:.4f}")
        mlflow.log_metric('train_loss', train_loss, step=rnd*self.epochs_in_round)
        mlflow.log_metric('val_loss', val_loss, step=rnd*self.epochs_in_round)
        mlflow.log_metric('train_r2', train_r2, step=rnd*self.epochs_in_round)
        mlflow.log_metric('val_r2', val_r2, step=rnd*self.epochs_in_round)
        return val_loss


class Checkpointer:
    def __init__(self, checkpoint_dir: str) -> None:
        """Saves best model checkpoints to {checkpoint_dir}

        Args:
            checkpoint_dir (str): Dir for saving model checkpoints
        """        
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
            logging.info(f"round {rnd}\tmin_val_loss: {self.history[-1]:.2f}\tsaving_checkpoint to {self.checkpoint_dir}")
            numpy.savez(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt'), *aggregated_weights)
        else:
            pass
    
    def copy_best_model(self, best_model_path: str):
        shutil.copy2(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt'), best_model_path)


class MCMixin:
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        """Mixin for strategies which can log metrics and save checkpoints

        Args:
            mlflow_logger (MlflowLogger): Mlflow Logger
            checkpointer (Checkpointer): Model checkpoint saver
        """        
        self.mlflow_logger = mlflow_logger
        self.checkpointer = checkpointer
        super().__init__(**kwargs)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: RESULTS,
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None
        val_loss = self.mlflow_logger.log_losses(rnd, results)
        self.checkpointer.add_loss_to_history(val_loss)
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
