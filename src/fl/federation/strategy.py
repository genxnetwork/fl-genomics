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
    FitRes,
    parameters_to_weights
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def fit_round(rnd: int):
    """Send round number to client."""
    return {'current_round': rnd}


def on_evaluate_config_fn(rnd: int):
    return {'current_round': rnd}


RESULTS = List[Tuple[ClientProxy, EvaluateRes]]


class MlflowStrategy(FedAvg):

    def __init__(self, checkpoint_dir: str, **kwargs) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.history = []
        super().__init__(**kwargs)

    def _calculate_agg_metric(self, metric_name: str, results: RESULTS) -> float:
        losses = [r.metrics[metric_name] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        return sum(losses) / sum(examples)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: RESULTS,
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh val_loss of each client by number of examples used
        val_loss = self._calculate_agg_metric('val_loss', results)
        self.history.append(val_loss)
        train_r2 = self._calculate_agg_metric('train_r2', results)
        val_r2 = self._calculate_agg_metric('val_r2', results)
        logging.info(f"round {rnd}\ttrain_r2: {train_r2:.4f}\tval_r2: {val_r2:.4f}\tval_loss: {val_loss:.2f}")
        mlflow.log_metric('val_loss', val_loss, step=rnd)
        mlflow.log_metric('train_r2', train_r2, step=rnd)
        mlflow.log_metric('val_r2', val_r2, step=rnd)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None and len(self.history) > 0 and self.history[-1] == min(self.history):
            # Save aggregated_weights
            aggregated_weights = parameters_to_weights(aggregated_parameters)
            logging.info(f"round {rnd}\tmin_val_loss: {self.history[-1]:.2f}\tsaving_checkpoint to {self.checkpoint_dir}")
            numpy.savez(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt'), *aggregated_weights)
        return aggregated_parameters, aggregated_metrics

    def copy_best_model(self, best_model_path: str):
        shutil.copy2(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt'), best_model_path)