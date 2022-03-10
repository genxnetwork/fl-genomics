from typing import List, Tuple, Optional, Dict
import logging
import mlflow

from flwr.common import (
    EvaluateRes,
    Scalar,
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

    def __init__(self, **kwargs) -> None:
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
        train_loss = self._calculate_agg_metric('train_loss', results)
        logging.info(f"round {rnd}\ttrain_loss: {train_loss:.4f}\tval_loss: {val_loss:.4f}")
        mlflow.log_metric('agg_val_loss', val_loss, step=rnd)
        mlflow.log_metric('agg_train_loss', train_loss, step=rnd)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)