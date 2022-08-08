import json
import pickle
import shutil
from threading import local
from typing import List, Tuple, Optional, Dict, cast
import logging
import numpy
import os
import mlflow

from flwr.common import (
    EvaluateRes,
    EvaluateIns,
    Scalar,
    Weights,
    Parameters,
    FitRes,
    FitIns,
    parameters_to_weights,
    weights_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, QFedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager

from fl.federation.utils import weights_to_bytes, bytes_to_weights
from nn.utils import ClfFederatedMetrics, LassoNetRegMetrics, Metrics, RegFederatedMetrics


def fit_round(rnd: int):
    """Send round number to client."""
    return {'current_round': rnd}


def on_evaluate_config_fn(rnd: int):
    return {'current_round': rnd}


RESULTS = List[Tuple[ClientProxy, EvaluateRes]]


class StrategyLogger:
    def log_losses(self, rnd: int, metrics: Metrics) -> None:
        pass

    def log_weights(self, rnd: int, layers: List[str], weights: List[Weights], aggregated_weights: Weights) -> None:
        pass


class MlflowLogger(StrategyLogger):
    def __init__(self, epochs_in_round: int, model_type: str) -> None:
        """Logs server-side per-round metrics to mlflow
        """        
        self.epochs_in_round = epochs_in_round
        if model_type not in ['lassonet_regressor', 'mlp_regressor', 'mlp_classifier']:
            raise ValueError(f'model_type should be one of the lassonet_regressor, mlp_regressor and not {model_type}')

        self.model_type = model_type

    def log_losses(self, rnd: int, metrics: Metrics) -> None:
        """Logs val_loss, val and train r^2 or auc to mlflow

        Args:
            rnd (int): current FL round
            metrics (Metrics): Metrics to log to mlflow

        Returns:
            Metrics: Metrics reduced over clients
        """       
        # logging.info(avg_metrics)
        logging.info(f'round {rnd}\t' + str(metrics))
        metrics.log_to_mlflow()

    def log_weights(self, rnd: int, layers: List[str], weights: List[Weights], aggregated_weights: Weights) -> None:
        pass
        

class Checkpointer:
    def __init__(self, checkpoint_dir: str) -> None:
        """Saves best model checkpoints to {checkpoint_dir}

        Args:
            checkpoint_dir (str): Dir for saving model checkpoints
        """    
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.history = []
        self.best_metrics = None
        self.last_metrics = None

    def add_loss_to_history(self, metrics: Metrics) -> None:
        self.history.append(metrics.val_loss)

    def set_best_metrics(self, metrics: Metrics) -> None:
        if self.history[-1] == min(self.history):
            self.best_metrics = metrics

    def save_checkpoint(self, rnd: int, aggregated_parameters: Parameters) -> None:
        """Checks if current model has minimum loss and if so, saves it to 'best_temp_model.ckpt'

        Args:
            rnd (int): Current FL round
            aggregated_parameters (Parameters): Aggregated model parameters
        """        
        if aggregated_parameters is not None:
            if len(self.history) == 0 or (len(self.history) > 0 and self.history[-1] == min(self.history)):
                # Save aggregated_weights
                aggregated_weights = parameters_to_weights(aggregated_parameters)
                # print(f"round {rnd}\tmin_val_loss: {self.history[-1]:.2f}\tsaving_checkpoint to {self.checkpoint_dir}")
                numpy.savez(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt'), *aggregated_weights)
        else:
            pass
    
    def load_best_parameters(self) -> Parameters:
        weights = list(numpy.load(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt.npz')).values())
        return weights_to_parameters(weights)
    
    def copy_best_model(self, best_model_path: str):
        shutil.copy2(os.path.join(self.checkpoint_dir, f'best_temp_model.ckpt.npz'), best_model_path)


class MCMixin:
    def __init__(self, strategy_logger: StrategyLogger, checkpointer: Checkpointer, **kwargs) -> None:
        """Mixin for strategies which can log metrics and save checkpoints

        Args:
            strategy_logger (StrategyLogger): Logger for logging metrics to mlflow, neptune or stdout
            checkpointer (Checkpointer): Model checkpoint saver
        """        
        self.strategy_logger = strategy_logger
        self.checkpointer = checkpointer
        kwargs['on_evaluate_config_fn'] = self.on_evaluate_config_fn_closure
        super().__init__(**kwargs)

    def _aggregate_metrics(self, rnd: int, results: RESULTS) -> Metrics:

        metric_list = [pickle.loads(r[1].metrics['metrics']) for r in results]
        fed_metrics = RegFederatedMetrics(metric_list, rnd*self.strategy_logger.epochs_in_round)
        # LassoNetRegMetrics averaged by clients axis, i.e. one aggregated metric value for each alpha value
        # Other metrics are averaged by client axis but have only one value in total for train, val, test datasets
        avg_metrics = fed_metrics.reduce()
        return avg_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: RESULTS,
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        if not results:
            return None

        metrics = self._aggregate_metrics(rnd, results)
        self.strategy_logger.log_losses(rnd, metrics)
        reduced_metrics = metrics.reduce() 
        self.checkpointer.add_loss_to_history(reduced_metrics)
        # unreduced metrics because in case of lassonet_regressor we need best_col attribute
        self.checkpointer.set_best_metrics(metrics)
        self.checkpointer.last_metrics = metrics
        return super().aggregate_evaluate(rnd, results, failures)
    
    def _metrics_to_layer_names(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[str]:
        metrics = results[0][1].metrics
        return json.loads(str(metrics['layers'], encoding='utf-8'))

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        self.checkpointer.save_checkpoint(rnd, aggregated_parameters)

        client_weights = [parameters_to_weights(res[1].parameters) for res in results]
        # layer_names = self._metrics_to_layer_names(results)
        aggregated_weights = parameters_to_weights(aggregated_parameters)
        # self.strategy_logger.log_weights(rnd, layer_names, client_weights, aggregated_weights)

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if rnd == -1:
            # print(f'loading best parameters for final evaluation')
            parameters = self.checkpointer.load_best_parameters()
        print(f'DEBUG: starting to configure eval')
        return super().configure_evaluate(rnd, parameters, client_manager)

    def on_evaluate_config_fn_closure(self, rnd: int):
        return {'current_round': rnd}


class MCFedAvg(MCMixin,FedAvg):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        super().__init__(mlflow_logger, checkpointer, **kwargs)


class MCQFedAvg(MCMixin,QFedAvg):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        super().__init__(mlflow_logger, checkpointer, **kwargs)

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        cen_ev = super().evaluate(parameters)
        if cen_ev is not None:
            return cen_ev
        # aggregate_fit in QFedAvg needs a previous loss value to calculate lipschitz constants
        # flwr implementation evaluates old weights in centralized mode using evaluate function impl from fedavg
        # evaluate in fedavg works if and only if eval_fn init parameter was provided
        # instead of centralized evaluation we pull loss from checkpointer loss history
        return (self.checkpointer.history[-1], {}) if len(self.checkpointer.history) > 0 else (100, {})


class MCFedAdagrad(MCMixin,FedAdagrad):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        initial_parameters = weights_to_parameters([numpy.zeros((1,1))])
        self.weights_inited_properly = False
        super().__init__(mlflow_logger, checkpointer, initial_parameters=initial_parameters, **kwargs)

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        cen_ev = super().evaluate(parameters)
        # fedadam calculates updates using old and new weights
        # we need to set initial weights
        # evaluate called by server after getting random initial parameters from one of the clients
        # we set those parameters as initial weights
        if not self.weights_inited_properly:
            print(f'initializing weights!!!')
            self.current_weights = parameters_to_weights(parameters)
            self.weights_inited_properly = True
        else:
            print(f'we do not need to initialize weights')
        return cen_ev

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None


class MCFedAdam(MCMixin,FedAdam):
    def __init__(self, mlflow_logger: MlflowLogger, checkpointer: Checkpointer, **kwargs) -> None:
        initial_parameters = weights_to_parameters([numpy.zeros((1,1))])
        self.weights_inited_properly = False
        super().__init__(mlflow_logger, checkpointer, initial_parameters=initial_parameters, **kwargs)

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        cen_ev = super().evaluate(parameters)
        # fedadam calculates updates using old and new weights
        # we need to set initial weights
        # evaluate called by server after getting random initial parameters from one of the clients
        # we set those parameters as initial weights
        if not self.weights_inited_properly:
            print(f'initializing weights!!!')
            self.current_weights = parameters_to_weights(parameters)
            self.weights_inited_properly = True
        else:
            print(f'we do not need to initialize weights')
        return cen_ev

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None


class Scaffold(FedAvg):
    def __init__(self, K: int = 1, local_lr: float = 0.01, global_lr: float = 0.1, **kwargs) -> None:
        self.K = K
        self.local_lr = local_lr
        self.global_lr = global_lr
        self.c_global = None
        self.old_weights = None
        super().__init__(**kwargs)

    def aggregate_fit(
        self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # List[List[numpy.ndarray]]
        client_weights = [parameters_to_weights(fit_res.parameters) for _, fit_res in results]

        for layer_index, old_layer_weight in enumerate(self.old_weights):
            layer_diff = sum([(1/(self.K*self.local_lr))*(old_layer_weight - cw[layer_index]) for cw in client_weights])
            cg_layer = self.c_global[layer_index]
            # print(f'AGGREGATE_FIT: {layer_index}\tcg_layer: {cg_layer.shape}\tlayer_diff: {layer_diff.shape}')
            self.c_global[layer_index] = cg_layer + 1/len(client_weights)*(-cg_layer + layer_diff)

        new_weights = aggregate([(cw, 1) for cw in client_weights])
        for layer_index, (old_layer_weight, new_layer_weight) in enumerate(zip(self.old_weights, new_weights)):
            # print('weights aggregation: ', layer_index, type(old_layer_weight), type(new_layer_weight))
            # print(old_layer_weight.shape, new_layer_weight.shape)
            # if layer_index == 6:
            #     print(f'new_layer_weight is {new_layer_weight}')
            if not isinstance(new_layer_weight, numpy.ndarray):
                continue
                # new_layer_weight = numpy.array([new_layer_weight])
            self.old_weights[layer_index] += self.global_lr * (new_layer_weight - old_layer_weight)

        return weights_to_parameters(self.old_weights), {}

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        client_configs = super().configure_fit(rnd, parameters, client_manager)
        for client_proxy, fit_ins in client_configs:
            fit_ins.config['c_global'] = weights_to_bytes(self.c_global)
        return client_configs


class MCScaffold(MCMixin, Scaffold):
    def __init__(self, strategy_logger: StrategyLogger, checkpointer: Checkpointer, **kwargs) -> None:
        initial_parameters = weights_to_parameters([numpy.zeros((1,1))])
        self.weights_inited_properly = False
        super().__init__(strategy_logger, checkpointer, initial_parameters=initial_parameters, **kwargs)

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        cen_ev = super().evaluate(parameters)
        # scaffold calculates updates using old and new weights
        # we need to set initial weights
        # evaluate called by server after getting random initial parameters from one of the clients
        # we set those parameters as initial weights
        if not self.weights_inited_properly:
            print(f'initializing weights!!!')
            self.old_weights = parameters_to_weights(parameters)
            self.c_global = [numpy.zeros_like(ow) for ow in self.old_weights]
            self.weights_inited_properly = True
        else:
            print(f'we do not need to initialize weights')
        return cen_ev

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None