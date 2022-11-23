import json
import pickle
import shutil
from threading import local
from typing import List, Tuple, Optional, Dict, cast
import logging
import numpy
from numpy.linalg import norm
import os
import mlflow
import plotly.graph_objects as go

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
from nn.utils import ClfFederatedMetrics, ClfMetrics, LassoNetRegMetrics, Metrics, RegFederatedMetrics
from utils.landscape import add_beta_to_loss_landscape


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
        known_model_types = ['lassonet_regressor', 'mlp_regressor', 'mlp_classifier', 'linear_regressor']
        if model_type not in known_model_types:
            raise ValueError(f'model_type should be one of the {known_model_types} and not {model_type}')

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
        logging.info(f'We will save checkpoints to {self.checkpoint_dir}')
        self.history = []
        self.best_metrics = None
        self.last_metrics = None

    def add_loss_to_history(self, metrics: Metrics) -> None:
        logging.info(f'adding metrics {metrics} to history')
        self.history.append(metrics.val_loss)

    def set_best_metrics(self, metrics: Metrics) -> None:
        if self.history[-1] == min(self.history):
            self.best_metrics = metrics
            
    def _clear_old_checkpoints(self, start_rnd: int, current_rnd: int) -> None:
        best_rnd = numpy.argmin(self.history)
        for old_rnd in range(start_rnd, current_rnd):
            if old_rnd != best_rnd:
                os.remove(os.path.join(self.checkpoint_dir, f'round-{old_rnd}.ckpt.npz'))

    def save_checkpoint(self, rnd: int, aggregated_parameters: Parameters) -> None:
        """Saves current round weights to {self.checkpoint_dir}/round-{rnd}.ckpt
        Removes old checkpoint with loss worse then minimum each 10 rounds

        Args:
            rnd (int): Current FL round
            aggregated_parameters (Parameters): Aggregated model parameters
        """        
        if rnd != 0 and rnd % 10 == 0:
            self._clear_old_checkpoints(max(1, rnd - 10), rnd)
            
        if aggregated_parameters is not None:
            aggregated_weights = parameters_to_weights(aggregated_parameters)
            numpy.savez(os.path.join(self.checkpoint_dir, f'round-{rnd}.ckpt'), *aggregated_weights)
        else:
            pass
    
    def load_best_parameters(self) -> Parameters:
        best_rnd = numpy.argmin(self.history)
        logging.info(f'history is {self.history}')
        logging.info(f'best round: {best_rnd}\tval_loss: {self.history[best_rnd]}')
        weights = list(numpy.load(os.path.join(self.checkpoint_dir, f'round-{best_rnd}.ckpt.npz')).values())
        return weights_to_parameters(weights)
    
    def copy_best_model(self, best_model_path: str):
        best_rnd = numpy.argmin(self.history)
        shutil.copy2(os.path.join(self.checkpoint_dir, f'round-{best_rnd}.ckpt.npz'), best_model_path)


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
        if isinstance(metric_list[0], ClfMetrics):
            fed_metrics = ClfFederatedMetrics(metric_list, rnd)
        else:
            fed_metrics = RegFederatedMetrics(metric_list, rnd)
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
            self.current_weights = parameters_to_weights(parameters)
            self.weights_inited_properly = True
            logging.info(f'weights were initialized')
        else:
            logging.info(f'this strategy does not require weights initialization')
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
            self.current_weights = parameters_to_weights(parameters)
            self.weights_inited_properly = True
            logging.info(f'weights were initialized')
        else:
            logging.info(f'this strategy does not require weights initialization')
        return cen_ev

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None


class Scaffold(FedAvg):
    def __init__(self, K: int = 1, local_lr: float = 0.01, global_lr: float = 0.1, grad_lr: float = 1.0, 
                 local_gamma: float = 1.0, epochs_in_round: int = 1, batch_size: int = 64,
                 **kwargs) -> None:
        """FL strategy for heterogenious data which uses control variates for adjusting gradients on nodes.
        c_global variate is an estimation of true gradient of all clients.
        c_local variate is an estimation of node gradient.
        Scaffold corrects local update steps to the direction of global gradient, adding c_global - c_local to gradient data before backprop.
        https://arxiv.org/abs/1910.06378

        Args:
            K (int, optional): Number of training steps (gradient updates) taken by each local model. Defaults to 1.
            local_lr (float, optional): Local learning rate for gradient updates. Defaults to 0.01.
            global_lr (float, optional): Global learning rate for updating global model weights after aggregation. Defaults to 0.1.
        """        
        self.K = K
        self.local_lr = local_lr
        self.global_lr = global_lr
        self.grad_lr = grad_lr
        self.local_gamma = local_gamma
        self.epochs_in_round = epochs_in_round
        self.batch_size = batch_size
        self.c_global = None
        self.old_weights = None
        super().__init__(**kwargs)

    def _plot_2d_landscape(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]]):
        local_betas = [bytes_to_weights(res.metrics['local_beta'])[0] for _, res in results]
        true_betas = [bytes_to_weights(res.metrics['true_beta'])[0] for _, res in results]
        beta_grids = [bytes_to_weights(res.metrics['beta_grid'])[0] for _, res in results]
        
        # print(Z[90:, 20:30])
        # global loss landscape
        beta_grid = sum(beta_grids) / len(beta_grids)
        fig = go.Figure()
        points_num = 100
        beta_space = numpy.linspace(-1, 1, num=points_num, endpoint=True)
        fig.add_trace(go.Contour(
                      z=beta_grid, 
                      x=beta_space, # horizontal axis
                      y=beta_space, # vertical axis,
                      contours=dict(start=numpy.nanmin(beta_grid), end=numpy.nanmax(beta_grid), size=0.1)
        ))

        for i, (local_beta, true_beta) in enumerate(zip(local_betas, true_betas)):
            add_beta_to_loss_landscape(fig, true_beta, local_beta, f'SGD_{i}')
        fig.add_trace(go.Scatter(x=[self.c_global[0][0, 0]], y=[self.c_global[0][0, 1]], mode='markers', name=f'c_global'))
        
        mlflow.log_figure(fig, f'global_loss_landscape_rnd_{rnd}.png')
        
    def _get_mean_local_lr(self, rnd: int):
        assert rnd > 0
        start_lr = self.local_lr*self.local_gamma**((rnd - 1)*self.epochs_in_round)
        end_lr = self.local_lr*self.local_gamma**(rnd*self.epochs_in_round - 1)
        mean_lr = numpy.mean(numpy.geomspace(start_lr, end_lr))
        return mean_lr
    
    def _get_batch_counts(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[int]:
        counts = []
        for _, result in results:
            batches_per_epoch = result.num_examples // self.batch_size + (result.num_examples % self.batch_size > 0)
            count = batches_per_epoch * self.epochs_in_round
            counts.append(self.K) # temporary, for debugging purposes
        return counts

    def aggregate_fit(
        self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # List[List[numpy.ndarray]]
        # self._plot_2d_landscape(rnd, results)
        client_weights = [parameters_to_weights(fit_res.parameters) for _, fit_res in results]

        mean_local_lr = self._get_mean_local_lr(rnd)
        batch_counts = self._get_batch_counts(results)
        logging.info(f'batch_counts are {batch_counts}')
        logging.info(f'mean_local_lr is {mean_local_lr} for rnd {rnd} and epochs in round {self.epochs_in_round}')
        
        for layer_index, old_layer_weight in enumerate(self.old_weights):
            cg_layer = self.c_global[layer_index]
            c_deltas = [-cg_layer + (1/(bc*mean_local_lr))*(old_layer_weight - cw[layer_index]) \
                for bc, cw in zip(batch_counts, client_weights)]
            c_avg_delta = sum(c_deltas)/len(client_weights)
            # logging.info(f'AGGREGATE_FIT: {layer_index}\tcg_layer: {norm(cg_layer):.4f}\tc_avg_delta: {norm(c_avg_delta):.4f}')
            for i, cw in enumerate(client_weights):
                logging.debug(f'client: {i}\t{norm(old_layer_weight - cw[layer_index]):.4f}')
            
            self.c_global[layer_index] = cg_layer + c_avg_delta

        new_weights = aggregate([(cw, bc) for bc, cw in zip(batch_counts, client_weights)])
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
            self.old_weights = parameters_to_weights(parameters)
            self.c_global = [numpy.zeros_like(ow) for ow in self.old_weights]
            self.weights_inited_properly = True
            logging.info(f'weights were initialized')
        else:
            logging.info(f'this strategy does not require weights initialization')
        return cen_ev

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None