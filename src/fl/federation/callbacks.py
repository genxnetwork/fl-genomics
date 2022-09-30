from typing import Dict, OrderedDict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.core import LightningModule
import mlflow
import torch
import numpy
import plotly.graph_objects as go
from flwr.common import Weights, Scalar

from fl.federation.utils import ModuleParams, weights_to_bytes
from nn.models import BaseNet
from utils.landscape import mse_on_beta_grid, add_beta_to_loss_landscape


class ClientCallback:
    """Base class for FLClient callbacks
       Client will call on_before_fit before fitting a model in federated setting,
       on_after_fit after fitting and so on 
    """    
    def __init__(self):
        pass

    def on_before_fit(self, model: BaseNet):
        pass

    def on_after_fit(self, model: BaseNet) -> Dict[str, Scalar]:
        return {}

    def on_before_evaluate(self, model: BaseNet):
        pass

    def on_after_evaluate(self, model: BaseNet):
        pass


class PlotLandscapeCallback(ClientCallback):
    """FLCLient callback for drawing the loss landscape for linear model with two parameters as a contour plot
    """    
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray, beta: numpy.ndarray):
        """Create PlotLandscapeCallback with features, target and true model cofficients data
        The underlying true model is x*beta=y

        Args:
            x (numpy.ndarray): Features array, should have two columns
            y (numpy.ndarray): Target array
            beta (numpy.ndarray): True model coefficients array, should have two values
        """        
        super().__init__()
        self.x = x
        self.y = y
        self.beta = beta

    def on_after_fit(self, model: BaseNet) -> Dict[str, Scalar]:
        """Generates loss landscape using true coefficients and adds history of found coefficients as path on loss landscape plot

        Args:
            model (BaseNet): model with beta_history attribute which should be a list of numpy arrays of shape (2, 1)

        Returns:
            Dict[str, Scalar]: Dict with three items, true beta coefficients, local beta history and local loss landscape
        """        
        beta_space, Z = mse_on_beta_grid(self.x, self.y, beta_range=(-1, 1))
        # print(Z[90:, 20:30])
        fig = go.Figure()
        fig.add_trace(go.Contour(
                z=Z, # I don't know why we need T to work
                x=beta_space, # horizontal axis
                y=beta_space, # vertical axis,
                contours=dict(start=numpy.nanmin(Z), end=numpy.nanmax(Z), size=0.5)
        ))
        beta_history = numpy.array(model.beta_history).squeeze()
        add_beta_to_loss_landscape(fig, self.beta, beta_history, 'SGD')

        mlflow.log_figure(fig, 'loss_landscape.png')

        return {'true_beta': weights_to_bytes([self.beta]), 'local_beta': weights_to_bytes([beta_history]), 'beta_grid': weights_to_bytes([Z])}
        
        
class ScaffoldCallback(Callback):
    """PytorchLightning Callback for updating gradient data as required by Scaffold FL strategy before backpropagation

    """    
    c_global: ModuleParams
    c_local: ModuleParams

    def __init__(self, K: int = 1, log_grad=False, log_diff=False) -> None:
        """Init function

        Args:
            K (int, optional): Number of training steps, i.e. number of batches used for training a model. Defaults to 1.
            log_grad (bool, optional): If True, logs gradient l2 norm to mlflow. Defaults to False.
            log_diff (bool, optional): If True, logs l2 norm of difference between local and global varaites to mlflow. Defaults to False.
        """        
        self.K = K
        self.c_global = None
        self.c_local = None
        self.log_grad = log_grad
        self.log_diff = log_diff
        super().__init__()

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        params = pl_module.named_parameters()
        for name, v in params:
            # print(f'on_after_backward: {name}: {v.shape}, {v.grad}')
            if v.grad is not None and self.c_local is not None:
                v.grad.data += (self.c_global[name].to(v.grad.data.device) - self.c_local[name].to(v.grad.data.device))
                global_fl_step = trainer.max_steps*pl_module.fl_current_epoch()+trainer.global_step
                if global_fl_step % 10 == 0 and self.log_grad:
                    mlflow.log_metric(f'{name}.grad.l2', v.grad.data.norm().detach().item(), global_fl_step)
                if trainer.global_step == 0 and self.log_diff:
                    mlflow.log_metric(f'{name}.c_diff.l2', (self.c_local[name] - self.c_global[name]).norm().item(), global_fl_step)


    def update_c_local(
        self, eta: float, c_global: ModuleParams, old_params: ModuleParams, new_params: ModuleParams
    ) -> None:
        """Updates local c variate using local learning rate, global c variate and model parameters before and after fit

        Args:
            eta (float): local learning rate
            c_global (ModuleParams): global c variate sent by FL server
            old_params (ModuleParams): old model params before fit
            new_params (ModuleParams): new model params after fit
        """        
        if self.c_local is None:
            self.c_local = OrderedDict([(layer, torch.zeros_like(value)) for layer, value in c_global.items()])

        for layer in c_global.keys():
            cg, op, np = c_global[layer], old_params[layer], new_params[layer]
            # print(f'update_c_local layer {layer}: {cg.shape}, {op.shape}, {np.shape}')
            self.c_local[layer] -= (cg - (1/(self.K*eta))*(op - np))

    
        