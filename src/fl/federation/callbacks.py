from typing import Dict, OrderedDict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.core import LightningModule
import mlflow
import torch

from flwr.common import Weights

from fl.federation.utils import ModuleParams


class ScaffoldCallback(Callback):

    c_global: ModuleParams
    c_local: ModuleParams

    def __init__(self, K: int = 1, log_grad=False, log_diff=False) -> None:
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
            if v.grad is not None:
                v.grad.data += (self.c_global[name].to(v.grad.data.device) - self.c_local[name].to(v.grad.data.device))
                global_fl_step = trainer.max_steps*pl_module.fl_current_epoch()+trainer.global_step
                if global_fl_step % 10 == 0 and self.log_grad:
                    mlflow.log_metric(f'{name}.grad.l2', v.grad.data.norm().detach().item(), global_fl_step)
                if trainer.global_step == 0 and self.log_diff:
                    mlflow.log_metric(f'{name}.c_diff.l2', (self.c_local[name] - self.c_global[name]).norm().item(), global_fl_step)


    def update_c_local(
        self, eta: float, c_global: ModuleParams, old_params: ModuleParams, new_params: ModuleParams
    ) -> None:
        if self.c_local is None:
            self.c_local = OrderedDict([(layer, torch.zeros_like(value)) for layer, value in c_global.items()])

        for layer in c_global.keys():
            cg, op, np = c_global[layer], old_params[layer], new_params[layer]
            # print(f'update_c_local layer {layer}: {cg.shape}, {op.shape}, {np.shape}')
            self.c_local[layer] -= (cg - (1/(self.K*eta))*(op - np))

    
        