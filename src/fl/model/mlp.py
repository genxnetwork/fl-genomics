from typing import Dict, Any, List, Tuple, Optional
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear, BatchNorm1d
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, relu
import mlflow


class BaseNet(LightningModule):
    def __init__(self, input_size: int, l1: float, optim_params: Dict, scheduler_params: Dict) -> None:
        """Base class for all NN models, should not be used directly

        Args:
            input_size (int): Size of input data
            l1 (float): L1 regularization parameter
            optim_params (Dict): Parameters of optimizer
            scheduler_params (Dict): Parameters of learning rate scheduler
        """        
        super().__init__()
        self.layer = Linear(input_size, 1)
        self.l1 = l1
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        self.current_round = 0

    def forward(self, x):
        out = self.layer(x)
        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        raw_loss = self.calculate_loss(y_hat, y)
        reg = self.regularization()
        loss = raw_loss + reg
        return {'loss': loss, 'raw_loss': raw_loss.detach(), 'reg': reg.detach(), 'batch_len': x.shape[0]}

    def regularization(self) -> torch.Tensor:
        """Calculates l1 regularization of input layer by default

        Returns:
            torch.Tensor: Regularization loss
        """        
        return self.l1 * torch.norm(self.layer.weight, p=1)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        return {'val_loss': loss, 'batch_len': x.shape[0]}

    def calculate_avg_epoch_metric(self, outputs: List[Dict[str, Any]], metric_name: str) -> float:
        total_len = sum(out['batch_len'] for out in outputs)
        avg_loss = sum(out[metric_name].item()*out['batch_len'] for out in outputs)/total_len
        return avg_loss

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'loss')
        avg_raw_loss = self.calculate_avg_epoch_metric(outputs, 'raw_loss')
        avg_reg = self.calculate_avg_epoch_metric(outputs, 'reg')
        mlflow.log_metric('local_train_loss', avg_loss, self._fl_current_epoch())
        mlflow.log_metric('local_raw_loss', avg_raw_loss, self._fl_current_epoch())
        mlflow.log_metric('local_reg', avg_reg, self._fl_current_epoch())
        mlflow.log_metric('lr', self._get_current_lr(), self._fl_current_epoch())
        # self.log('train_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        mlflow.log_metric('local_val_loss', avg_loss, self._fl_current_epoch())
        self.log('val_loss', avg_loss)

    def _fl_current_epoch(self):
        return self.current_round * self.scheduler_params['epochs_in_round'] + self.current_epoch

    def _get_current_lr(self):
        optim = self.trainer.optimizers[0] 
        lr = optim.param_groups[0]['lr']
        return lr
    
    def _configure_adamw(self):
        optimizer = torch.optim.AdamW([
            {
                'params': self.parameters(), 
                'initial_lr': self.optim_params['lr']/self.scheduler_params['div_factor'], 
                'max_lr': self.optim_params['lr'],
                'min_lr': self.optim_params['lr']/self.scheduler_params['final_div_factor']}
            ], lr=self.optim_params['lr'], weight_decay=self.optim_params['weight_decay'])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.optim_params['lr'],
                                                        div_factor=self.scheduler_params['div_factor'],
                                                        final_div_factor=self.scheduler_params['final_div_factor'],
                                                        total_steps=int(self.scheduler_params['rounds']*(1.5*self.scheduler_params['epochs_in_round'])+2),
                                                        pct_start=0.1,
                                                        last_epoch=self.current_round*self.scheduler_params['epochs_in_round'],
                                                        #last_epoch=self.curren
                                                        cycle_momentum=False)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optim_init = {
            'adamw': self._configure_adamw 
        }[self.optim_params['name']]
        return optim_init()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])


class LinearRegressor(BaseNet):
    def __init__(self, input_size: int, l1: float, optim_params: Dict, scheduler_params: Dict) -> None:
        super().__init__(input_size, l1, optim_params, scheduler_params)
        self.layer = Linear(input_size, 1)
        self.l1 = l1

    def calculate_loss(self, y_hat, y):
        return mse_loss(y_hat.squeeze(1), y)
    

class LinearClassifier(BaseNet):
    def __init__(self, input_size: int, l1: float, lr: float, momentum: float, epochs: float) -> None:
        super().__init__()
        self.layer = Linear(input_size, 1)
        self.l1 = l1
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return binary_cross_entropy_with_logits(y_hat, y)
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()
        return {'val_loss': loss, 'batch_len': x.shape[0], 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        avg_accuracy = self.calculate_avg_val_accuracy(outputs, 'val_accuracy')
        self.log('val_loss', avg_loss)
        self.log('val_accuracy', avg_accuracy)


class MLPRegressor(BaseNet):
    def __init__(self, input_size: int, hidden_size: int, l1: float, optim_params: Dict, scheduler_params: Dict) -> None:
        super().__init__(input_size, l1, optim_params, scheduler_params)
        self.input = Linear(input_size, hidden_size)
        self.bn = BatchNorm1d(hidden_size)
        self.hidden = Linear(hidden_size, 1)
        self.l1 = l1
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = torch.sigmoid(self.input(x))
        return self.hidden(x)

    def regularization(self):
        reg = self.l1 * torch.norm(self.input.weight, p=1)
        return reg

    def calculate_loss(self, y_hat, y):
        return mse_loss(y_hat.squeeze(1), y)
