from typing import Dict, Any, List, Tuple, Optional
import numpy
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear, BatchNorm1d
from torch.nn.init import uniform_ as init_uniform_
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, relu
from torch.utils.data import DataLoader
from torchmetrics import R2Score
import mlflow

from nn.lightning import DataModule
from nn.utils import LassoNetRegMetrics, Metrics, RegLoaderMetrics, RegMetrics


class BaseNet(LightningModule):
    def __init__(self, input_size: int, optim_params: Dict, scheduler_params: Dict) -> None:
        """Base class for all NN models, should not be used directly

        Args:
            input_size (int): Size of input data
            optim_params (Dict): Parameters of optimizer
            scheduler_params (Dict): Parameters of learning rate scheduler
        """        
        super().__init__()
        self.layer = Linear(input_size, 1)
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

    def calculate_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('subclasses of BaseNet should implement loss calculation')

    def regularization(self) -> torch.Tensor:
        raise NotImplementedError('subclasses of BaseNet should implement regularization')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        return {'val_loss': loss, 'batch_len': x.shape[0]}
    
    def calculate_avg_epoch_metric(self, outputs: List[Dict[str, Any]], metric_name: str) -> float:
        total_len = sum(out['batch_len'] for out in outputs)
        avg_loss = sum(out[metric_name].item()*out['batch_len'] for out in outputs)/total_len
        return avg_loss if isinstance(avg_loss, float) else avg_loss.item()

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'loss')
        avg_raw_loss = self.calculate_avg_epoch_metric(outputs, 'raw_loss')
        avg_reg = self.calculate_avg_epoch_metric(outputs, 'reg')
        mlflow.log_metric('train_loss', avg_loss, self.fl_current_epoch())
        mlflow.log_metric('raw_loss', avg_raw_loss, self.fl_current_epoch())
        mlflow.log_metric('reg', avg_reg, self.fl_current_epoch())
        mlflow.log_metric('lr', self._get_current_lr(), self.fl_current_epoch())
        # self.log('train_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        mlflow.log_metric('val_loss', avg_loss, self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)    

    def fl_current_epoch(self):
        return self.current_round * self.scheduler_params['epochs_in_round'] + self.current_epoch

    def _get_current_lr(self):
        optim = self.trainer.optimizers[0] 
        lr = optim.param_groups[0]['lr']
        return lr
    
    def _configure_adamw(self):
        last_epoch = self.current_round*self.scheduler_params['epochs_in_round']
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
                                                        anneal_strategy='linear',
                                                        epochs=int(self.scheduler_params['rounds']*(1.5*self.scheduler_params['epochs_in_round'])+2),
                                                        pct_start=0.1,
                                                        steps_per_epoch=1,
                                                        last_epoch=last_epoch,
                                                        cycle_momentum=False)
        
        
        return [optimizer], [scheduler]

    def set_covariate_weights(self, weights: numpy.ndarray):
        raise NotImplementedError('for this model setting covariate weights is not implemented')


    def _configure_sgd(self):
        last_epoch = self.current_round*self.scheduler_params['epochs_in_round']
        optimizer = torch.optim.SGD([
            {
                'params': self.parameters(), 
                'lr': self.optim_params['lr']*self.scheduler_params['gamma']**last_epoch,
                'initial_lr': self.optim_params['lr'], 
            }], lr=self.optim_params['lr'], weight_decay=self.optim_params['weight_decay'])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_params['gamma'], last_epoch=self.current_round*self.scheduler_params['epochs_in_round']
        )
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optim_init = {
            'adamw': self._configure_adamw,
            'sgd': self._configure_sgd
        }[self.optim_params['name']]
        return optim_init()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])

    def predict(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pred = []
        y_true = []
        for x, y in loader:
            y_pred.append(self(x).detach().cpu())
            y_true.append(y.cpu())
        return torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)

    def predict_and_eval(self, datamodule: DataModule, **kwargs: Any) -> Metrics:
        raise NotImplementedError('subclasses of BaseNet should implement predict_end_eval')


class LinearRegressor(BaseNet):
    def __init__(self, input_size: int, l1: float, optim_params: Dict, scheduler_params: Dict) -> None:
        super().__init__(input_size, optim_params, scheduler_params)
        self.layer = Linear(input_size, 1)
        self.l1 = l1

    def regularization(self) -> torch.Tensor:
        """Calculates l1 regularization of input layer by default

        Returns:
            torch.Tensor: Regularization loss
        """        
        return self.l1 * torch.norm(self.layer.weight, p=1)

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


class MLPPredictor(BaseNet):
    def __init__(self, input_size: int, hidden_size: int, l1: float, optim_params: Dict, scheduler_params: Dict, loss = mse_loss) -> None:
        super().__init__(input_size, optim_params, scheduler_params)
        self.input = Linear(input_size, hidden_size)
        self.bn = BatchNorm1d(hidden_size)
        self.hidden = Linear(hidden_size, 1)
        self.loss = loss
        self.l1 = l1
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = relu(self.input(x))
        return self.hidden(x)

    def regularization(self):
        reg = self.l1 * torch.norm(self.input.weight, p=1)
        return reg

    def calculate_loss(self, y_hat, y):
        return self.loss(y_hat.squeeze(1), y)


class LassoNetRegressor(BaseNet):
    def __init__(self, input_size: int, hidden_size: int, 
                 optim_params: Dict, scheduler_params: Dict,
                 cov_count: int = 0, 
                 alpha_start: float = -1, alpha_end: float = -1, init_limit: float = 0.01,
                 logger = None) -> None:
        super().__init__(input_size, optim_params, scheduler_params)
        
        assert alpha_end > alpha_start
        self.alphas = numpy.logspace(alpha_start, alpha_end, num=hidden_size, endpoint=True, base=10)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cov_count = cov_count
        self.layer = Linear(self.input_size, self.hidden_size)
        self.bn = BatchNorm1d(self.input_size)
        self.r2_score = R2Score(num_outputs=self.hidden_size, multioutput='raw_values')
        init_uniform_(self.layer.weight, a=-init_limit, b=init_limit) 
        self.stdout_logger = logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(self.bn(x))
        return out 

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.unsqueeze(1).tile(dims=(1, self.hidden_size))
        # print(y.shape, y_hat.shape)
        return mse_loss(y_hat, y)
    
    def regularization(self) -> torch.Tensor:

        alphas = torch.tensor(self.alphas, device=self.layer.weight.device, dtype=torch.float32)
        if self.cov_count == 0:
            return torch.dot(alphas, torch.norm(self.layer.weight, p=1, dim=1))/self.hidden_size 
        else:
            w = self.layer.weight[:, :self.layer.weight.shape[1] - self.cov_count]
            return torch.dot(alphas, torch.norm(w, p=1, dim=1))/self.hidden_size

    def _unreduced_mse_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        return [mse_loss(y_pred[:, i], y_true).item() for i in range(self.hidden_size)]

    def _unreduced_r2_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        y = y_true.unsqueeze(1).tile(dims=(1, self.hidden_size))
        r2 = [r.item() for r in self.r2_score(y_pred, y)]
        return r2

    def _unreduced_pred_metrics(self, prefix: str, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[RegLoaderMetrics]:
        losses = self._unreduced_mse_loss(y_pred, y_true)
        r2s = self._unreduced_r2_score(y_pred, y_true)
        return [RegLoaderMetrics(prefix, loss, r2, self.fl_current_epoch(), samples=y_true.shape[0]) for loss, r2 in zip(losses, r2s)]

    def predict_and_eval(self, datamodule: DataModule, test=False, best_col : Optional[int] = None, logger = None) -> Metrics:
        train_loader, val_loader, test_loader = datamodule.predict_dataloader()
        logger.info(f'got predict_dataloaders')
        y_train_pred, y_train = self.predict(train_loader)
        logger.info(f'predicted train dataloader')
        y_val_pred, y_val = self.predict(val_loader)
        logger.info(f'predicted val dataloader')
        
        train_metrics = self._unreduced_pred_metrics('train', y_train_pred, y_train)
        logger.info('got train metrics')
        val_metrics = self._unreduced_pred_metrics('val', y_val_pred, y_val)
        logger.info(f'got val metrics')
        
        if test:
            y_test_pred, y_test = self.predict(test_loader)
            test_metrics = self._unreduced_pred_metrics('test', y_test_pred, y_test)
            if best_col is None:
                best_col = numpy.argmax([vm.r2 for vm in val_metrics])
            metrics = RegMetrics(train_metrics[best_col], val_metrics[best_col], test_metrics[best_col], epoch=self.fl_current_epoch())
        else:
            metrics = LassoNetRegMetrics(train_metrics, val_metrics, None, self.fl_current_epoch())
            
        return metrics

    def get_best_predictions(self, train_preds: torch.Tensor, val_preds: torch.Tensor, test_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        best_col = torch.amax(self.r2_score(val_preds))
        return train_preds[:, best_col], val_preds[:, best_col], test_preds[:, best_col]

    def set_covariate_weights(self, weights: numpy.ndarray):
        cov_weights = torch.tensor(weights, dtype=self.layer.weight.dtype).unsqueeze(0).tile((self.hidden_size, 1))

        weight = self.layer.weight.data
        snp_count = self.layer.weight.shape[1] - cov_weights.shape[1]
        #('covariate weight shapes are ', weight.shape, snp_count, cov_weights.shape, weight[:, snp_count:].shape)
        weight[:, snp_count:] = cov_weights
        self.layer.weight = torch.nn.Parameter(weight)
    