from typing import Dict, Any, List, Tuple, Optional
import numpy
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import torch
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, Dropout
from torch.nn.init import uniform_ as init_uniform_
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, relu6, softmax, relu, selu, binary_cross_entropy
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, R2Score
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import Metric
import time
import logging

from configs.phenotype_config import TYPE_LOSS_DICT
from nn.lightning import DataModule
# from nn.utils import ClfLoaderMetrics, ClfMetrics, LassoNetRegMetrics, Metrics, RegLoaderMetrics, RegMetrics
from nn.metrics import ModelMetrics, DatasetMetrics, RegMetrics, ClfMetrics, LassoNetModelMetrics
from utils.loaders import Y
from utils.ml import RawPreds


class BaseNet(LightningModule):
    def __init__(self, input_size: int, optim_params: Dict, scheduler_params: Dict) -> None:
        """Base class for all NN models, should not be used directly

        Args:
            input_size (int): Size of input data
            optim_params (Dict): Parameters of optimizer
            scheduler_params (Dict): Parameters of learning rate scheduler
        """
        super().__init__()
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        self.current_round = 1
        self.mlflow_client = MlflowClient()
        self.history: List[Metric] = []
        self.logged_count = 0

    def _add_to_history(self, name: str, value, step: int):
        timestamp = int(time.time() * 1000)
        self.history.append(Metric(name, value, timestamp, step))
        if len(self.history) % 50 == 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-50:])
            self.logged_count = len(self.history)

    def on_train_end(self) -> None:
        unlogged = len(self.history) - self.logged_count
        if unlogged > 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-unlogged:])
            self.logged_count = len(self.history)
        return super().on_train_end()

    def on_predict_end(self) -> None:
        unlogged = len(self.history) - self.logged_count
        if unlogged > 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-unlogged:])
            self.logged_count = len(self.history)
        return super().on_validation_end()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        raw_loss = self.calculate_loss(y_hat, y)
        reg = self.regularization()
        loss = raw_loss + reg
        return {'loss': loss, 'raw_loss': raw_loss.detach(), 'reg': reg.detach(), 'batch_len': x.shape[0]}

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

        step = self.fl_current_epoch()
        self._add_to_history('train_loss', avg_loss, step)
        self._add_to_history('raw_loss', avg_raw_loss, step)
        self._add_to_history('reg', avg_reg, step)
        self._add_to_history('lr', self.get_current_lr(), step)

        '''
        mlflow.log_metrics({
            'train_loss': avg_loss,
            'raw_loss': avg_raw_loss,
            'reg': avg_reg,
            'lr': self.get_current_lr()
        }, step=step)

        mlflow.log_metric('train_loss', avg_loss, self.fl_current_epoch())
        mlflow.log_metric('raw_loss', avg_raw_loss, self.fl_current_epoch())
        mlflow.log_metric('reg', avg_reg, self.fl_current_epoch())
        mlflow.log_metric('lr', self.get_current_lr(), self.fl_current_epoch())
        '''
        # self.log('train_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        self._add_to_history('val_loss', avg_loss, step=self.fl_current_epoch())
        # mlflow.log_metric('val_loss', avg_loss, self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)

    def fl_current_epoch(self):
        return (self.current_round - 1) * self.scheduler_params['epochs_in_round'] + self.current_epoch

    def get_current_lr(self):
        if self.trainer is not None:
            optim = self.trainer.optimizers[0]
            lr = optim.param_groups[0]['lr']
        else:
            return self.optim_params['lr']
        return lr

    def _configure_adamw(self):
        last_epoch = (self.current_round - 1) * self.scheduler_params['epochs_in_round']
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
        last_epoch = (self.current_round - 1) * self.scheduler_params['epochs_in_round'] if self.scheduler_params is not None else 0

        optimizer = torch.optim.SGD([
            {
                'params': self.parameters(),
                'lr': self.optim_params['lr']*self.scheduler_params['gamma']**last_epoch if self.scheduler_params is not None else self.optim_params['lr'],
                'initial_lr': self.optim_params['lr'],
            }], lr=self.optim_params['lr'], weight_decay=self.optim_params.get('weight_decay', 0))

        schedulers = [torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_params['gamma'], last_epoch=last_epoch
        )] if self.scheduler_params is not None else None
        return [optimizer], schedulers

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
    
    def loader_metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> DatasetMetrics:
        raise NotImplementedError('subclasses of BaseNet should implement loader_metrics')

    def predict_and_eval(self, datamodule: DataModule, **kwargs: Any) -> ModelMetrics:
        raise NotImplementedError('subclasses of BaseNet should implement predict_end_eval')


class LinearRegressor(BaseNet):
    def __init__(self, input_size: int, l1: float, optim_params: Dict, scheduler_params: Dict) -> None:
        super().__init__(input_size, optim_params, scheduler_params)
        self.layer = Linear(input_size, 1)
        self.l1 = l1
        # TODO: move to callback
        # self.beta_history = []
        self.r2_score = R2Score()

    def regularization(self) -> torch.Tensor:
        """Calculates l1 regularization of input layer by default

        Returns:
            torch.Tensor: Regularization loss
        """
        return self.l1 * torch.norm(self.layer.weight, p=1)

    def calculate_loss(self, y_hat, y):
        return mse_loss(y_hat.squeeze(1), y)

    def forward(self, x) -> Any:
        return self.layer(x)

    def on_after_backward(self) -> None:
        # print(w)
        # self.beta_history.append(self.layer.weight.detach().cpu().numpy().copy())
        mlflow.log_metric('grad_norm', torch.norm(self.layer.weight.grad).item(), self.fl_current_epoch())
        return super().on_after_backward()

    def loader_metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> RegMetrics:
        mse = mse_loss(y_hat.squeeze(1), y)
        r2 = self.r2_score(y_hat.squeeze(1), y)
        return RegMetrics(mse.item(), r2.item(), self.fl_current_epoch(), y_hat.shape[0])

    def predict_and_eval(self, datamodule: DataModule, test=False) -> ModelMetrics:
        train_loader, val_loader, test_loader = datamodule.predict_dataloader()
        y_train_pred, y_train = self.predict(train_loader)
        y_val_pred, y_val = self.predict(val_loader)

        train_metrics = self.loader_metrics(y_train_pred, y_train)
        val_metrics = self.loader_metrics(y_val_pred, y_val)

        if test:
            y_test_pred, y_test = self.predict(test_loader)
            test_metrics = self.loader_metrics(y_test_pred, y_test)
        else:
            test_metrics = None
        metrics = ModelMetrics(train_metrics, val_metrics, test_metrics)
        return metrics


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

    def forward(self, x) -> Any:
        return self.layer(x)

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

    def loader_metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> RegMetrics:
        bce = binary_cross_entropy_with_logits(y_hat, y)
        accuracy = accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        return ClfMetrics(bce.item(), accuracy.item(), self.fl_current_epoch(), y_hat.shape[0])


class MLPPredictor(BaseNet):
    def __init__(self, input_size: int, hidden_size: int, l1: float, optim_params: Dict, scheduler_params: Dict, loss = mse_loss) -> None:
        super().__init__(input_size, optim_params, scheduler_params)
        self.input = Linear(input_size, hidden_size)
        self.bn = BatchNorm1d(hidden_size)
        self.hidden = Linear(hidden_size, hidden_size)
        self.hidden2 = Linear(hidden_size, 1)
        self.loss = loss
        self.l1 = l1
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        self.r2_score = R2Score(num_outputs=1, multioutput='uniform_average')

    def forward(self, x):
        x = selu(self.input(x))
        x = selu(self.hidden(x))
        return self.hidden2(x)

    def regularization(self):
        reg = self.l1 * torch.norm(self.input.weight, p=1)
        return reg

    def calculate_loss(self, y_hat, y):
        return self.loss(y_hat.squeeze(1), y)

    def loader_metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> RegMetrics:
        mse = mse_loss(y_hat.squeeze(1), y)
        r2 = self.r2_score(y_hat.squeeze(1), y)
        return RegMetrics(mse.item(), r2.item(), self.fl_current_epoch(), y_hat.shape[0])


class MLPClassifier(BaseNet):
    def __init__(self, nclass, nfeat, optim_params, scheduler_params, loss, hidden_size=800, hidden_size2=200, binary=False) -> None:
        super().__init__(input_size=None, optim_params=optim_params, scheduler_params=scheduler_params)
        # self.bn = BatchNorm1d(nfeat)
        self.nclass = nclass
        self.fc1 = Linear(nfeat, hidden_size)
        # self.bn2 = BatchNorm1d(hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size2)
        # self.bn3 = BatchNorm1d(hidden_size2)
        self.fc3 = Linear(hidden_size2, nclass)
        self.loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.bn(x)
        x = selu(self.fc1(x))
        x = selu(self.fc2(x))
        # x = softmax(, dim=1)
        return torch.sigmoid(self.fc3(x))

    def regularization(self):
        return torch.tensor(0)

    def calculate_loss(self, y_hat, y):
        # print(f'SHAPES IN LOSS ARE: {y_hat.shape}, {y.shape}')
        return self.loss(y_hat.squeeze(1), y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        raw_loss = self.calculate_loss(y_hat, y)
        reg = self.regularization()
        loss = raw_loss + reg

        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()

        return {
            'loss': loss,
            'raw_loss': raw_loss.detach(),
            'reg': reg.detach(),
            'batch_len': x.shape[0],
            'accuracy': accuracy
        }

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        super(MLPClassifier, self).training_epoch_end(outputs)

        step = self.fl_current_epoch()
        avg_accuracy = self.calculate_avg_epoch_metric(outputs, 'accuracy')
        self._add_to_history('train_accuracy', avg_accuracy, step)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)

        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()

        return {'val_loss': loss, 'val_accuracy': accuracy, 'batch_len': x.shape[0]}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        avg_accuracy = self.calculate_avg_epoch_metric(outputs, 'val_accuracy')
        self._add_to_history('val_loss', avg_loss, step=self.fl_current_epoch())
        self._add_to_history('val_accuracy', avg_accuracy, step=self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)

    def loader_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> ClfMetrics:
        loss = self.calculate_loss(y_pred, y_true)
        accuracy = Accuracy(num_classes=self.nclass)
        return ClfMetrics(loss.item(), accuracy(y_pred, y_true).item(), epoch=self.fl_current_epoch(), samples=y_pred.shape[0])


class DeepMLPClassifier(MLPClassifier):
    def __init__(self, nclass, nfeat, optim_params, scheduler_params, loss, hidden_size=800, hidden_size2=200, dropout=0.0, binary=False) -> None:
        super().__init__(nclass, nfeat, optim_params=optim_params, scheduler_params=scheduler_params,
                         loss=loss, hidden_size=hidden_size, hidden_size2=hidden_size2)
        self.nclass = nclass
        layers = [
            BatchNorm1d(nfeat), 
            Linear(nfeat, hidden_size), BatchNorm1d(hidden_size), ReLU(), Dropout(dropout),
            Linear(hidden_size, hidden_size), BatchNorm1d(hidden_size), ReLU(), Dropout(dropout),
            Linear(hidden_size, hidden_size2), BatchNorm1d(hidden_size2), ReLU(), Dropout(dropout),
            Linear(hidden_size2, hidden_size2), BatchNorm1d(hidden_size2), ReLU(), Dropout(dropout),
            Linear(hidden_size2, nclass),
            
        ]
        self.layers = Sequential(*layers)
        self.loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layers(x)
        return torch.sigmoid(x)
    
    def _add_to_history(self, name: str, value, step: int):
        timestamp = int(time.time() * 1000)
        self.history.append(Metric(name, value, timestamp, step))
        if len(self.history) % 5 == 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-5:])
            self.logged_count = len(self.history)


class LassoNetRegressor(BaseNet):
    def __init__(self, input_size: int, hidden_size: int,
                 optim_params: Dict, scheduler_params: Dict,
                 cov_count: int = 0,
                 alpha_start: float = -1, alpha_end: float = -1, init_limit: float = 0.01, use_bn: bool = True,
                 loss = None,
                 logger = None) -> None:
        super().__init__(input_size, optim_params, scheduler_params)

        assert alpha_end > alpha_start
        self.alphas = numpy.logspace(alpha_start, alpha_end, num=hidden_size, endpoint=True, base=10)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cov_count = cov_count
        self.use_bn = use_bn
        self.layer = Linear(self.input_size, self.hidden_size)
        if self.use_bn:
            self.bn = BatchNorm1d(self.input_size)
        self.r2_score = R2Score(num_outputs=self.hidden_size, multioutput='raw_values')
        init_uniform_(self.layer.weight, a=-init_limit, b=init_limit)
        self.stdout_logger = logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bn:
            x = self.bn(x)
        out = self.layer(x)
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

    def loader_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[RegMetrics]:
        losses = self._unreduced_mse_loss(y_pred, y_true)
        r2s = self._unreduced_r2_score(y_pred, y_true)
        return [RegMetrics(loss, r2, self.fl_current_epoch(), samples=y_true.shape[0]) for loss, r2 in zip(losses, r2s)]

    def predict_and_eval(self, datamodule: DataModule, test=False, return_preds=False) -> LassoNetModelMetrics:
        train_loader, val_loader, test_loader = datamodule.predict_dataloader()
        y_train_pred, y_train = self.predict(train_loader)
        y_val_pred, y_val = self.predict(val_loader)

        train_metrics = self.loader_metrics(y_train_pred, y_train)
        val_metrics = self.loader_metrics(y_val_pred, y_val)

        if test:
            y_test_pred, y_test = self.predict(test_loader)
            test_metrics = self.loader_metrics(y_test_pred, y_test)
        else:
            test_metrics = None
        metrics = LassoNetModelMetrics(train_metrics, val_metrics, test_metrics)

        
        if return_preds:
            preds = RawPreds(
                Y(y_train.numpy(), y_val.numpy(), None), 
                Y(y_train_pred.numpy(), y_val_pred.numpy(), None),
                "continuous"
            )
            if test:
                preds.y_true.test = y_test.numpy()
                preds.y_pred.test = y_test_pred.numpy()
                
            return metrics, preds
        
        return metrics

    def get_best_predictions(self, train_preds: torch.Tensor, val_preds: torch.Tensor, test_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        best_col = torch.amax(self.r2_score(val_preds))
        return train_preds[:, best_col], val_preds[:, best_col], test_preds[:, best_col]

    def set_covariate_weights(self, weights: numpy.ndarray):
        cov_weights = torch.tensor(weights, dtype=self.layer.weight.dtype).unsqueeze(0).tile((self.hidden_size, 1))

        weight = self.layer.weight.data
        snp_count = self.layer.weight.shape[1] - cov_weights.shape[1]
        logging.info(f'covariate weight shapes are: weight={weight.shape}, snp_count={snp_count}, cov_weights={cov_weights.shape}, {weight[:, snp_count:].shape}')
        weight[:, snp_count:] = cov_weights
        self.layer.weight = torch.nn.Parameter(weight)


class LassoNetClassifier(LassoNetRegressor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bn:
            x = self.bn(x)
        out = self.layer(x)
        return torch.sigmoid(out)

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.unsqueeze(1).tile(dims=(1, self.hidden_size))
        return binary_cross_entropy(y_hat, y)
    
    def one_col_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> ClfMetrics:
         
        loss = binary_cross_entropy(y_pred, y_true).item()
        accuracy = ((y_pred > 0.5).float() == y_true).float().mean().item()
        y_true_n, y_pred_n = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
        auc = roc_auc_score(y_true_n, y_pred_n)
        aupr = average_precision_score(y_true_n, y_pred_n)
        return ClfMetrics(loss, accuracy, auc, self.fl_current_epoch(), y_true.shape[0], aupr)
    
    def loader_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[ClfMetrics]:
        
        result: List[ClfMetrics] = []
        for col in range(y_pred.shape[1]):
            col_metrics = self.one_col_metrics(y_pred[:, col], y_true)
            result.append(col_metrics)
            
        return result

    def predict_and_eval(self, datamodule: DataModule, test=False, return_preds=False) -> LassoNetModelMetrics:
        if return_preds:
            metrics, preds = super().predict_and_eval(datamodule, test=test, return_preds=return_preds)
            preds.task_type = "binary"
            return metrics, preds
        return super().predict_and_eval(datamodule, test=test, return_preds=return_preds)