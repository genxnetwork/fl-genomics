import torch
import numpy

import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

class BaseNet(LightningModule):
    def __init__(self, train_dataset, val_dataset, test_dataset=None,
                 batch_size=32, lr=0.001, total_steps=1000, weight_decay=0.0):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.total_steps = total_steps
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss.detach())
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    gamma=0.25,
                                                    step_size=16)
        return [optimizer], []

    def train_dataloader(self):
        if self.batch_size is not None:
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        else:
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # r2 = r2_score(y.detach().cpu(), y_hat.detach().cpu())
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss.detach())
        return {'val_loss': avg_loss}

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1)
        return loader

    def predict(self, data_loader):
        preds = []
        for batch in data_loader:
            y_hat = self(batch[0]).detach().cpu()
            preds.append(y_hat)

        return torch.cat(preds, dim=0)

class LinearNet(BaseNet):
    def __init__(self, train_dataset, val_dataset, test_dataset=None,
                 input_size=10, num_workers=4, cov_size=1, 
                 batch_size=32, lr=0.001, total_steps=1000, weight_decay=0.0, alpha=None):

        super().__init__(train_dataset, val_dataset, batch_size=batch_size, lr=lr, total_steps=total_steps, weight_decay=weight_decay)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.input_size = input_size
        self.cov_size = cov_size
        self.num_workers = num_workers
        self.layer = nn.Linear(self.input_size, 1)
        self.cov = nn.Linear(self.cov_size, 1)
        nn.init.uniform_(self.cov.weight, -0.05, 0.05)
        self.out = nn.Linear(2, 1)
        nn.init.uniform_(self.layer.weight, a=-0.009, b=0.009) # AUC for 0.000100 alpha is 0.71860 for std 0.002 

    def forward(self, x) -> torch.tensor:
        p = self.layer(x[:, :self.input_size])
        c = self.cov(x[:, self.input_size:])
        t = torch.cat([p, c], dim=1)
        return torch.sigmoid(self.out(t))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        if self.alpha is not None:
            loss += self.alpha * torch.norm(self.layer.weight, p=1)

        self.log('train_loss', loss.detach())
        auc = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        self.log('auc', auc)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)
        y_pred = (y_hat > 0.5).long()
        auc = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        accuracy = (y == y_pred).float().mean()
        return {'val_loss': F.binary_cross_entropy(y_hat, y), 'val_accuracy': accuracy, 'val_auc': auc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_auc = sum(x['val_auc'] for x in outputs)/len(outputs)
        self.log('val_loss', avg_loss.detach())
        self.log('val_accuracy', avg_accuracy)
        self.log('val_auc', avg_auc)

#         tensorboard = self.logger.experiment
#         tensorboard.add_histogram('input_weight', self.layer.weight.detach(), global_step=self.current_epoch)

        return {'val_loss': avg_loss}

    def predict(self, data_loader):
        preds = []
        for batch in data_loader:
            y_hat = self(batch[0]).detach().cpu()
            preds.append(y_hat.argmax(dim=1).long())

        return torch.cat(preds, dim=0)

    def predict_raw_proba(self, data_loader):
        preds = []
        for batch in data_loader:
            y_hat = self(batch[0]).detach().cpu()
            preds.append(y_hat)

        return torch.cat(preds, dim=0)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=1, collate_fn=collate_fn)
        fake_loader = FakeLoader(loader, num_workers=self.num_workers)
        return fake_loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=1, collate_fn=collate_fn)
        fake_loader = FakeLoader(loader, num_workers=self.num_workers)
        return fake_loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=1, collate_fn=collate_fn)
        fake_loader = FakeLoader(loader, num_workers=self.num_workers)
        return fake_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.layer.parameters(), 'lr': self.lr},
                {'params': self.cov.parameters(), 'lr': self.lr*20}
            ], weight_decay=self.weight_decay)
        return [optimizer], []

class EnsembleLASSO(LinearNet):
    def __init__(self, train_dataset, val_dataset, test_dataset=None,
                 input_size=10, num_workers=4, hidden_size=10, 
                 batch_size=32, total_steps=1000, 
                 alpha_start=None, alpha_end=None, init_limit=0.008, optim_params=None):

        super().__init__(train_dataset, val_dataset, batch_size=batch_size, lr=1e-3, total_steps=total_steps, weight_decay=0.0)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        # self.alphas = numpy.linspace(alpha_start, alpha_end, num=hidden_size)
        self.alphas = numpy.logspace(alpha_start, alpha_end, num=hidden_size, endpoint=True, base=10)
        self.input_size = input_size
        self.num_workers = num_workers
        self.hidden_size = hidden_size
        self.optim_params = optim_params 

        self.mse = nn.MSELoss()
        self.layer = nn.Linear(self.input_size, hidden_size)
        self.bn = nn.BatchNorm1d(self.input_size)
        nn.init.uniform_(self.layer.weight, a=-init_limit, b=init_limit) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.layer(self.bn(x[:, :self.input_size]))
        out = self.layer(self.bn(x))
        # out = self.layer(x)
        return out 

    def training_step(self, batch, batch_idx):
        x, y = batch[0].squeeze(0).float(), batch[1].squeeze(0).float()
        y_hat = self(x)
        y = y.tile(dims=(1, self.hidden_size))
        mse = self.mse(y_hat, y)        
        self.log('train_mse', mse.detach())
        l1 = 0.0
        if self.alphas is not None:
            alphas = torch.tensor(self.alphas, device=self.layer.weight.device, dtype=torch.float32)
            l1 = torch.dot(alphas, torch.norm(self.layer.weight, p=1, dim=1))/self.hidden_size
            # for i, alpha in enumerate(self.alphas):
                # mse loss is the mean of all model losses, therefore we need to average l1 as well.
            #    l1 += alpha * torch.norm(self.layer.weight[i, :], p=1) / self.hidden_size
        self.log('train_l1', l1)
        loss = mse + l1
        self.log('train_loss', loss.detach())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].squeeze(0).float(), batch[1].squeeze(0).float()

        # print(f'VALIDATION BATCH SIZES ARE: ', x.shape, y.shape)
        y_hat = self(x)
        y = y.tile(dims=(1, self.hidden_size))
        loss = self.mse(y_hat, y)

        val_losses = []
        for alpha, col in zip(self.alphas, range(self.hidden_size)):
            mse = self.mse(y[:, col], y_hat[:, col])
            val_losses.append(mse)
            self.log(f'val_mse@{alpha:.6f}', mse)

        # accuracy = (y[:, 0] == y_pred).float().mean()
        return {'val_loss': loss, 'val_losses': val_losses}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss.detach(), prog_bar=True)
        val_losses = torch.stack([torch.stack(x['val_losses']) for x in outputs])
        min_loss = val_losses.mean(dim=0).min()
        self.log('val_loss', min_loss, prog_bar=True)
        print(f'val loss {min_loss}')
#         mlflow.log_metric('val_loss', min_loss.item(), self.current_epoch)
        tensorboard = self.logger.experiment
        tensorboard.add_histogram('input_weight', self.layer.weight.detach(), global_step=self.current_epoch)

        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        if self.optim_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,  betas=(0.9, 0.999), eps=1e-8)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        elif self.optim_params.name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optim_params.lr, momentum=self.optim_params.momentum)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.optim_params.gamma)
        elif self.optim_params.name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_params.lr,  betas=(0.9, 0.999), eps=1e-8)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.optim_params.gamma)
        else:
            raise ValueError(f'Optimizer name {self.optim_params.name} is unknown')
        return [optimizer], [scheduler]

    def predict(self, data_loader):
        preds = []
        for batch in data_loader:
            x, y = batch
            y_hat = self(x.squeeze(0).float()).detach()
            preds.append(y_hat)

        return torch.cat(preds, dim=0)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, prefetch_factor=2)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, prefetch_factor=2)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, prefetch_factor=2)
        return loader