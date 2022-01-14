from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset):
        super().__init__(None, None, None, None)