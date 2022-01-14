from typing import Any
from pytorch_lightning import LightningModule


class Net(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        