from dataclasses import dataclass
from utils.loaders import Y


@dataclass
class RawPreds:
    y_true: Y
    y_pred: Y