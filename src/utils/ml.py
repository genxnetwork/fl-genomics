from dataclasses import dataclass
from utils.loaders import Y


@dataclass
class RawPreds:
    y_true: Y
    y_pred: Y
    # one of binary, discrete, continuous
    task_type: str = "binary" 