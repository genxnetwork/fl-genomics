from collections import OrderedDict
from io import BytesIO
import numpy
from typing import List, cast

from flwr.common import Weights
import torch


ModuleParams = OrderedDict[str, torch.Tensor]

def weights_to_bytes(weights: Weights) -> bytes:
    bytes_io = BytesIO()
    numpy.savez(bytes_io, *weights)
    return bytes_io.getvalue()

def bytes_to_weights(tensor: bytes) -> Weights:
    bytes_io = BytesIO(tensor)
    npz_bytes = numpy.load(bytes_io, allow_pickle=False)
    weights = [cast(numpy.ndarray, npz_bytes[array]) for array in npz_bytes.files]
    npz_bytes.close()
    return weights

def weights_to_module_params(layer_names: List[str], weights: Weights) -> ModuleParams:
    params_dict = zip(layer_names, weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if v.shape != ()})
    return state_dict
