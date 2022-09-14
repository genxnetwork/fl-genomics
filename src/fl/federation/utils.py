from collections import OrderedDict
from io import BytesIO
import numpy
from typing import List, cast

from flwr.common import Weights
import torch


ModuleParams = OrderedDict[str, torch.Tensor]

def weights_to_bytes(weights: Weights) -> bytes:
    """Converts list of numpy arrays to bytes for gRPC transfer

    Args:
        weights (Weights): List of numpy arrays. Typically these are model parameters

    Returns:
        bytes: Weights in byte form
    """    
    bytes_io = BytesIO()
    numpy.savez(bytes_io, *weights)
    return bytes_io.getvalue()

def bytes_to_weights(tensor: bytes) -> Weights:
    """Converts bytes to the list of numpy arrays after gRPC transfer

    Args:
        tensor (bytes): Weights in byte form

    Returns:
        Weights: Weights as list of numpy arrays
    """    
    bytes_io = BytesIO(tensor)
    npz_bytes = numpy.load(bytes_io, allow_pickle=False)
    weights = [cast(numpy.ndarray, npz_bytes[array]) for array in npz_bytes.files]
    npz_bytes.close()
    return weights

def weights_to_module_params(layer_names: List[str], weights: Weights) -> ModuleParams:
    """Coverts list of numpy arrays to pytorch model state dict using corresponding layer_names

    Args:
        layer_names (List[str]): List of names of model layers
        weights (Weights): List of model weights as numpy arrays

    Returns:
        ModuleParams: Pytorch model state dict which can be used for model initialization
    """    
    params_dict = zip(layer_names, weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if v.shape != ()})
    return state_dict
