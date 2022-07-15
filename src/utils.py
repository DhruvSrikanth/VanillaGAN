import torch
import numpy as np

def print_config(config: dict) -> None:
    """
    Print the configuration.
    Parameters:
        config: (type dict) configuration.
    """
    print("\nGiven below is the configuration present in the config file:")
    i = 1
    for config_param, config_value in config.items():
        print(f"\t{i}. Parameter : {config_param} | Value : {config_value}")
        i += 1
    print("\n")

def print_strategy(strategy: dict) -> None:
    """
    Print the strategy.
    Parameters:
        strategy: (type dict) strategy.
    """

    names = {}
    try:
        names['Optimizer'] = type(strategy['optimizer']).__name__
    except KeyError:
        raise Exception('No optimizer specified.')
    try:
        names['Criterion'] = strategy['criterion'].__name__
    except KeyError:
        raise Exception('No criterion specified.')
    

    print("\nGiven below is the strategy for training the model:")
    i = 1
    for name, value in names.items():
        print(f"\t{i}. {name} : {value}")
        i += 1
    print("\n")

def decide_device(device: str) -> str:
    '''
    Decide which device to use.
    Params:
        device: (type str) device to use (mps for MAC GPU, CUDA for NVIDIA GPU, CPU for CPU)
    Returns:
        device: (type str) device to use
    '''
    device = device.lower()
    if device == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    elif device == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def resize_image(dims: tuple, image: np.ndarray) -> np.ndarray:
    """
    Resize an image to the given dimensions.
    Parameters:
        dims: (type tuple) dimensions to resize the image to.
        image: (type numpy.ndarray) image to resize.
    Returns:
        image: (type numpy.ndarray) resized image.
    """
    return image.resize(dims)
