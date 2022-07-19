import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets

import numpy as np

def get_dataloader(type: str, batch_size: int, num_workers: int=1, transform: object=None, samples: int=None):
    """
    Get data loader for MNIST dataset.
    Params:
        type : (type str) type of dataset to load. Valid types are 'train' and 'test'.
        batch_size : (type int) batch size of data loader.
        num_workers : (type int) number of workers to use for data loader.
        transform : (type object) transform to apply to the dataset.
        samples : (type int) number of samples to load.
    Returns:
        data_loader : (type torch.utils.data.DataLoader) data loader for MNIST dataset.
    """
    # Type check the type
    type = type.lower()

    # Check if the type is valid
    if type != 'train' and type != 'test':
        raise ValueError(f"Invalid type: {type}. Expected 'train' or 'test'.")
    
    # Get the dataset
    dataset = datasets.MNIST(f'./data/{type}', train=type == 'train', download=True, transform=transform)

    # Check for valid samples
    if not samples:
        if len(dataset) < samples:
            raise ValueError(f"Number of samples - {samples} is greater than the number of samples in the dataset - {len(dataset)}.")
        samples = len(dataset)
    
    # Get a subset of the dataset
    sampled_dataset = Subset(dataset, np.arange(samples))
    sample_sampler = RandomSampler(sampled_dataset) 
    
    # Create the data loader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=sample_sampler)
    
    return dataloader