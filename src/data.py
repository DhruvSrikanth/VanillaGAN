import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets

import numpy as np

def get_dataloader(type: str, batch_size: int, shuffle: bool, num_workers: int=1, transform: object=None, samples: int=None):
    """
    Get data loader for MNIST dataset.
    Params:
        type : (type str) type of dataset to load. Valid types are 'train' and 'test'.
        batch_size : (type int) batch size of data loader.
        shuffle : (type bool) whether to shuffle the dataset.
        num_workers : (type int) number of workers to use for data loader.
        transform : (type object) transform to apply to the dataset.
        samples : (type int) number of samples to load.
    Returns:
        data_loader : (type torch.utils.data.DataLoader) data loader for MNIST dataset.
    """
    type = type.lower()
    if type == 'train':
        shuffle = True
    elif type == 'test':
        shuffle = False
    else:
        raise ValueError(f"Invalid type: {type}. Expected 'train' or 'test'.")
    dataset = datasets.MNIST(f'./data/{type}', train=type == 'train', download=True, transform=transform)
    if not samples:
        samples = len(dataset)
    sampled_dataset = Subset(dataset, np.arange(samples))
    sample_sampler = RandomSampler(sampled_dataset) 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sample_sampler)
    
    return dataloader