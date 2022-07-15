import torch
from torch.utils.data import DataLoader
from torchvision import datasets

def get_dataloader(type: str, batch_size: int, shuffle: bool, num_workers:int=1, transform:object=None) -> DataLoader:
    """
    Get data loader for MNIST dataset.
    Params:
        type : (type str) type of dataset to load. Valid types are 'train' and 'test'.
        batch_size : (type int) batch size of data loader.
        shuffle : (type bool) whether to shuffle the dataset.
        num_workers : (type int) number of workers to use for data loader.
    Returns:
        data_loader : (type torch.utils.data.DataLoader) data loader for MNIST dataset.
    """
    type = type.lower()
    if type == 'train':
        dataset = datasets.MNIST('./data/train', train=True, download=True, transform=transform)
    elif type == 'test':
        dataset = datasets.MNIST('./data/test', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Invalid type: {type}. Expected 'train' or 'test'.")
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader