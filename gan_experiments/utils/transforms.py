import numpy as np

import torch

class ImageTransforms(object):
    """
    Class to handle the transforms.
    """
    def __init__(self) -> None:
        """
        Initialize the transforms.
        Parameters:
            None
        Returns:
            None
        """
        pass

    def resize_image(self, image: np.ndarray, dims: tuple) -> np.ndarray:
        """
        Resize an image to the given dimensions.
        Parameters:
            dims: (type tuple) dimensions to resize the image to.
            image: (type numpy.ndarray) image to resize.
        Returns:
            image: (type numpy.ndarray) resized image.
        """
        return image.resize(dims)

    def normalize_tensor(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        """
        Normalize a tensor.
        Parameters:
            tensor: (type torch.Tensor) tensor to normalize.
        Returns:
            tensor: (type torch.Tensor) normalized tensor.
        """
        return tensor / 255.0
