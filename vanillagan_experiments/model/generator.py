import torch
import torch.nn as nn

import typing
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim: int, n_blocks: int, out_shape: tuple, name:str=None) -> None:
        '''
        Initialize the generator.
        Parameters:
            z_dim: The dimension of the latent space.
            n_blocks: The number of blocks in the generator.
            out_shape: The shape of the output image.
            name: The name of the generator.
        Returns:
            None
        '''
        super(Generator, self).__init__()
        self.name = "Generator" if name is None else name
        self.z_dim = z_dim
        self.n_blocks = n_blocks
        self.out_shape = out_shape

        def block(in_features: tuple, out_features: tuple, normalize: bool=True, regularize: bool=True) -> typing.List[nn.Module]:
            '''
            Each block that makes up the generator.
            Parameters:
                in_features: The input features of the block.
                out_features: The output features of the block.
                normalize: Whether or not to add batch normalization.
                regularize: Whether or not to add regularization.
            Returns:
                A list of modules that make up the block.
            '''
            # Fully connected layer
            layers = [nn.Linear(in_features=in_features, out_features=out_features)]

            if normalize:
                # Batch normalization layer
                layers.append(nn.BatchNorm1d(num_features=out_features, eps=0.8))
            
            # Activation layer
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            if regularize:
                # Regularization layer
                layers.append(nn.Dropout(p=0.5))
            
            return layers
        
        # Define input block
        self.in_block = nn.ModuleDict({
            'in_block': nn.Sequential(*block(in_features=self.z_dim, out_features= 2 * self.z_dim, normalize=False, regularize=False))
        })

        # Define intermediate blocks
        self.inter_blocks = nn.ModuleDict({})
        in_dim = 2 * self.z_dim
        for i in range(self.n_blocks):
            out_dim = 2 * in_dim
            self.inter_blocks[f'inter_block_{i+1}'] = nn.Sequential(*block(in_features=in_dim, out_features=out_dim, normalize=True, regularize=True))
            in_dim = out_dim
        
        # Define output block
        self.out_block = nn.ModuleDict({
            'out_block': nn.Sequential(
                nn.Linear(in_features=out_dim, out_features=int(np.prod(self.out_shape))),
                nn.Tanh())
        })

        # Initialize weights
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m: nn.Module) -> None:
        '''
        Initialize the weights of the generator.
        Parameters:
            m: The module to initialize.
        Returns:
            None
        '''
        if isinstance(m, nn.Linear):
            # Initialize weight to random normal
            nn.init.xavier_normal_(m.weight)
            # Initialize bias to zero
            nn.init.zeros_(m.bias)
        
    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Forward pass of the generator.
        Parameters:
            z: The latent space.
        Returns:
            The output sample.
        '''
        x = z

        # Input block
        x = self.in_block['in_block'](x)

        # Intermediate blocks
        for i in range(self.n_blocks):
            x = self.inter_blocks[f'inter_block_{i+1}'](x)
        
        # Output block
        x = self.out_block['out_block'](x)
        
        # Reshape output
        sample = x.view(x.size(0), *self.out_shape)
        
        return sample