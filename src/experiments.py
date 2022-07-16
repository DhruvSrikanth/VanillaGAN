from utils import print_config, decide_device, print_strategy
from data import get_dataloader
from model import GAN


import torch
from torchvision import transforms



class Experiments():
    def __init__(self, config: dict) -> None:
        '''
        Initialize the experiments.
        Parameters:
            config: The configuration dictionary.
        Returns:
            None
        '''
        self.config = config
        # Transforms to be applied to the data (convert to tensor, normalize, etc.)
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        # Compose the transforms
        transform = transforms.Compose(transform)

        # Get the data loader
        self.dataloaders = {
            'train': get_dataloader(type='train', batch_size=self.config['batch size'], num_workers=self.config['num workers'], transform=transform, samples=self.config['train samples']),
            'test': get_dataloader(type='test', batch_size=self.config['batch size'], num_workers=self.config['num workers'], transform=transform, samples=self.config['test samples']),
        }

    def train(self, verbose: bool=True) -> None:
        '''
        Train the model.
        Parameters:
            verbose: Whether to print the progress.
        Returns:
            None
        '''
        print(f"Training the model:\n{'-'*50}\n")
        # Print the configuration
        if verbose:
            print_config(self.config)
        
        # Decide the device
        use_device = decide_device(self.config['device'])
        device = torch.device(device=use_device)

        # Create the model
        model = GAN(z_dim=self.config['latent dimension'], g_blocks=self.config['generator blocks'], d_blocks=self.config['discriminator blocks'], out_shape=self.config['image shape'], device=device, name="GAN")
        if verbose:
            print(f"Given below is the model architecture: \n\t{model.generator}\n\t{model.discriminator}\n")


        # Define the strategy for training the generator
        generator_stategy = {
            'optimizer': torch.optim.Adam(model.generator.parameters(), lr=self.config['learning rate'], betas=(self.config['beta1'], self.config['beta2'])), 
            'criterion': torch.nn.BCELoss(),
        }

        # Define the strategy for training the discriminator
        discriminator_strategy = {
            'optimizer': torch.optim.Adam(model.discriminator.parameters(), lr=self.config['learning rate'], betas=(self.config['beta1'], self.config['beta2'])),
            'criterion': torch.nn.BCELoss(),
            'epochs': self.config['discriminator epochs'],
        }
        
        if verbose:
            print_strategy(strategy=generator_stategy, model=model.generator.name)
            print_strategy(strategy=discriminator_strategy, model=model.discriminator.name)

        
        # Train the model
        model.train(dataloader=self.dataloaders['train'], batch_size=self.config['batch size'], generator_strategy=generator_stategy, discriminator_strategy=discriminator_strategy, epochs=self.config['epochs'], sample_interval=self.config['sample interval'], sample_save_path=self.config['sample save path'], model_save_path=self.config['model save path'], log_path=self.config['log path'])
        print(f"Model trained:\n{'-'*50}\n")