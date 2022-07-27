from .dataloader import get_dataloader
from .model import VanillaGAN
from .utils import Config, ImageTransforms

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
        self.config = Config(config=config)

        # Transforms to be applied to the data (convert to tensor, normalize, etc.)
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        # Compose the transforms
        transform = transforms.Compose(transform)

        # Get the data loader
        self.dataloaders = {
            'train': get_dataloader(type='train', batch_size=self.config.data_config['batch size'], num_workers=self.config.data_config['num workers'], transform=transform, samples=self.config.data_config['train samples']),
            'test': get_dataloader(type='test', batch_size=self.config.data_config['batch size'], num_workers=self.config.data_config['num workers'], transform=transform, samples=self.config.data_config['test samples']),
        }

    def train(self, verbose: bool=True, checkpoint: dict=None) -> None:
        '''
        Train the model.
        Parameters:
            verbose: Whether to print the progress.
            checkpoint: The checkpoint to load the models from.
        Returns:
            None
        '''
        print(f"Training the model:\n{'-'*50}\n")
        # Print the configuration
        if verbose:
            self.config.print_config()
        
        # Decide the device
        use_device = self.config.decide_device()
        device = torch.device(device=use_device)

        # Create the model
        model = VanillaGAN(z_dim=self.config.hyperparameters['latent dimension'], g_blocks=self.config.generator_config['generator blocks'], d_blocks=self.config.discriminator_config['discriminator blocks'], out_shape=self.config.data_config['image shape'], device=device, name="Vanilla GAN")
        starting_epoch = 0
        if checkpoint is not None:
            if 'generator' in checkpoint and 'discriminator' in checkpoint and 'epoch' in checkpoint:
                starting_epoch = checkpoint['epoch'] + 1
                model.load_model(generator_model_path=checkpoint['generator'], discriminator_model_path=checkpoint['discriminator'])
            else:
                raise ValueError("The checkpoint must contain the generator and discriminator models paths.")
        if verbose:
            print(f"Given below is the model architecture: \n\t{model.generator}\n\t{model.discriminator}\n")


        # Define the strategy for training the generator
        generator_stategy = {
            'optimizer': torch.optim.Adam(model.generator.parameters(), lr=self.config.hyperparameters['learning rate'], betas=(self.config.hyperparameters['beta1'], self.config.hyperparameters['beta2'])), 
            'criterion': torch.nn.BCELoss(),
            'epochs': self.config.hyperparameters['generator epochs'],
        }

        # Define the strategy for training the discriminator
        discriminator_strategy = {
            'optimizer': torch.optim.Adam(model.discriminator.parameters(), lr=self.config.hyperparameters['learning rate'], betas=(self.config.hyperparameters['beta1'], self.config.hyperparameters['beta2'])),
            'criterion': torch.nn.BCELoss(),
            'epochs': self.config.hyperparameters['discriminator epochs'],
        }
        
        if verbose:
            self.config.print_strategy(strategy=generator_stategy, model=model.generator.name)
            self.config.print_strategy(strategy=discriminator_strategy, model=model.discriminator.name)

        
        # Train the model
        model.train(dataloader=self.dataloaders['train'], generator_strategy=generator_stategy, discriminator_strategy=discriminator_strategy, epochs=self.config.hyperparameters['epochs'], starting_epoch=starting_epoch, sample_interval=self.config.save_config['sample interval'], sample_save_path=self.config.save_config['sample save path'], model_save_path=self.config.save_config['model save path'], log_path=self.config.log_config['log path'], experiment_number=self.config.log_config['experiment number'])
        print(f"Model trained:\n{'-'*50}\n")