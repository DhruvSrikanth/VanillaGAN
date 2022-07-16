import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.io import read_image
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm

import typing
import warnings

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
            'in_block': nn.Sequential(*block(in_features=self.z_dim, out_features=128, normalize=False, regularize=False))
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
               
class Discriminator(nn.Module):
    def __init__(self, in_shape: tuple, n_blocks: int, name:str=None) -> None:
        '''
        Initialize the discriminator.
        Parameters:
            in_shape: The shape of the input image.
            n_blocks: The number of blocks in the discriminator.
            name: The name of the discriminator.
        Returns:
            None
        '''
        super(Discriminator, self).__init__()
        self.name = "Discriminator" if name is None else name
        self.in_shape = in_shape
        self.n_blocks = n_blocks
        
        def block(in_features, out_features, normalize=True, regularize=True) -> typing.List[nn.Module]:
            '''
            Each block that makes up the discriminator.
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
        
        # Starting intermediate latent dimension
        self.inter_dim = 512

        # Define input block
        self.in_block = nn.ModuleDict({
            'in_block': nn.Sequential(*block(in_features=int(np.prod(in_shape)), out_features=self.inter_dim, normalize=False, regularize=False))
        })

        # Define intermediate blocks
        self.inter_blocks = nn.ModuleDict({})
        in_dim = self.inter_dim
        for i in range(self.n_blocks):
            out_dim =  int(in_dim / 2)
            if out_dim >= 2:
                self.inter_blocks[f'inter_block_{i+1}'] = nn.Sequential(*block(in_features=in_dim, out_features=out_dim, normalize=True, regularize=True))
                in_dim = out_dim
            else:
                warnings.warn(f'Discriminator limited to {i} blocks')
                break
            
        # Define output block
        self.out_block = nn.ModuleDict({
            'out_block': nn.Sequential(
                nn.Linear(in_features=out_dim, out_features=1),
                nn.Sigmoid())
        })

        # Initialize weights
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m: nn.Module) -> None:
        '''
        Initialize the weights of the discriminator.
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
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Forward pass of the discriminator.
        Parameters:
            x: The input image.
        Returns:
            The output score.
        '''
        # Reshape input
        x = x.view(x.size(0), -1)

        # Input block
        x = self.in_block['in_block'](x)

        # Intermediate blocks
        for i in range(self.n_blocks):
            x = self.inter_blocks[f'inter_block_{i+1}'](x)
        
        # Output block
        validity = self.out_block['out_block'](x)
        
        return validity

class GAN(nn.Module):
    def __init__(self, z_dim: int, g_blocks: int, d_blocks: int, out_shape: tuple, device: torch.device, name: str=None) -> None:
        '''
        Initialize the GAN.
        Parameters:
            z_dim: The dimension of the latent space.
            g_blocks: The number of blocks in the generator.
            d_blocks: The number of blocks in the discriminator.
            out_shape: The shape of the output image.
            device: The device to use.
            name: The name of the GAN.
        Returns:
            None
        '''
        super(GAN, self).__init__()
        self.name = "GAN" if name is None else name
        self.z_dim = z_dim
        self.g_blocks = g_blocks
        self.d_blocks = d_blocks
        self.out_shape = out_shape
        self.device = device

        # Initialize generator
        self.generator = Generator(z_dim=self.z_dim, n_blocks=self.g_blocks, out_shape=self.out_shape, name='Generator').to(self.device)

        # Initialize discriminator
        self.discriminator = Discriminator(in_shape=self.out_shape, n_blocks=self.d_blocks, name='Discriminator').to(self.device)
    
    def discriminator_train_step(self, dataloader, discriminator_optimizer: torch.optim, discriminator_loss_fn: torch.nn.Module) -> float:
        '''
        Each training step of the discriminator.
        Parameters:
            dataloader: The dataloader to use.
            discriminator_optimizer: The optimizer to use.
            discriminator_loss_fn: The loss function to use.
        Returns:
            The loss of the discriminator.
        '''

        # Running loss
        running_loss = 0.0

        # Set the model to training mode
        self.discriminator.train()

        # Iteratate over the batches of the training dataset
        with tqdm(dataloader, desc=f'Training : {self.discriminator.name}') as pbar:
            for input, something in pbar:
                # Move data to device and configure input
                real_samples = Variable(input.type(torch.FloatTensor)).to(self.device)
                
                # Adversarial ground truths
                valid = Variable(torch.FloatTensor(input.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
                fake = Variable(torch.FloatTensor(input.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

                # Zero the gradients
                discriminator_optimizer.zero_grad()

                # Sample noise
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (input.shape[0], self.z_dim)))).to(self.device)

                # Forward pass through generator to get fake samples
                fake_sample = self.generator(z)

                # Forward pass to get validity scores of real and fake samples
                real_output = self.discriminator(real_samples)
                fake_output = self.discriminator(fake_sample.detach())

                # Compute loss: discriminator's ability to classify real from generated samples
                real_loss = discriminator_loss_fn(real_output, valid)
                fake_loss = discriminator_loss_fn(fake_output, fake)
                d_loss = (real_loss + fake_loss) / 2

                # Backward pass
                d_loss.backward()

                # Update the parameters of the discriminator
                discriminator_optimizer.step()

                # Update the running loss
                running_loss += d_loss.item()
                running_loss /= len(dataloader.dataset)
                
                # Update the progress bar
                pbar.set_postfix(discriminator_loss='{:.6f}'.format(running_loss))
                pbar.update()

        return running_loss
    
    def discriminator_train_loop(self, dataloader, discriminator_optimizer: torch.optim, discriminator_loss_fn: torch.nn.Module, k: int=1) -> float:
        '''
        Training loop of the discriminator.
        Parameters:
            k: The number of training steps to perform.
            dataloader: The dataloader to use.
            discriminator_optimizer: The optimizer to use.
            discriminator_loss_fn: The loss function to use.
        Returns:
            The loss of the discriminator.
        '''
        # Running loss
        running_loss = 0.0

        # For each training step of the discriminator
        for _ in range(k):
            # Perform a training step
            running_loss += self.discriminator_train_step(dataloader=dataloader, discriminator_optimizer=discriminator_optimizer, discriminator_loss_fn=discriminator_loss_fn)
            running_loss /= k
        
        return running_loss
    
    def generator_train_step(self, dataloader, generator_optimizer: torch.optim, generator_loss_fn: torch.nn.Module) -> float:
        '''
        Each training step of the generator.
        Parameters:
            dataloader: The dataloader to use.
            generator_optimizer: The optimizer to use.
            generator_loss_fn: The loss function to use.
        Returns:
            The loss of the generator.
        '''
        # Running loss
        running_loss = 0.0

        # Set the model to training mode
        self.generator.train()

        # Iteratate over the batches of the training dataset
        with tqdm(dataloader, desc=f'Training : {self.generator.name}') as pbar:
            for input, _ in pbar:

                # Adversarial ground truth
                valid = Variable(torch.FloatTensor(input.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)

                # Zero the gradients
                generator_optimizer.zero_grad()

                # Sample noise
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (input.shape[0], self.z_dim)))).to(self.device)

                # Forward pass to get fake samples
                fake_sample = self.generator(z)

                # Forward pass to get validity scores
                fake_output = self.discriminator(fake_sample)

                # Loss measures generator's ability to fool the discriminator
                g_loss = generator_loss_fn(fake_output, valid)

                # Backward pass
                g_loss.backward()

                # Update the parameters of the generator
                generator_optimizer.step()

                # Update the running loss
                running_loss += g_loss.item()
                running_loss /= len(dataloader.dataset)
                
                # Update the progress bar
                pbar.set_postfix(generator_loss='{:.6f}'.format(running_loss))
                pbar.update()

        return running_loss
    
    def train(self, dataloader, batch_size: int, generator_strategy: dict, discriminator_strategy: dict, epochs: int, sample_interval: int, sample_save_path: str, model_save_path: str, log_path: str) -> None:
        '''
        Training loop for the GAN.
        Parameters:
            dataloader: The dataloader to use.
            generator_strategy: The strategy to use for the generator (Must include 'optimizer' and 'criterion' keys).
            discriminator_strategy: The strategy to use for the discriminator (Must include 'optimizer', 'criterion' and 'discriminator_epochs' keys).
            epochs: The number of epochs to train for.
            sample_interval: The number of epochs between each sample generation to save.
            sample_save_path: The path to save the samples to.
            model_save_path: The path to save the model to.
            log_path: The path to save the logs to.
        Returns:
            None
        '''
        # Log results to tensorboard
        writer = SummaryWriter(f"{log_path}/experiment_1")
        
        # Add models to tensorboard
        writer.add_graph(self.generator, torch.randn(batch_size, self.z_dim))
        writer.add_graph(self.discriminator, torch.randn(batch_size, self.out_shape))
        
        # Training loop for the GAN
        for epoch in range(epochs):
            print('-' * 50)
            print(f'Starting Epoch {epoch + 1}/{epochs}:')

            # Train the discriminator
            discriminator_loss = self.discriminator_train_loop(k=discriminator_strategy['epochs'], dataloader=dataloader, discriminator_optimizer=discriminator_strategy['optimizer'], discriminator_loss_fn=discriminator_strategy['criterion'])

            # Train the generator
            generator_loss = self.generator_train_step(dataloader=dataloader, generator_optimizer=generator_strategy['optimizer'], generator_loss_fn=generator_strategy['criterion'])

            # Print the losses
            print(f'Epoch: {epoch + 1} - Generator loss: {generator_loss:.6f} - Discriminator loss: {discriminator_loss:.6f}')

            if epoch % sample_interval == 0:
                # Save the samples
                self.save_batch(save_path=sample_save_path, batch_size=batch_size, epoch=epoch, loss=generator_loss, n_images=4, writer=writer)
                print(f'Saved samples to {sample_save_path}.')
            
            # Save the model
            self.save_model(save_path=model_save_path, epoch=epoch, generator_loss=generator_loss, discriminator_loss=discriminator_loss)
            print(f'Saved model to {model_save_path}.')
            
            print('-' * 50)
        
        # Release the resource
        writer.close()
    
    def save_batch(self, save_path: str, epoch: int, batch_size: int, loss: int, writer, n_images: int=4) -> None:
        '''
        Save a batch of samples to a file.
        Parameters:
            save_path: The path to save the samples to.
            epoch: The epoch number.
            batch_size: The batch_size.
            loss: The loss of the generator.
            n_images: The number of images to save.
            writer: The tensorboard writer.
        '''
        # Sample noise
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.z_dim)))).to(self.device)

        # Forward pass to get fake sample
        fake_sample = self.generator(z)
        save_image(fake_sample.data[:n_images**2], f"{save_path}/generated_samples_epoch_{epoch}_loss_{loss}.png", nrow=n_images, normalize=True)

        # Read in and add to tensorboard
        img_grid = read_image(f"{save_path}/generated_samples_epoch_{epoch}_loss_{loss}.png")
        writer.add_image(f'sample_epoch_{epoch}', img_grid)
    
    def save_model(self, save_path: str, epoch: int, generator_loss: int, discriminator_loss: int) -> None:
        '''
        Save the model.
        Parameters:
            save_path: The path to save the model to.
            epoch: The epoch number.
            generator_loss: The loss of the generator.
            discriminator_loss: The loss of the discriminator.
        Returns:
            None
        '''
        # Save the generator
        torch.save(self.generator.state_dict(), f"{save_path}/generator_epoch_{epoch}_loss_{generator_loss}.pt")
        # Save the discriminator
        torch.save(self.discriminator.state_dict(), f"{save_path}/discriminator_epoch_{epoch}_loss_{discriminator_loss}.pt")

