import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter

from ..utils import ImageTransforms

import numpy as np
from tqdm import tqdm
import time

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

class VanillaGAN(nn.Module):
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
        super(VanillaGAN, self).__init__()
        self.name = "Vanilla GAN" if name is None else name
        self.z_dim = z_dim
        self.g_blocks = g_blocks
        self.d_blocks = d_blocks
        self.out_shape = out_shape
        self.device = device

        # Initialize generator
        self.generator = Generator(z_dim=self.z_dim, n_blocks=self.g_blocks, out_shape=self.out_shape, name='Generator').to(self.device)

        # Initialize discriminator
        self.discriminator = Discriminator(in_shape=self.out_shape, n_blocks=self.d_blocks, name='Discriminator').to(self.device)
    
    def discriminator_train_step(self, real_samples: torch.FloatTensor, batch_size: int, discriminator_optimizer: torch.optim, discriminator_loss_fn: torch.nn.Module) -> dict:
        '''
        Each training step of the discriminator.
        Parameters:
            real_samples: The real samples. 
            batch_size: The batch size. 
            discriminator_optimizer: The optimizer to use.
            discriminator_loss_fn: The loss function to use.
        Returns:
            The loss of the discriminator containing the following keys:
                running_loss: The running loss of the discriminator.
                running_loss_fake: The running loss of the discriminator due to fake samples by the generator.
                running_loss_real: The running loss of the discriminator due to real samples from the data.
        '''

        # Running loss
        running_loss = 0.0
        running_real_loss = 0.0
        running_fake_loss = 0.0

        # Set the model to training mode
        self.discriminator.train()

        # Zero the gradients
        discriminator_optimizer.zero_grad()

        # Forward pass to get validity scores of real and fake samples
        generated_labels = self.discriminator(real_samples)
        # Compute loss: discriminator's ability to classify real from generated samples
        valid_labels = self.get_validity_labels(batch_size=batch_size, type='real')

        # Compute real loss
        real_loss = discriminator_loss_fn(generated_labels, valid_labels)
        running_real_loss = real_loss.item()

        # Backpropagate real loss
        real_loss.backward()

        # Update discriminator's weights
        discriminator_optimizer.step()

        # Zero the gradients
        discriminator_optimizer.zero_grad() 

        # Sample noise
        noise_tensor = self.sample_noise(batch_size=batch_size)
        
        # Forward pass through generator to get fake samples
        fake_samples = self.generator(noise_tensor)

        # Get the validity scores of the fake samples 
        fake_output = self.discriminator(fake_samples.detach())

        # Get fake labels
        fake_labels = self.get_validity_labels(batch_size=batch_size, type='fake')

        # Compute loss: discriminator's ability to classify real from generated samples
        fake_loss = discriminator_loss_fn(fake_output, fake_labels)
        running_fake_loss = fake_loss.item()

        # Backpropagate fake loss
        fake_loss.backward()

        # Update discriminator's weights
        discriminator_optimizer.step()

        d_loss = (real_loss + fake_loss) / 2
        # d_loss.backward()
        
        # Update the parameters of the discriminator
        # discriminator_optimizer.step()

        # Update the running loss
        running_loss = d_loss.item()

        return {'total': running_loss, 'real':running_real_loss, 'fake':running_fake_loss}
    
    def discriminator_train_loop(self, real_samples: torch.FloatTensor, batch_size: int, discriminator_optimizer: torch.optim, discriminator_loss_fn: torch.nn.Module, k: int=1) -> dict:
        '''
        Training loop of the discriminator.
        Parameters:
            k: The number of training steps to perform.
            real_samples: The real samples. 
            batch_size: The batch size. 
            discriminator_optimizer: The optimizer to use.
            discriminator_loss_fn: The loss function to use.
        Returns:
            The loss of the discriminator containing the following keys:
                running_loss: The running loss of the discriminator.
                running_loss_fake: The running loss of the discriminator due to fake samples by the generator.
                running_loss_real: The running loss of the discriminator due to real samples from the data.
        '''
        # Running loss
        running_loss = 0.0
        running_real_loss = 0.0
        running_fake_loss = 0.0

        # For each training step of the discriminator
        for _ in range(k):
            # Perform a training step
            loss = self.discriminator_train_step(batch_size=batch_size, real_samples=real_samples, discriminator_optimizer=discriminator_optimizer, discriminator_loss_fn=discriminator_loss_fn)
            running_loss += loss['total']
            running_real_loss += loss['real']
            running_fake_loss += loss['fake']
        running_loss /= k
        running_real_loss /= k
        running_fake_loss /= k
        
        return {'total': running_loss, 'real':running_real_loss, 'fake':running_fake_loss}
    
    def generator_train_step(self, batch_size: int, generator_optimizer: torch.optim, generator_loss_fn: torch.nn.Module) -> float:
        '''
        Each training step of the generator.
        Parameters: 
            batch_size: The batch size.
            generator_optimizer: The optimizer to use.
            generator_loss_fn: The loss function to use.
        Returns:
            The loss of the generator.
        '''
        # Running loss
        running_loss = 0.0

        # Set the model to training mode
        self.generator.train()

        # Zero the gradients
        generator_optimizer.zero_grad()

        # Sample noise
        noise_tensor = self.sample_noise(batch_size=batch_size)

        # Forward pass to get fake samples
        fake_samples = self.generator(noise_tensor)

        # Forward pass to get validity scores
        fake_output = self.discriminator(fake_samples.detach())

        # Get labels for valid samples
        valid_labels = self.get_validity_labels(batch_size=batch_size, type='real')

        # Loss measures generator's ability to fool the discriminator
        g_loss = generator_loss_fn(fake_output, valid_labels)

        # Backward pass
        g_loss.backward()

        # Update the parameters of the generator
        generator_optimizer.step()

        # Update the running loss
        running_loss += g_loss.item()

        return running_loss
    
    def generator_train_loop(self, batch_size: int, generator_optimizer: torch.optim, generator_loss_fn: torch.nn.Module, l: int=1) -> float:
        '''
        Training loop of the generator.
        Parameters:
            batch_size: The batch size.
            l: The number of training steps to perform.
            generator_optimizer: The optimizer to use.
            generator_loss_fn: The loss function to use.
        Returns:
            The loss of the generator.
        '''
        # Running loss
        running_loss = 0.0

        # For each training step of the generator
        for _ in range(l):
            # Perform a training step
            running_loss += self.generator_train_step(batch_size=batch_size, generator_optimizer=generator_optimizer, generator_loss_fn=generator_loss_fn)
        running_loss /= l
        
        return running_loss

    def train(self, dataloader, generator_strategy: dict, discriminator_strategy: dict, epochs: int, starting_epoch :int, sample_interval: int, sample_save_path: str, model_save_path: str, log_path: str, experiment_number: int) -> None:
        '''
        Training loop for the GAN.
        Parameters:
            dataloader: The dataloader to use.
            generator_strategy: The strategy to use for the generator (Must include 'optimizer' and 'criterion' keys).
            discriminator_strategy: The strategy to use for the discriminator (Must include 'optimizer', 'criterion' and 'discriminator_epochs' keys).
            epochs: The number of epochs to train for.
            starting_epoch: The epoch to start training from.
            sample_interval: The number of epochs between each sample generation to save.
            sample_save_path: The path to save the samples to.
            model_save_path: The path to save the model to.
            log_path: The path to save the logs to.
            experiment_number: The experiment number.
        Returns:
            None
        '''
        # Log results to tensorboard
        writer = SummaryWriter(f"{log_path}/experiment_{experiment_number}")
        
        
        # Add models to tensorboard
        # self.visualize_model(batch_size=batch_size, writer=writer)
        
        # Training loop for the GAN
        for epoch in range(starting_epoch, epochs):
            print('-' * 50)
            print(f'Starting Epoch {epoch + 1}/{epochs}:')
            start_time = time.time()

            # For each batch in the dataloader
            with tqdm(dataloader, desc=f'Training : {self.name}') as pbar:
                for imgs, _ in pbar:
                    # Move data to device and configure input
                    real_samples = Variable(imgs.type(torch.FloatTensor)).to(self.device)
                    
                    # Recompute batch size for current mini batch
                    batch_size = real_samples.size(0)

                    # Train the discriminator
                    discriminator_loss = self.discriminator_train_loop(k=discriminator_strategy['epochs'], real_samples=real_samples, batch_size=batch_size, discriminator_optimizer=discriminator_strategy['optimizer'], discriminator_loss_fn=discriminator_strategy['criterion'])
                    
                    # Train the generator
                    generator_loss = self.generator_train_loop(l=generator_strategy['epochs'], batch_size=batch_size, generator_optimizer=generator_strategy['optimizer'], generator_loss_fn=generator_strategy['criterion'])

                    # Update the progress bar
                    pbar.set_postfix(Losses=f"g_loss: {generator_loss:.4f} - d_loss: {discriminator_loss['total']:.4f}")
                    pbar.update()
            
            # Print the losses
            print(f"Epoch: {epoch + 1} - Generator loss: {generator_loss:.6f} - Discriminator loss: {discriminator_loss['total']:.6f} - Time Taken: {time.time() - start_time:.2f}")
            # Add the losses to tensorboard
            self.visualize_loss(epoch=epoch, writer=writer, generator_loss=generator_loss, discriminator_loss=discriminator_loss['total'], discriminator_loss_real=discriminator_loss['real'], discriminator_loss_fake=discriminator_loss['fake'])

            # visualize distribution
            self.visualize_distribution(batch_size=batch_size, dataloader=dataloader, epoch=epoch, writer=writer)

            if epoch % sample_interval == 0:
                # Save the samples
                self.save_batch(save_path=sample_save_path, batch_size=batch_size, epoch=epoch, loss=generator_loss, n_images=4, writer=writer)
                print(f'Saved samples to {sample_save_path}.')
            
            # Save the model
            self.save_model(save_path=model_save_path, epoch=epoch, generator_loss=generator_loss, discriminator_loss=discriminator_loss)
            print(f'Saved model to {model_save_path}.')
            
            print('-' * 50 + '\n')
        
        # Release the resource
        writer.close()
    
    def get_validity_labels(self, batch_size: int, type: str) -> torch.Tensor:
        '''
        Get the labels for the validity of the samples.
        Parameters:
            batch_size: The batch size.
            type: The type of the samples. Valid types are 'real' and 'fake'.
        Returns:
            The labels for the validity of the samples.
        '''
        # Get the labels for the validity of the samples
        if type == 'real':
            return Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(self.device)
        elif type == 'fake':
            return Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(self.device)
        else:
            raise ValueError(f'Invalid type: {type}, Valid types include: real, fake.')

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        '''
        Sample noise tensor from a normal distribution.
        Parameters:
            batch_size: The number of samples to sample.
        Returns:
            The sampled noise.
        '''
        # Sample noise from a normal distribution
        # z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.z_dim)))).to(self.device)

        # Sample noise from a multivariate distribution
        # z = Variable(torch.FloatTensor(np.random.multivariate_normal([0]*self.z_dim, np.eye(self.z_dim), batch_size)).to(self.device))

        # sample noise from a uniform distribution
        z = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (batch_size, self.z_dim)))).to(self.device)

        return z

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
        z = self.sample_noise(batch_size=batch_size)

        # Set the model to evaluation mode
        self.generator.eval()
        
        # Forward pass to get fake sample
        fake_sample = self.generator(z)
        save_image(fake_sample.data[:n_images**2], f"{save_path}/samples_epoch_{epoch}_loss_{loss}.png", nrow=n_images, normalize=True)

        # Read in and add to tensorboard
        img_grid = read_image(f"{save_path}/samples_epoch_{epoch}_loss_{loss}.png", mode=torchvision.io.ImageReadMode.GRAY)
        writer.add_image(f'Samples', img_grid, global_step=epoch)
    
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
        torch.save(self.generator.state_dict(), f"{save_path}/generator_epoch_{epoch}_loss_2{round(generator_loss, 4)}.pt")
        # Save the discriminator
        torch.save(self.discriminator.state_dict(), f"{save_path}/discriminator_epoch_{epoch}_loss_{round(discriminator_loss['total'], 4)}.pt")
    
    def load_model(self, generator_model_path: str, discriminator_model_path: str) -> None:
        '''
        Load the model.
        Parameters:
            generator_model_path: The path to the generator model.
            discriminator_model_path: The path to the discriminator model.
        Returns:
            None
        '''
        # Load the generator
        self.generator.load_state_dict(torch.load(generator_model_path))
        self.generator.to(self.device)

        # Load the discriminator
        self.discriminator.load_state_dict(torch.load(discriminator_model_path))
        self.discriminator.to(self.device)

    def visualize_distribution(self, epoch: int, batch_size: int, dataloader: object, writer: object) -> None:
        '''
        Visualize the distribution of real and inferred data.
        Parameters:
            epoch: The epoch number.
            batch_size: The batch_size.
            dataloader: The dataloader to use.
            writer: The tensorboard writer.
        Returns:
            None
        '''
        # Get the image transformations
        transforms = ImageTransforms()

        # Sample noise
        z = self.sample_noise(batch_size=batch_size)

        # Set the model to evaluation mode
        self.generator.eval()
        
        # Forward pass to get fake sample
        fake_sample = self.generator(z)
        writer.add_histogram('Inferred Distribution', values=transforms.normalize_tensor(fake_sample), global_step=epoch, bins=256)

        # Get real sample from dataloader
        real_sample = next(iter(dataloader))[0]
        real_sample = Variable(torch.FloatTensor(real_sample)).to(self.device)
        writer.add_histogram('Actual Distribution', values=transforms.normalize_tensor(real_sample), global_step=epoch, bins=256)
    
    def visualize_model(self, batch_size: int, writer):
        '''
        Visualize the model.
        Parameters:
            batch_size: The batch size to use.
            writer: The tensorboard writer to use.
        Returns:
            None
        '''
        generator_input = self.sample_noise(batch_size=batch_size)
        writer.add_graph(self.generator, generator_input)

        self.generator.eval()
        discriminator_input = self.generator(generator_input)
        writer.add_graph(self.discriminator, discriminator_input)
    
    def visualize_loss(self, epoch: int, generator_loss: float, discriminator_loss: float, discriminator_loss_real: float, discriminator_loss_fake: float, writer: object) -> None:
        '''
        Visualize the loss.
        Parameters:
            epoch: The epoch number.
            generator_loss: The loss of the generator.
            discriminator_loss: The loss of the discriminator.
            discriminator_loss_real: The loss of the discriminator for the real data.
            discriminator_loss_fake: The loss of the discriminator for the fake data.
            writer: The tensorboard writer.
        Returns:
            None
        '''
        # writer.add_scalar('Generator loss', generator_loss, epoch)
        # writer.add_scalar('Discriminator loss', discriminator_loss, epoch)
        writer.add_scalars('Loss Curves', {
            'Generator loss': generator_loss, 
            'Discriminator loss (total)': discriminator_loss, 
            'Discriminator loss (real samples)': discriminator_loss_real,
            'Discriminator loss (fake samples)': discriminator_loss_fake
        }, epoch)
