import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import tqdm

import typing
import warnings

class Generator(nn.Module):
    def __init__(self, z_dim: int, n_blocks: int, out_shape: tuple, name:str=None) -> None:
        super(Generator, self).__init__()
        self.name = "Generator" if name is None else name
        self.z_dim = z_dim
        self.n_blocks = n_blocks
        self.out_shape = out_shape

        def block(in_features: tuple, out_features: tuple, normalize:bool=True, regularize:bool=True) -> typing.List[nn.Module]:
            layers = [nn.Linear(in_features=in_features, out_feature=out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_features, eps=0.8))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            if regularize:
                layers.append(nn.Dropout(p=0.5))
            return layers
        
        self.in_block = nn.ModuleDict({
            'in_block': nn.Sequential(*block(in_features=self.z_dim, out_features=128, normalize=False, regularize=False))
        })

        self.inter_blocks = nn.ModuleDict({})
        in_dim = 2 * self.z_dim
        for i in range(self.n_blocks):
            out_dim = 2 * in_dim
            self.inter_blocks[f'inter_block_{i+1}'] = nn.Sequential(*block(in_features=in_dim, out_features=out_dim, normalize=True, regularize=True))
            in_dim = out_dim
        
        self.out_block = nn.ModuleDict({
            'out_block': nn.Sequential(
                nn.Linear(in_features=out_dim, out_features=int(np.prod(self.out_shape))),
                nn.Tanh())
        })

        # Initialize weights
        self.apply(self._init_weights)

        @torch.no_grad()
        def _init_weights(self, m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = z
            x = self.in_block['in_block'](x)
            for i in range(self.n_blocks):
                x = self.inter_blocks[f'inter_block_{i+1}'](x)
            x = self.out_block['out_block'](x)
            sample = x.view(x.size(0), *self.out_shape)
            return sample
               
class Discriminator(nn.Module):
    def __init__(self, in_shape: tuple, n_blocks: int, name:str=None) -> None:
        super(Discriminator, self).__init__()
        self.name = "Discriminator" if name is None else name
        self.in_shape = in_shape
        self.n_blocks = n_blocks

        def block(in_features, out_features, normalize=True, regularize=True) -> typing.List[nn.Module]:
            layers = [nn.Linear(in_features=in_features, out_feature=out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_features, eps=0.8))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            if regularize:
                layers.append(nn.Dropout(p=0.5))
            return layers
        
        self.inter_dim = 512
        self.in_block = nn.ModuleDict({
            'in_block': nn.Sequential(*block(in_features=int(np.prod(in_shape)), out_features=self.inter_dim, normalize=False, regularize=False))
        })

        self.inter_blocks = nn.ModuleDict({})
        in_dim = self.inter_dim
        for i in range(self.n_blocks):
            out_dim =  in_dim / 2
            if out_dim >= 2:
                self.inter_blocks[f'inter_block_{i+1}'] = nn.Sequential(*block(in_features=in_dim, out_features=out_dim, normalize=True, regularize=True))
                in_dim = out_dim
            else:
                warnings.warn(f'Discriminator limited to {i} blocks')
                break
            
        
        self.out_block = nn.ModuleDict({
            'out_block': nn.Sequential(
                nn.Linear(in_features=out_dim, out_features=1),
                nn.Sigmoid())
        })

        # Initialize weights
        self.apply(self._init_weights)

        @torch.no_grad()
        def _init_weights(self, m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.view(x.size(0), -1)
            x = self.in_block['in_block'](x)
            for i in range(self.n_blocks):
                x = self.inter_blocks[f'inter_block_{i+1}'](x)
            validity = self.out_block['out_block'](x)
            return validity

class GAN(nn.Module):
    def __init__(self, z_dim: int, g_blocks: int, d_blocks: int, out_shape: tuple, device: torch.device, name: str=None) -> None:
        super(GAN).__init__()
        self.name = "GAN" if name is None else name
        self.z_dim = z_dim
        self.g_blocks = g_blocks
        self.d_blocks = d_blocks
        self.out_shape = out_shape
        self.device = device

        self.generator = Generator(z_dim=self.z_dim, n_blocks=self.g_blocks, out_shape=self.out_shape, name='Generator')
        self.discriminator = Discriminator(in_shape=self.out_shape, n_blocks=self.d_blocks, name='Discriminator')        
    
    def discriminator_train_step(self, dataloader: torch.utils.data.Dataloader, discriminator_optimizer: torch.optim, discriminator_loss_fn: torch.nn.Module) -> float:
        running_loss = 0.0

        # Set the model to training mode
        self.discriminator.train()

        # Iteratate over the batches of the training dataset
        with tqdm(dataloader, desc=f'Training : {self.discriminator.name}') as pbar:
            for input, _ in pbar:  

                # Move data to device
                input = input.to(self.device)
                # Configure input
                real_input = Variable(input.type(torch.Tensor))
                
                # Adversarial ground truths
                valid = Variable(torch.Tensor(input.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
                fake = Variable(torch.Tensor(input.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

                # Zero the gradients
                discriminator_optimizer.zero_grad()

                # Sample noise
                z = Variable(torch.Tensor(np.random.normal(0, 1, (input.shape[0], self.z_dim)))).to(self.device)

                # Forward pass to get fake inputs
                fake_input = self.generator(z)

                # Forward pass to get validity scores
                real_output = self.discriminator(real_input)
                fake_output = self.discriminator(fake_input.detach())

                # Compute loss: discriminator's ability to classify real from generated samples
                real_loss = discriminator_loss_fn(real_output, valid)
                fake_loss = discriminator_loss_fn(fake_output, fake)
                d_loss = (real_loss + fake_loss) / 2

                # Backward pass
                d_loss.backward()

                # Update the parameters
                discriminator_optimizer.step()

                # Update the running loss
                running_loss += d_loss.item()
                running_loss /= len(dataloader.dataset)
                
                # Update the progress bar
                pbar.set_postfix(discriminator_loss='{:.6f}'.format(running_loss))
                pbar.update()

        return running_loss
    
    def discriminator_train_loop(self, k: int=1, dataloader: torch.utils.data.Dataloader=None, discriminator_optimizer: torch.optim=None, discriminator_loss_fn: torch.nn.Module=None) -> float:
        running_loss = 0.0
        for _ in range(k):
            running_loss += self.discriminator_train_step(dataloader=dataloader, discriminator_optimizer=discriminator_optimizer, discriminator_loss_fn=discriminator_loss_fn)
            running_loss /= k
        return running_loss
    
    def generator_train_step(self, dataloader: torch.utils.data.Dataloader, generator_optimizer: torch.optim, generator_loss_fn: torch.nn.Module) -> float:
        running_loss = 0.0

        # Set the model to training mode
        self.generator.train()

        # Iteratate over the batches of the training dataset
        with tqdm(dataloader, desc=f'Training : {self.generator.name}') as pbar:
            for input, _ in pbar:  

                # Move data to device
                input = input.to(self.device)
                
                # Adversarial ground truth
                valid = Variable(torch.Tensor(input.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)

                # Zero the gradients
                generator_optimizer.zero_grad()

                # Sample noise
                z = Variable(torch.Tensor(np.random.normal(0, 1, (input.shape[0], self.z_dim)))).to(self.device)

                # Forward pass to get fake inputs
                fake_input = self.generator(z)

                # Forward pass to get validity scores
                fake_output = self.discriminator(fake_input)

                # Loss measures generator's ability to fool the discriminator
                g_loss = generator_loss_fn(fake_output, valid)

                g_loss.backward()
                generator_optimizer.step()

                # Update the running loss
                running_loss += g_loss.item()
                running_loss /= len(dataloader.dataset)
                
                # Update the progress bar
                pbar.set_postfix(generator_loss='{:.6f}'.format(running_loss))
                pbar.update()

        return running_loss
    
    def train(self, dataloader: torch.utils.data.Dataloader, generator_strategy: dict, discriminator_strategy: dict, epochs: int, sample_interval: int, sample_save_path: str, model_save_path: str) -> None:
        for epoch in range(epochs):
            print('-' * 50)
            print(f'Starting Epoch {epoch + 1}/{epochs}:')

            # Train the discriminator
            discriminator_loss = self.discriminator_train_loop(k=discriminator_strategy['disciminator_epochs'], dataloader=dataloader, discriminator_optimizer=discriminator_strategy['optimizer'], discriminator_loss_fn=discriminator_strategy['criterion'])

            # Train the generator
            generator_loss = self.generator_train_step(dataloader=dataloader, generator_optimizer=generator_strategy['optimizer'], generator_loss_fn=generator_strategy['criterion'])

            # Print the losses
            print(f'Epoch: {epoch + 1} - Generator loss: {generator_loss:.6f} - Discriminator loss: {discriminator_loss:.6f}')

            if epoch % sample_interval == 0:
                input_shape = dataloader[0]
                self.save_batch(save_path=sample_save_path, input_shape=input_shape, epoch=epoch, loss=generator_loss, n_images=5)
                print(f'Saved samples to {sample_save_path}.')
            
            self.save_model(save_path=model_save_path, epoch=epoch, loss=generator_loss, discriminator_loss=discriminator_loss)
            print(f'Saved model to {model_save_path}.')
            
            print('-' * 50)
    
    def save_batch(self, save_path: str, epoch: int, input_shape: tuple, loss: int, n_images: int=5) -> None:
        # Sample noise
        z = Variable(torch.Tensor(np.random.normal(0, 1, (input_shape.shape[0], self.z_dim)))).to(self.device)

        # Forward pass to get fake inputs
        fake_input = self.generator(z)
        save_image(fake_input.data[:n_images**2], f"{save_path}/generated_samples_epoch_{epoch}_loss_{loss}.png", nrow=n_images, normalize=True)
    
    def save_model(self, save_path: str, epoch: int, generator_loss: int, discriminator_loss: int) -> None:
        torch.save(self.generator.state_dict(), f"{save_path}/generator_epoch_{epoch}_loss_{generator_loss}.pth")
        torch.save(self.discriminator.state_dict(), f"{save_path}/discriminator_epoch_{epoch}_loss_{discriminator_loss}.pth")
                

