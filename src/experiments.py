from utils import print_config, decide_device, print_strategy
from data import get_dataloader
from model import NeuralNetwork

import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict

class Experiments():
    def __init__(self, config, adversarial=False, frequency_disentanglement=False):
        self.config = config
        self.adversarial = adversarial
        self.frequency_disentanglement = frequency_disentanglement
        if self.frequency_disentanglement:
            fc = self.config['cutoff frequency'] if 'cutoff frequency' in self.config else 0.1
            b = self.config['bandwidth'] if 'bandwidth' in self.config else 0.08
            self.frequency_filter = FrequencyFilters(method=self.config['frequency filter method'], fc=fc, b=b)

        # Transforms to be applied to the data (convert to tensor, normalize, etc.)
        transform = []
        if self.adversarial:
            pass
        if self.frequency_disentanglement:
            transform.append(self.frequency_filter)
        
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        # Compose the transforms
        transform = transforms.Compose(transform)

        # Get the data loader
        self.dataloaders = {
            'train': get_dataloader(type='train', batch_size=self.config['batch size'], shuffle=True, num_workers=self.config['num workers'], transform=transform),
            'test': get_dataloader(type='test', batch_size=self.config['batch size'], shuffle=False, num_workers=self.config['num workers'], transform=transform)
        }

        self.observations = OrderedDict({})

    def train_model(self, verbose=True, reproduce=False):
        print(f"Training the model:\n{'-'*50}\n")
        # Print the configuration
        if verbose:
            print_config(self.config)

        # Set the seed
        if reproduce:
            torch.manual_seed(self.config['initial seed'])
            if self.config['device'].lower() == 'cuda':
                torch.cuda.manual_seed(self.config['initial seed'])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Decide the device
        use_device = decide_device(self.config['device'])
        device = torch.device(device=use_device)

        

        # Create the model and move it to the device
        model = NeuralNetwork().to(device)
        if verbose:
            print(f"Given below is the model architecture: \n\t{model}\n")


        # Define the strategy for training the model
        strategy = {
            'optimizer': optim.Adadelta(model.parameters(), lr=self.config['learning rate']), 
            'scheduler': optim.lr_scheduler.StepLR(optimizer=optim.Adadelta(model.parameters(), lr=self.config['learning rate']), step_size=self.config['step size'], gamma=self.config['gamma']), 
            'criterion': F.nll_loss,
        }
        if verbose:
            print_strategy(strategy)

        
        # Train the model
        performance = train(model=model, device=device, dataloaders=self.dataloaders, strategy=strategy, epochs=self.config['epochs'], save_frequency=self.config['save frequency'], adversarial=self.adversarial, frequency_disentanglement=self.frequency_disentanglement)

        # Update the observations
        self.observations['train'] = performance

        print(f"Model trained:\n{'-'*50}\n")
    
    def test_model(self, verbose=True, reproduce=False):
        print(f"Testing the model\n{'-'*50}\n")
        
        # Print the configuration
        if verbose:
            print_config(self.config)

        # Set the seed
        if reproduce:
            torch.manual_seed(self.config['initial seed'])
            if self.config['device'].lower() == 'cuda':
                torch.cuda.manual_seed(self.config['initial seed'])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Decide the device
        use_device = decide_device(self.config['device'])
        device = torch.device(device=use_device)

        

        # Create the model and move it to the device
        model = NeuralNetwork().to(device)
        if verbose:
            print(f"Given below is the model architecture: \n\t{model}\n")
        
        # Get the best model path from the training observations
        try:
            best_model_path = self.observations['train']['best model path'][0]
            best_model_accuracy = self.observations['train']['best model accuracy'][0]
            for i in range(len(self.observations['train']['epoch'])):
                if self.observations['train']['accuracy'][i] > best_model_accuracy:
                    best_model_path = self.observations['train']['best model path'][i]
                    best_model_accuracy = self.observations['train']['best model accuracy'][i]
        except KeyError:
            print("No model path found. Please train the model first.")
            return
        
        # Load the best model
        model.load_state_dict(torch.load(best_model_path))


        # Define the strategy for testing the model
        strategy = {
            'criterion': F.nll_loss
        }

        if verbose:
            print_strategy(strategy)

        
        
        # Evaluate the model
        loss, accuracy = test(model=model, device=device, dataloader=self.dataloaders['test'], loss_fn=strategy['criterion'])

        # Update the observations
        self.observations['test'] = {
            'trained model path': best_model_path, 
            'trained model accuracy': best_model_accuracy, 
            'test accuracy': accuracy, 
        }

        # Print metrics
        print('Validation loss: {:.4f}, Accuracy: {:.4f} %\n'.format(loss, accuracy)) 

        print(f"Model tested:\n{'-'*50}\n")
