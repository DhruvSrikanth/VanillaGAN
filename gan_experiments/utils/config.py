import torch

class Config(object):
    def __init__(self, config: dict) -> None:
        '''
        Initialize the configuration.
        Parameters:
            config: The configuration dictionary.
        Returns:
            None
        '''
        self.config_keys = {
            'device' : [
                'home directory',
                'device type', 
                'initial seed'
            ], 
            'data' : [
                'batch size', 
                'num workers',
                'train samples',
                'test samples', 
                'image shape'
            ], 
            'generator' : [
                'generator blocks', 
            ], 
            'discriminator' : [
                'discriminator blocks', 
            ], 
            'hyperparameters' : [
                'latent dimension', 
                'learning rate', 
                'beta1', 
                'beta2',  
                'epochs', 
                'discriminator epochs', 
                'generator epochs'
            ], 
            'save' : [
                'sample interval', 
                'sample save path', 
                'model save path'
            ], 
            'log' : [
                'log path', 
                'experiment number'
            ]
        }
        
        def check_config(config: dict, type=None) -> None:
            '''
            Check if the configuration is valid.
            Parameters:
                config: The configuration dictionary.
                type: The type of the configuration.
            Returns:
                None
            '''
            # Check for all keys in the config file
            keys = self.config_keys.keys() if not type else self.config_keys[type]
            for key in keys:
                if key not in config:
                    raise Exception(f"{key} not found in the config file.")
        
        def update_config_directories() -> None:
            '''
            Update the config directories.
            Parameters:
                None
            Returns:
                None
            '''
            home_dir = self.config['device']['home directory']
            self.config['save']['sample save path'] = f"{home_dir}/{self.config['save']['sample save path']}"
            self.config['save']['model save path'] = f"{home_dir}/{self.config['save']['model save path']}"
            self.config['log']['log path'] = f"{home_dir}/{self.config['log']['log path']}"
        

        
        check_config(config=config, type=None)
        self.config = config

        check_config(config=config['device'], type='device')
        self.device_config = self.config['device']
        
        check_config(config=config['data'], type='data')
        self.data_config = self.config['data']

        check_config(config=config['generator'], type='generator')
        self.generator_config = self.config['generator']

        check_config(config=config['discriminator'], type='discriminator')
        self.discriminator_config = self.config['discriminator']

        check_config(config=config['hyperparameters'], type='hyperparameters')
        self.hyperparameters = self.config['hyperparameters']

        check_config(config=config['save'], type='save')
        self.save_config = self.config['save']
        
        check_config(config=config['log'], type='log')
        self.log_config = self.config['log']

        update_config_directories()
     
    def print_config(self) -> None:
        """
        Print the configuration.
        Parameters:
            None
        Returns:
            None
        """
        print(f"\nGiven below is the configuration present in the config file for the {self.config['device']['device type'].lower()}:")
        i = 1
        for config_param, config_value in self.config.items():
            print(f"\t{i}. Parameter : {config_param} | Value : {config_value}")
            i += 1
        print("\n")

    
    def print_strategy(self, strategy: dict, model: str) -> None:
        """
        Print the strategy.
        Parameters:
            strategy: (type dict) strategy.
            model: (type str) model name.
        Returns:
            None
        """

        names = {}
        try:
            names['Optimizer'] = type(strategy['optimizer']).__name__
        except KeyError:
            raise Exception('No optimizer specified.')
        try:
            names['Criterion'] = type(strategy['criterion']).__name__
        except KeyError:
            raise Exception('No criterion specified.')

        print(f"\nGiven below is the strategy for training the {model} model:")
        i = 1
        for name, value in names.items():
            print(f"\t{i}. {name} : {value}")
            i += 1
        print("\n")

    def decide_device(self) -> str:
        '''
        Decide which device to use.
        Parameters:
            None
        Returns:
            device: (type str) device to use
        '''
        device = self.config['device']['device type'].lower()
        if device == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
