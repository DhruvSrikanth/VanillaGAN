import os

class DirectoryStructure(object):
    def __init__(self, home_dir: str, verbose: bool=True):
        '''
        Initialize the directory structure.
        Parameters:
            home_dir: The home directory.
            verbose: Whether to print the progress.
        Returns:
            None
        '''
        if home_dir is None:
            raise ValueError('Please specify the home directory.')
        
        self.home_dir = home_dir
        self.verbose = verbose

        # Create the directory structure
        if self.verbose:
            print('-' * 50)
            print('Creating the directory structure...')
        self.create_directory_structure()
        if self.verbose:
            print('Directory structure created.')
            print('-' * 50 + '\n')

    def create_directory_structure(self) -> None:
        if self.home_dir not in os.listdir(os.getcwd()):
            os.system(f'mkdir {self.home_dir}')
        
        dir_list = ['data', 'weights', 'logs', 'samples']
        for dir_name in dir_list:
            if dir_name not in os.listdir(self.home_dir):
                os.system(f'mkdir {self.home_dir}/{dir_name}')
            else:
                # os.system(f'rm -rf {self.home_dir}/{dir_name}')
                # os.system(f'mkdir {self.home_dir}/{dir_name}')
                pass
        
        # Create the sub-directorys of 'data'
        dir_list = ['train', 'test']
        for dir_name in dir_list:
            if dir_name not in os.listdir(self.home_dir + 'data'):
                os.system(f'mkdir {self.home_dir}data/{dir_name}')
            else:
                # os.system(f'rm -rf {self.home_dir}/data/{dir_name}')
                pass