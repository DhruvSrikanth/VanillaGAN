from config import config
from vanillagan_experiments import Experiments, DirectoryStructure

def train_from_scratch_example() -> None:
    '''
    Train a model from scratch.
    Returns:
        None
    '''
    # Create directory structure for the experiment
    create_directory_structure = DirectoryStructure(home_dir=config['device']['home directory'])
    create_directory_structure.create_directory_structure()

    # Create the experiments
    experiments = Experiments(config=config)

    # Train the model
    experiments.train(verbose=False, checkpoint=None)

def train_from_checkpoint_example() -> None:
    '''
    Train a model from a checkpoint.
    Returns:
        None
    '''
    # Create the experiments
    experiments = Experiments(config=config)

    checkpoint = {
        'generator': './weights/generator_epoch_1_loss_22.0718.pt',
        'discriminator': './weights/discriminator_epoch_1_loss_0.1685.pt'
    }

    # Train the model
    experiments.train(verbose=False, checkpoint=checkpoint)
    
if __name__ == '__main__':
    example = 1 # 1 for train from scratch, 2 for train from checkpoint
    if example == 1:
        train_from_scratch_example()
    elif example == 2:
        train_from_checkpoint_example()