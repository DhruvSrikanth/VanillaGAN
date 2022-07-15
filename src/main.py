from config import config
from experiments import Experiments

def main() -> None:
    '''
    Main function to run experiments.
    Returns:
        None
    '''
    # Create the experiments
    experiments = Experiments(config)

    # Train the model
    experiments.train(verbose=True)
    
if __name__ == '__main__':
    main()