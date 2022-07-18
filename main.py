from config import config
import expvanillagan

def main() -> None:
    '''
    Main function to run experiments.
    Returns:
        None
    '''
    # Create directory structure for the experiment
    create_directory_structure = expvanillagan.DirectoryStructure(home_dir=config['device']['home directory'])
    create_directory_structure.create_directory_structure()

    # Create the experiments
    experiments = expvanillagan.Experiments(config=config)

    # Train the model
    experiments.train(verbose=False)
    
if __name__ == '__main__':
    main()