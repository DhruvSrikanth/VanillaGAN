# Vanilla GAN Experiments

## Setting up the Vanilla GAN Experiments package

- Setup the environment - 
```
make install
```

- Allow the `.envrc` to activate vitual environment - 
```
direnv allow
```

- Install the requirements specified in `requirements.txt` - 
```
make install
```

## Using the Vanilla GAN Experiments package

For **every** experiment run:

- Clean the directory structure - 
```
make clean
```

- Reset the directory structure - 
```
make reset
```

- Define `config` parameters for package (each key in `config.py` - `config` dictionary **must** have value)

For **each** experiment run:

- Train the model - 
```
make experiments
```

- Visualize model training - inferred and true data distributions, losses [`Generator`, `Discriminator`, `Discriminator (Real Samples)`, `Discriminator (Fake Sampless)`] and generated samples (Uses `Tensorboard`) - 
```
make visualize
```

## Example For Using API - 

```
from config import config
from gan_experiments import Experiments, DirectoryStructure

def example() -> None:
    # Create directory structure for the experiment
    create_directory_structure = DirectoryStructure(home_dir=config['device']['home directory'])
    create_directory_structure.create_directory_structure()

    # Create the experiments
    experiments = Experiments(config=config)

    # Train the model
    experiments.train(verbose=False)
    
if __name__ == '__main__':
    example()
```
