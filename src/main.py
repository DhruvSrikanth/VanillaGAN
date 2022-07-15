from config import config
from experiments import Experiments

def main():
    # Create the experiments object
    baseline = Experiments(config, adversarial=False, frequency_disentanglement=False)
    # Train the model (without adversarial noise applied)
    baseline.train_model(verbose=True, reproduce=True)
    # Test this model
    baseline.test_model(verbose=True, reproduce=True)
    # Get the observations
    baseline_observations = baseline.observations

    # Create the experiments object
    adversarial = Experiments(config, adversarial=True, frequency_disentanglement=False)
    # Train the model (with adversarial noise applied)
    adversarial.train_model(verbose=True, reproduce=True)
    # Test this model
    adversarial.test_model(verbose=True, reproduce=True)
    # Get the observations
    adversarial_observations = adversarial.observations

    # Create the experiments object
    frequency_disentanglement = Experiments(config, adversarial=False, frequency_disentanglement=True)
    # Train the model (with frequency disentanglement applied)
    frequency_disentanglement.train_model(verbose=True, reproduce=True)
    # Test this model
    frequency_disentanglement.test_model(verbose=True, reproduce=True)
    # Get the observations
    frequency_disentanglement_observations = frequency_disentanglement.observations

    # Create the experiments object
    adversarial_frequency_disentanglement = Experiments(config, adversarial=True, frequency_disentanglement=True)
    # Train the model (with adversarial noise and frequency disentanglement applied)
    adversarial_frequency_disentanglement.train_model(verbose=True, reproduce=True)
    # Test this model
    adversarial_frequency_disentanglement.test_model(verbose=True, reproduce=True)
    # Get the observations
    adversarial_frequency_disentanglement_observations = adversarial_frequency_disentanglement.observations


    # Print the observations
    print(f"\n{'-'*50}\n")
    print(f"\nThe baseline observations are: \n{baseline_observations}\n")
    print(f"\nThe adversarial observations are: \n{adversarial_observations}\n")
    print(f"\nThe frequency disentanglement observations are: \n{frequency_disentanglement_observations}\n")
    print(f"\nThe adversarial frequency disentanglement observations are: \n{adversarial_frequency_disentanglement_observations}\n")
    print(f"\n{'-'*50}\n")

    
if __name__ == '__main__':
    main()