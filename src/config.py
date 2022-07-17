config = {
    # General parameters
    'initial seed': 1, 
    'device': 'mps', 
    # Data parameters
    'batch size': 32, 
    'num workers': 8, 
    'image shape': (1, 28, 28),
    'train samples': 60000, 
    'test samples': 4000,
    # Model parameters
    'generator blocks' : 2,
    'discriminator blocks' : 2,
    # Training parameters
    'latent dimension': 64,
    'learning rate': 0.01, 
    'beta1': 0.5,
    'beta2': 0.999,
    'epochs': 100, 
    'discriminator epochs': 1,
    # Save parameters
    'sample interval': 1,
    'sample save path': './samples',
    'model save path': './weights',
    # Log parameters
    'log path': './logs',
    'experiment number': 1
}