config = {
    # General parameters
    'initial seed': 1, 
    'device': 'mps', 
    # Data parameters
    'batch size': 8, 
    'num workers': 4, 
    'image shape': (1, 28, 28),
    # Model parameters
    'generator blocks' : 3,
    'discriminator blocks' : 3,
    # Training parameters
    'latent dimension': 64,
    'learning rate': 0.01, 
    'beta1': 0.5,
    'beta2': 0.999,
    'epochs': 2, 
    'discriminator epochs': 1,
    # Save parameters
    'sample interval': 1,
    'sample save path': './samples',
    'model save path': './models',
    # Log parameters
    'log path': './logs',
}