config = {
    # General parameters
    'initial seed': 1, 
    'device': 'mps', 
    # Data parameters
    'batch size': 8, 
    'num workers': 4, 
    'image shape': (1, 28, 28),
    # Training parameters
    'learning rate': 0.01, 
    'beta1': 0.5,
    'beta2': 0.999,
    'epochs': 2, 
    # Save parameters
    'sample interval': 1,
    'sample save path': './samples',
    'model save path': './models',
}