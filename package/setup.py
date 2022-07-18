import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="expvanillagan",                     # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Dhruv Srikanth",                     # Full name of the author
    description="Package for running experiments on vanilla GANs",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.9',                # Minimum version requirement of the package
    py_modules=["expvanillagan"],             # Name of the python package
    package_dir={'':'expvanillagan/src'},     # Directory of the source code of the package
    install_requires=[
        'numpy', 
        'tqdm', 
        'matplotlib', 
        'Pillow', 
        'torch', 
        'torchvision', 
        'tensorboard', 
        'torchaudio', 
        'typing'
        ]                     # Install other dependencies if any
)