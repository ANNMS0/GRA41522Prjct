# Final Project - VAE (GRA4152 OOP)

This repository contains code for the final project in the course
**GRA4152 – Object-Oriented Programming with Python (Autumn 2025)**.

Follow the instructions below to set up the environment and run the project.

## Environment Setup

Create the conda environment and install the required packages:
conda create --name tf python==3.11 tensorflow==2.15.0 -c conda-forge
conda activate tf
pip install scikit-learn
pip install matplotlib

If your system does not have `wget`, install it in the environment:
conda install -c menpo wget

## Running the Project

Train the Variational Autoencoder (VAE) using:

python train_vae.py

Default values:  
- Dataset: `bw`  
- Batch size: `128`  
- Epochs: `10`  

Dataset options:  
- `bw` or `mnist_bw` for black-and-white MNIST  
- `color` or `mnist_color` for color MNIST  

Additional options:
--visualize_latent # visualize latent space (t-SNE)
--generate_from_prior # generate images from the prior p(z)
--generate_from_posterior # generate images using q(z|x)


All datasets are automatically downloaded by the `DataLoader` class using `wget` and saved in the `data` subfolder for convenience.  
Generated images and plots are saved in the working directory.

## Provided Files

The files `losses.py`, `neural_networks.py` and `utils.py` were provided by the course and are used directly in the implementation.
The files `dataloader.py`, `vae.py` and `train_vae.py` were made by me in accordance with the project requirements

