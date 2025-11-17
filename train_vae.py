import argparse
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataloader import DataLoader
from vae import VAE
from utils import optimizer, plot_grid

"""
Training script for the VAE.

This module is the entry point for the project. It wires together the
DataLoader, the VAE model, and the different evaluation utilities
(latent space visualization and image generation from prior/posterior).

Everything is controlled via command-line flags so the experiments can
be reproduced without touching the code.
"""

def visualize_latent(vae, loader, args):
    """
    Encode test data into the latent space and run t-SNE for visualization.

    Uses a subset of the test set (up to 2000 points) to keep t-SNE
    reasonably fast (32 gb ram computers aren't cheap), then plots a 2D scatter plot colored by the class labels
    and saves it as an image file.

    Parameters
    ----------
    vae : VAE
        Trained VAE model.
    loader : DataLoader
        DataLoader instance providing test data and labels.
    args : argparse.Namespace
        Parsed command-line arguments, mainly used for the dataset name.
    """
    # encode the data, run t-sne on the latent means and save a scatter plot
    x_test = loader.get_test_data()
    y_test = loader.get_test_labels()

    # use a subset so TSNE doesn't take forever
    max_points = 2000
    x_test = x_test[:max_points]
    y_test = y_test[:max_points]

    # 1. Encode to get mu and log sigma^2
    z_mean, z_logvar = vae.encode(x_test)

    # 2. Sample z using the reparameterization trick
    z = vae.reparameterize(z_mean, z_logvar)

    #  3. Move it to numpy to use it
    z = z.numpy()

    # 4. Run t-sne on z
    tsne = TSNE(n_components=2, init="random", learning_rate="auto")
    z_2d = tsne.fit_transform(z)

    # make a scatter plot that is colored by labels
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:,1], c=y_test, s=5, cmap="tab10")
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f"Latent space t-SNE({args.dset})")
    plt.tight_layout()
    plt.savefig(f"latent_tsne_{args.dset}.png")
    plt.close()

    # print a nice message
    print(f"[INFO] Latent t-SNE figure saved as 'latent_tsne_{args.dset}.png'")

def generate_from_prior(vae, args):
    """
    Generate samples from the prior p(z) and decode them into images.

    Draws 100 latent points from a standard normal prior, decodes them
    with the trained VAE, and saves a 10x10 grid of generated images.

    Parameters
    ----------
    vae : VAE
        Trained VAE model.
    args : argparse.Namespace
        Parsed command-line arguments, mainly used for the dataset name.
    """
    # 100 samples as plot_grid has a 10x10 format
    n = 100
    
    # 1. Sample z ~ N(0, I)
    z = tf.random.normal(shape=(n, vae.latent_dim))

    # 2. Decode into images
    x_hat = vae.decode(z)

    # 3. Reshape into a proper format
    if args.dset == "bw":
        x_hat = tf.reshape(x_hat, (n, 28, 28))
    else:
        x_hat = tf.reshape(x_hat, (n, 28, 28, 3))

    # 4. Scale and convert into uint8 images
    x_hat = tf.clip_by_value(255.0 * x_hat, 0.0, 255.0)
    x_hat = tf.cast(x_hat, tf.uint8).numpy()

    # 5. Save the grid using plot_grid
    plot_grid(x_hat, name=f"_{args.dset}_prior")

    # print a nice message
    print(f"[INFO] Prior samples grid saved as 'xhat_{args.dset}_prior.png'")

def generate_from_posterior(vae, loader, args):
    """
    Generate samples from the posterior q(z|x) using test images.

    Takes the first 100 test images, encodes them into the latent space,
    samples z using the same reparameterization trick as during training,
    decodes those samples, and visualizes a 10x10 grid of reconstructions.

    Parameters
    ----------
    vae : VAE
        Trained VAE model.
    loader : DataLoader
        DataLoader instance providing test data.
    args : argparse.Namespace
        Parsed command-line arguments, mainly used for the dataset name.
    """

    # 1. Take 100 test
    x_test = loader.get_test_data()
    x_test = x_test[:100]

    # 2. Encode to the posterior parameters
    z_mean, z_logvar = vae.encode(x_test)

    # 3. Sample z ~q(z|x) using the same reparmaterization as in training
    z = vae.reparameterize(z_mean, z_logvar)

    # 4. Decode 
    x_hat = vae.decode(z)

    # 5. Reshape into a proper format
    n = 100
    if args.dset == "bw":
        x_hat = tf.reshape(x_hat, (n, 28, 28))
    else:
        x_hat = tf.reshape(x_hat, (n, 28, 28, 3))

    # 6. Scale and convert into uint8 images
    x_hat = tf.clip_by_value(255.0 * x_hat, 0.0, 255.0)
    x_hat = tf.cast(x_hat, tf.uint8).numpy()

    # 7. Save the grid using plot_grid
    plot_grid(x_hat, name=f"_{args.dset}_posterior") 

    # print a nice message
    print(f"[INFO] Posterior samples grid saved as 'xhat_{args.dset}_posterior.png'")   


def main(args):
    """
    Main training and evaluation pipeline.

    - Normalizes the dataset name (bw/color)
    - Instantiates the DataLoader and VAE
    - Runs the training loop for the requested number of epochs
    - Optionally:
      * visualizes the latent space with t-SNE
      * generates samples from the prior
      * generates samples from the posterior

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments controlling dataset, batch size,
        number of epochs, and which post-training actions to run.
    """
    # map the input 
    if args.dset in ("bw", "mnist_bw"):
        args.dset = "bw"
    elif args.dset in ("color", "mnist_color"):
        args.dset = "color"
    else:
        raise ValueError(f"Unknown dataset: {args.dset}")
   
    # 1. Data
    loader = DataLoader(dset=args.dset, batch_size=args.batch_size)
    train_ds = loader.get_training_data()

    # 2. The model
    vae = VAE(dset=args.dset)

    # 3. The training loop
    for epoch in range(args.epochs):
        epoch_losses = []

        for batch in train_ds:
            # mini-batch of images
            loss, kl, recon = vae.train(batch, optimizer)
            epoch_losses.append(loss.numpy())

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # given by VAE.call in the last batch
        avg_elbo = -avg_loss


        print(
            f"Epoch {epoch+1}/{args.epochs} |"
            f"loss = {avg_loss:.4f} |"
            f"ELBO = {avg_elbo:.4f} |"
            f"KL = {kl.numpy():.4f} |"
            f"log p(x|z) = {recon.numpy():.4f}" 
        )
    # optionally visualize latent space
    if args.visualize_latent:
        visualize_latent(vae, loader, args)

    if args.generate_from_prior:
        generate_from_prior(vae, args)

    if args.generate_from_posterior:
        generate_from_posterior(vae, loader, args)

if __name__ == "__main__":
    """
    The buttons on this vending machine of a program essentially. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset", type=str, default="bw", choices=["bw", "mnist_bw", "color", "mnist_color"],
        help="Which dataset to use: 'bw' or 'color'?"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="What mini-batch size?"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="How many training epochs?"
    )
    # f.1
    parser.add_argument(
        "--visualize_latent",
        action="store_true",
        help="Visualize the latent space with t-SNE after training is completed?"
    )
    # f.2
    parser.add_argument(
        "--generate_from_prior",
        action="store_true",
        help="Generate samples from the prior p(z) after training is completed?"
    )
    # f.3
    parser.add_argument(
        "--generate_from_posterior",
        action="store_true",
        help="Generate samples from the posterior q(z|x) after training is completed?"
    )
    # f.4

    args = parser.parse_args()
    main(args)