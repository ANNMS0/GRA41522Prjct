import tensorflow as tf 
from tensorflow.keras import Model, layers
from abc import ABC, abstractmethod

from neural_networks import encoder_mlp, encoder_conv, decoder_mlp, decoder_conv
from losses import kl_divergence, log_diag_mvn

"""
VAE module.

Contains the BiCoder base class and the concrete Encoder, Decoder and VAE
classes used in the project. The basic idea is to keep the neural network components
and the VAE logic in one place, while the training script and data loading live
in their own modules.
"""
class BiCoder(layers.Layer, ABC):
    """
    Base class for encoder and decoder components.

    This abstracts away whatever architecture is used under the hood
    (MLP vs CNN, BW vs color). Subclasses are responsible for actually
    building the network and defining how the forward pass works.
    """
    def __init__(self, dset="bw"):
        """
        Parameters
        ----------
        dset : str
            Dataset type, either 'bw' or 'color'. This decides which
            underlying network architecture to use.
        """
        super().__init__()
        self.dset = dset

    @abstractmethod
    def build_network(self):
        """
        Create and assign the underlying neural network.

        Subclasses must set `self.network` to the appropriate function or
        Keras model, depending on the dataset and architecture.
        """
        pass

    @abstractmethod
    def call(self, x):
        """
        Define the forward pass for the component.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to the encoder or decoder.

        Returns
        -------
        Any
            The exact output depends on the subclass (e.g. mean and log-variance
            for the encoder, reconstruction for the decoder).
        """
        pass

class Encoder(BiCoder):
    """
    Encoder netwerk.

    Takes input data x and outputs the parameters of the approximate
    posterior q(z|x), i.e. mean and log-variance of a diagonal Gaussian.
    """
    def __init__(self, dset="bw"):
        """
        Parameters
        ----------
        dset : str
            Dataset type, 'bw' or 'color'. Chooses between MLP and CNN.
        """
        super().__init__(dset)
        self.build_network()

    def build_network(self):
        """
        Pick the appropriate encoder architecture based on the dataset.

        - BW  → MLP encoder
        - Color → CNN encoder
        """
        if self.dset == "bw":
            self.network = encoder_mlp
        elif self.dset == "color":
            self.network = encoder_conv
        else:
            raise ValueError(f"Unsupported dataset: {self.dset}")

    def call(self, x):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : tf.Tensor
            Mini-batch of input data.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            z_mean and z_logvar representing q(z|x).
        """
        out = self.network(x)
        latent_dim = out.shape[-1] // 2
        z_mean = out[:, :latent_dim]
        z_logvar = out[:, latent_dim:]
        return z_mean, z_logvar

class Decoder(BiCoder):
    """
    Decoder network.

    Takes latent vectors z and outputs the reconstruction parameters for p(x|z).
    In this project the decoder outputs the mean of a Gaussian likelihood.
    """
    def __init__(self, dset="bw"):
        """
        Parameters
        ----------
        dset : str
            Dataset type, 'bw' or 'color'. Chooses between MLP and CNN.
        """
        super().__init__(dset)
        self.build_network()

    def build_network(self):
        """
        Pick the appropriate decoder architecture based on the dataset.

        - BW  → MLP decoder
        - Color → CNN decoder
        """
        if self.dset == "bw":
            self.network = decoder_mlp
        elif self.dset == "color":
            self.network = decoder_conv
        else:
            raise ValueError(f"Unsupported dataset: {self.dset}")
    def call(self, x):
        """
        Forward pass of the decoder.

        Parameters
        ----------
        x : tf.Tensor
            Latent vectors z.

        Returns
        -------
        tf.Tensor
            Reconstructed mean x_mu.
        """
        return self.network(x)

class VAE(Model):
    """
    Variational Autoencoder model.

    Wraps together the encoder and decoder, implements the reparameterization
    trick, and computes the ELBO (reconstruction term minus KL divergence).
    """
    def __init__(self, dset="bw"):
        """
        Parameters
        ----------
        dset : str
            Dataset type, 'bw' or 'color', which controls the underlying
            architectures and the latent dimensionality.
        """
        super().__init__()
        self.dset = dset

        # needed components
        self.encoder = Encoder(dset)
        self.decoder = Decoder(dset)

        # latent dim depends on dataset, given by neural_networks.property
        if dset == "bw":
            self.latent_dim = 20
        elif dset == "color":
            self.latent_dim = 50
        else:
            raise ValueError(f"Unsupported dataset: {dset}")
        
        # decoder standard deviation = 0.75 as mandated in neural_networks.py 
        self.sigma_x = 0.75
        self.log_sigma_x = tf.math.log(tf.constant(self.sigma_x, dtype=tf.float32))

        # used later to complet vae_loss that is needed by utils.py 
        self.kl_loss = None
        self.recon_logprob = None 
        self.elbo = None 
        self.vae_loss = None 

    def reparameterize(self, z_mean, z_logvar):
        # z = mu + sigma * eps (reparametirization trick from the pdf)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_logvar) * eps 
        return z 

    def call(self, x):
        """
        Forward pass of the VAE, computing the loss components.

        Steps
        -----
        1. Encode x → (z_mean, z_logvar)
        2. Sample z via reparameterization
        3. Decode z → x_mu
        4. Compute log p(x|z)
        5. Compute KL(q(z|x) || p(z))
        6. Compute ELBO and the final loss

        Parameters
        ----------
        x : tf.Tensor
            Mini-batch of input data.

        Returns
        -------
        tf.Tensor
            Scalar VAE loss (negative ELBO) averaged over the batch.
        """

        # 1. Encode
        z_mean, z_logvar = self.encoder(x)
        
        # 2. Sample latent z
        z = self.reparameterize(z_mean, z_logvar)
        
        # 3. Decode
        x_mu = self.decoder(z)
        
        # 4. Reconstruct 
        log_px_z = log_diag_mvn(x, x_mu, self.log_sigma_x)
        
        # 5. KL term
        kl = kl_divergence(z_mean, z_logvar)
        
        # 6. ELBO and the losses (averages per batch)
        self.recon_logprob = tf.reduce_mean(log_px_z)
        self.kl_loss = tf.reduce_mean(kl) 
        self.elbo = self.recon_logprob - self.kl_loss
        
        # 7. What will be minimized
        self.vae_loss = -self.elbo 
        
        return self.vae_loss
    
    @tf.function
    def train(self, x, optimizer):
        """
        Single training step for a mini-batch.

        Wraps the forward pass in a GradientTape, computes gradients of the
        VAE loss with respect to all trainable variables, and applies one
        optimizer update.

        Parameters
        ----------
        x : tf.Tensor
            Mini-batch of input data.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer used to update the model parameters.

        Returns
        -------
        (tf.Tensor, tf.Tensor, tf.Tensor)
            Tuple of (loss, kl_loss, recon_logprob) for logging.
        """
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, self.kl_loss, self.recon_logprob

    def encode(self, x):
        """
        Encode data into mean and log-variance of the latent distribution.

        Parameters
        ----------
        x : tf.Tensor
            Input data.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            z_mean and z_logvar.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent vectors into reconstructed images.

        Parameters
        ----------
        z : tf.Tensor
            Latent vectors.

        Returns
        -------
        tf.Tensor
            Decoder output x_mu.
        """
        return self.decoder(z)

    def sample_prior(self, n_samples):
        """
        Sample from the prior p(z) = N(0, I) and decode the samples.

        Parameters
        ----------
        n_samples : int
            Number of latent points to draw from the prior.

        Returns
        -------
        tf.Tensor
            Decoder output for the sampled latent vectors.
        """
        latent_dim = self.latent_dim # store it in __init__
        z = tf.random.normal(shape=(n_samples, latent_dim))
        return self.decode(z)


