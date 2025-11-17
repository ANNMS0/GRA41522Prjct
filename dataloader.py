import numpy as np # dataset operations, also bw set uses npy
import pickle # data is in a .pkl format
import subprocess # helps automate wget and recommended
import tensorflow as tf # model operations
import os # verify if the dataset exists and dump it in data folder, using it so it works on any system

"""
DataLoader class used to download, preprocess and package the MNIST datasets
used in the VAE project. Handles both black-and-white and color versions of MNIST.

Everything is wrapped into a clean interface so the training script doesn’t need
to care about file formats, preprocessing details, or whether the dataset is already
on disk. You just pick the dataset and this class makes sure everything works.
"""

class DataLoader:
    """
    Handles downloading, preprocessing, and batching the dataset for the VAE.

    The class supports both 'bw' (black-and-white MNIST stored in .npy files)
    and 'color' (MNIST stored in .pkl dictionaries with multiple versions).
    
    Once initialized, the processed data is available as a TensorFlow
    `tf.data.Dataset` through `get_training_data()` and the raw test data
    and test labels are accessible for visualizations and posterior sampling.
    """
    def __init__(self, dset="bw", batch_size=128):
        """
        Initialize the DataLoader with a dataset type and batch size.

        Parameters
        ----------
        dset : str
        Either 'bw' / 'mnist_bw' for grayscale MNIST
        or 'color' / 'mnist_color' for the colored dataset.
        batch_size : int
        Batch size used when creating the TensorFlow dataset.
        """
        # batch_size: 128 as default, same as the one used in the provided git repository
        self.dset = dset 
        self.batch_size = batch_size
        self.data_dir = "data"
        # store the basic information

        self._set_paths_and_urls() # figure out which url and file path to use based on the data set
        self._ensure_data_dir() # check the data directory exists
        self._download_if_needed() # download files if not present
        self._load_data() # load the data into memory
        self._build_tf_dataset() # preprocess and build the dataset  
     
    def _set_paths_and_urls(self):
        """
        Decide which URLs and local file paths to use based on the dataset type.

        This keeps the rest of the logic clean and avoids scattering dataset-specific
        details around the code.
        """

        # Choose which urls and local file paths to use based on self.dset
        if self.dset in ("bw", "mnist_bw"):
            # black and white, mnist_bw
            self.train_url = "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0"
            self.test_url = "https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0"
            self.label_url = "https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0"
            # set the urls for each

            self.train_path = os.path.join(self.data_dir, "mnist_bw.npy")
            self.test_path = os.path.join(self.data_dir, "mnist_bw_te.npy")
            self.label_path = os.path.join(self.data_dir, "mnist_bw_y_te.npy")
            # setting the location for each dataset

        elif self.dset in ("color", "mnist_color"):
            # coloured, mnist_color
            self.train_url = "https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0"
            self.test_url = "https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0"
            self.label_url = "https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0"
            # set the urls for each

            self.train_path = os.path.join(self.data_dir, "mnist_color.pkl")
            self.test_path = os.path.join(self.data_dir, "mnist_color_te.pkl")
            self.label_path = os.path.join(self.data_dir, "mnist_color_y_te.npy")
            # setting the location for each dataset
        else:
            raise ValueError(f"Unknown dataset type: {self.dset}")
    
    def _ensure_data_dir(self):
        """
        Create the `data/` directory if it doesn't already exist.

        Keeps all downloaded datasets in one place and makes sure the code works
        on any system without manual folder creation.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _download_if_needed(self):
        """
        Download the dataset files with `wget` if they are not already present.

        The project explicitly recommends using wget from Python, so this method
        simply loops over the required URLs and fetches whatever is missing.
        """
        for url, path in [ 
            (self.train_url, self.train_path),
            (self.test_url, self.test_path),
            (self.label_url, self.label_path),
        ]:
            if not os.path.exists(path):
                subprocess.run(["wget", url, "-O", path], check=True)

    def _load_data(self):
        """
        Load the raw dataset into memory using the appropriate method.

        - BW data is stored as .npy, so `np.load` is enough.
        - Color data is stored as .pkl dictionaries, so we use `pickle.load`
        and pick one version (m4 in this case).

        The method only loads raw data — all preprocessing happens later.
        """
        if self.dset == "bw":
            # the bw data is in a npy format so np.load is enough
            self.x_train = np.load(self.train_path)
            self.x_test = np.load(self.test_path)
            self.y_test = np.load(self.label_path)
        elif self.dset == "color":
            # the images use .pkl so have to use pickle
            with open(self.train_path, "rb") as f:
                color_train = pickle.load(f)
            with open(self.test_path, "rb") as f:
                color_test = pickle.load(f)
            # this dataset is a dictionary of 5 different versions of MNIST, 
            # choosing m4 as 4 is my favourite number
            self.x_train = color_train["m4"]
            self.x_test = color_test["m4"]
            self.y_test = np.load(self.label_path)


    def _build_tf_dataset(self):
        """
        Preprocess the data and convert it into a `tf.data.Dataset` pipeline.

        Steps:
        - Convert to float32
        - Ensure values are in [0, 1]
        - Vectorize the BW dataset (flatten to 784 features)
        - Store both training and test sets
        - Build the TensorFlow dataset using from_tensor_slices(), shuffle(), batch()

        This is the dataset that is fed directly into the VAE during training.
        """
        
        # make sure it is float32
        x_train = self.x_train.astype("float32")
        x_test = self.x_test.astype("float32")

        # make sure data is in [0,1], it can be [0,255], got confused by utils.py, point 2.3 of the paper
        if x_train.max() > 1.0:
            x_train /= 255.0
            x_test /= 255.0

        # point 2.3 of the paper, vectorize "bw" data for the model so it has a 784 as the 2nd dimension
        if self.dset == "bw":
            # dataset with no channel is 3D and with a channel is 4D, writing it like this to make sure only those dimensions are used
            if x_train.ndim in (3, 4):
                # using reshape with -1 instead of mandating 784 so it works for any dataset size
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_test = x_test.reshape(x_test.shape[0], -1)
            else:
                raise ValueError(f"Unexpected input shape: {x_train.shape}")


        self.x_train = x_train
        self.x_test = x_test   

        # build the pipeline for the tensor flow dataset
        ds = tf.data.Dataset.from_tensor_slices(x_train)
        ds = ds.shuffle(buffer_size=len(x_train)).batch(self.batch_size)
        self.train_ds = ds

    def get_training_data(self):
        """
        Return the TensorFlow training dataset.

        This method exists so external code doesn't need to access internal attributes
        like `self.train_ds` and keeps the interface stable even if I change the
        internal implementation later.
        """
        return self.train_ds

    # for the tsne scatter plots and test images for reconstruction and posterior sampling
    def get_test_data(self):
        """
        Return the raw test images.

        Used for posterior sampling (q(z|x)) and reconstruction visualizations.
        """
        return self.x_test
    
    def get_test_labels(self):
        """
        Return the test labels.

        Only needed for visualizing the latent space with t-SNE (color-coded scatter plot).
        """
        return self.y_test 