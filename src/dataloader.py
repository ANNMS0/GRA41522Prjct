import numpy as np # dataset operations, also bw set uses npy
import pickle # data is in a .pkl format
import subprocess # helps automate wget and recommended
import tensorflow as tf # model operations
import os # verify if the dataset exists and dump it in data folder, using it so it works on any system


class DataLoader:
    def __init__(self, dset="bw", batch_size=128):
        # dset: "bw" for grayscale data, "color" for colored data
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
        # Choose which urls and local file paths to use based on self.dset
        if self.dset == "bw":
            # black and white, mnist_bw
            self.train_url = "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0"
            self.test_url = "https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0"
            self.label_url = "https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0"
            # set the urls for each

            self.train_path = os.path.join(self.data_dir, "mnist_bw.npy")
            self.test_path = os.path.join(self.data_dir, "mnist_bw_te.npy")
            self.label_path = os.path.join(self.data_dir, "mnist_bw_y_te.npy")
            # setting the location for each dataset

        elif self.deset == "color":
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
        # check if the data directory exists, if not, make it
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
