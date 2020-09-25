import os
import h5py
import numpy as np
import pickle

import torch
from torch.utils import data

class DataGenerator_Bag(data.Dataset):
    def __init__(self, bag_paths, labels, enc_size):
        self.bag_paths = bag_paths
        self.labels = labels
        self.enc_size = enc_size
        
    def __len__(self):
        return len(self.bag_paths) 

    def __getitem__(self, idx):
        bag_path = self.bag_paths[idx]
        y = self.labels[idx]
        X = self.load_bag(bag_path)
        return X, np.expand_dims(y,0)

    def load_bag(self, bag_path):
        bag_file = h5py.File(bag_path, 'r')
        if self.enc_size == 3:
            bag = bag_file['x_rgb']
        else:
            bag = bag_file['x_vae']
        return np.array(bag[:])

def load_model(model, weight_path):
    if not os.path.isfile(weight_path):
        raise Exception("Attempted to load model weights with invalid filename: " + weight_path)
    
    model.load_state_dict(torch.load(weight_path))
    return model

def save_model(checkpoints_dir, model, is_best=False):
    weights_path = os.path.join(checkpoints_dir, "checkpoint.last.pth.tar")
    if is_best:
        weights_path = os.path.join(checkpoints_dir, "checkpoint.best.pth.tar")   

    torch.save(model.state_dict(), weights_path)

def save_object(object, path):
    with open(path, "wb") as fh:
        pickle.dump(object, fh)

def load_object(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)