import os
import pickle
import numpy as np
import torch
from torch.utils import data
from skimage.io import imread
from albumentations import *
import h5py

def get_loss():
    return torch.nn.BCELoss(reduction="sum")

def get_optimizer(params, optim_type, LEARNING_RATE):
    if optim_type == "adam":
        optimizer = torch.optim.Adam(params=params, lr=LEARNING_RATE)
    elif optim_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params=params, lr=LEARNING_RATE)
    else:
        raise Exception("Invalid optimizer: " + optim_type)

    return optimizer

def get_aug_pipeline(aug_type, p=1):
    FLIP_PROB = 0.5
    if aug_type == "spatial": 
        return Compose([RandomRotate90(p=1),
                        HorizontalFlip(FLIP_PROB),
                        RandomRotate90(p=1)],
                        p=p)
    elif aug_type == "luminal":
        return Compose([HueSaturationValue(hue_shift_limit=0.04*255,
                                           sat_shift_limit=0.25*255,
                                           val_shift_limit=64.0/255.0*255,
                                           p=1),
                        RandomContrast(limit= 0.2, p=1)],
                        p=p)
    elif aug_type == "all":
        return Compose([RandomRotate90(p=1),
                        HorizontalFlip(p=FLIP_PROB),
                        RandomRotate90(p=1),
                        HueSaturationValue(hue_shift_limit=0.04*255,
                                           sat_shift_limit=0.25*255,
                                           val_shift_limit=64.0/255.0*255,
                                           p=1),
                        RandomContrast(limit= 0.2, p=1)],
                        p=p)
    else:
        print("Running with no augmentations")
        return None

#Augmentations following google brain paper
def aug_train(aug_type, p=1, RESIZE=False):
    if RESIZE:
        return Compose([Resize(299, 299),
                        get_aug_pipeline(aug_type, p)],
                        p=p)
    else:
        return get_aug_pipeline(aug_type, p)

def aug_val(RESIZE=False):
    if RESIZE:
        return Resize(299, 299)
    return None

class DataGenerator(data.Dataset):
    def __init__(self, ids, labels, augment, img_dir):
        self.ids = ids
        self.labels = labels
        self.augment = augment
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.ids) 

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        y = self.labels[idx]
        X = self.load_image(img_id)
        return X, np.expand_dims(y,0)

    def load_image(self, img_id):
        img_id = img_id+'.tif'
        img = imread(os.path.join(self.img_dir, img_id))
        if self.augment!=None:
            augmented = self.augment(image=img)
            img = augmented['image']
        img = img/255.0
        img = np.moveaxis(img, 2, 0) # Reshape image to 3 x W x H
        return img

# Generator for inference on tiled patches gathered in a .h5 file.
class DataGenerator_H5(data.Dataset):
    def __init__(self, h5_path, augment, TTA=False):
        super(DataGenerator_H5, self).__init__()

        self.augment = augment
        self.file = h5py.File(h5_path, 'r')
        #self.n_images, _, _, _ = self.file['x'].shape
        self.n_images, _, _, _ = self.file['imgs'].shape

    def __getitem__(self, index):
        #img = self.file['x'][index,:,:,:]
        img = self.file['imgs'][index,:,:,:]
        if self.augment!=None:
            augmented = self.augment(image=img)
            img = augmented['image']
        img = img/255.0
        img = np.moveaxis(img, 2, 0) # Reshape image to 3 X W x H
        return img

    def __len__(self):
        return self.n_images

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

