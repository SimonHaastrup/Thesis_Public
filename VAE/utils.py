import os
import pickle
import numpy as np
import torch
from torch.utils import data
from skimage.io import imread
from albumentations import *
import h5py

def get_optimizer(params, LEARNING_RATE):
    return torch.optim.Adam(params=params, lr=LEARNING_RATE)

def get_loss(rec_loss_type, gamma):
    def loss_f(x, x_rec, mu, var):
        return VAE_loss(x, x_rec, mu, var, rec_loss_type, gamma)
    return loss_f

#x and x_rec are batches
def VAE_loss(x, x_rec, mu, var, rec_loss_type, gamma):
    if rec_loss_type == "L1":
        rec_loss = torch.nn.L1Loss()(x_rec, x)
    elif rec_loss_type == "L2":
        rec_loss = torch.nn.MSELoss()(x_rec, x)
    elif rec_loss_type == "BCE":
        rec_loss = torch.nn.BCELoss()(x_rec, x)
    else:
        raise Exception("Invalid reconstruction loss type")
    # see Appendix B from VAE paper:
    kl_loss = -1 * gamma * torch.mean(torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=1))

    loss = rec_loss + kl_loss
    return loss, rec_loss, kl_loss

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
        return Compose([Resize(128, 128),
                        get_aug_pipeline(aug_type, p)],
                        p=p)
    else:
        return get_aug_pipeline(aug_type, p)

def aug_val(RESIZE=False):
    if RESIZE:
        return Resize(128, 128)
    return None

class DataGenerator(data.Dataset):
    def __init__(self, ids, augment, img_dir):
        self.ids = ids
        self.augment = augment
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.ids) 

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        X = self.load_image(img_id)
        return X

    def load_image(self, img_id):
        img_id = img_id+'.tif'
        img = imread(os.path.join(self.img_dir, img_id))
        if self.augment!=None:
            augmented = self.augment(image=img)
            img = augmented['image']
        img = img/255.0
        img = np.moveaxis(img, 2, 0) # Reshape image to 3 x W x H
        return img

# Generator for loading a tiled h5 image for encoding purposes.
class DataGenerator_H5(data.Dataset):
    def __init__(self, h5_path, augment):
        super(DataGenerator_H5, self).__init__()
        self.file = h5py.File(h5_path, 'r')
        self.n_images, _, _, _ = self.file['imgs'].shape
        self.augment = augment

    def __getitem__(self, index):
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

