import os
import torch
import numpy as np
import h5py

from model import get_model
from utils import load_model, save_object, load_object
from visualize import find_corner_threshold
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate(model, test_loader, loss_f, device):
    model.eval()
    predictions=[]
    targets=[]
    test_loss = 0
    with torch.no_grad():
        for x, target in test_loader:
            output, _, _ = model(x[0,:,:].to(device, dtype=torch.float)) #Indexing into x for BATCH_SIZE=1
            test_loss += loss_f(output, target.to(device, dtype=torch.float)).item()
            predictions.append(output.cpu())
            targets.append(target.cpu())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    test_loss /= len(predictions) #Division necessary since test_loss is sum of loss over all elements.
    auc_score = roc_auc_score(targets, predictions)
    print('\nTest set: Average loss: {:.6f}, AUROC: {:.4f}\n'.format(test_loss, auc_score))
    return test_loss, predictions, targets

def load_bag(bag_path, enc_size):
        bag_file = h5py.File(bag_path, 'r')
        if enc_size == 3:
            bag = bag_file['x_rgb']
        else:
            bag = bag_file['x_vae']
        return np.array(bag[:])

def infer(weights_path, # Path to model weights
          out_dir, # Path to output
          data_dir, # Path to dir with .h5 files containing encoded WSI
          E_DIM=256,
          ATTN_NEURONS=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model = get_model(device=device, attn_neurons=ATTN_NEURONS, enc_size=E_DIM)
    model = load_model(model, weights_path)
    model.eval()
    
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    h5_files.sort()
    val_targets = load_object("data/test/test_labels")

    predictions = [] #name, prediction

    print("Performing inference for {:d} files".format(len(h5_files)))
    for h5_file in h5_files:
        print("Working on: " + h5_file)
        tumor_name = h5_file.split(".")[0]

        #load x
        bag_path = os.path.join(data_dir, h5_file)
        x = torch.from_numpy(load_bag(bag_path, E_DIM))

        with torch.no_grad():
            prediction, _, _ = model(x.to(device, dtype=torch.float))
            predictions.append([tumor_name, prediction.item()])


    ### AUC addition ###
    val_predictions = [x for [_, x] in predictions]
    val_targets = load_object("data/test/test_labels")
    thresh = find_corner_threshold(val_predictions, val_targets)
    #print([1 if x > thresh else 0 for x in val_predictions])
    ### AUC addition ###

    save_object(predictions, os.path.join(out_dir, "predictions"))

def infer_val(weights_path, # Path to model weights
              out_dir, # Path to output
              val_wsi_path,
              E_DIM=256,
              ATTN_NEURONS=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model = get_model(device=device, attn_neurons=ATTN_NEURONS, enc_size=E_DIM)
    model = load_model(model, weights_path)
    model.eval()

    val_h5_paths = load_object(val_wsi_path)
    
    val_h5_paths.sort()
    val_targets = [0 if x.split("\\")[-1].split("_")[0]=="normal" else 1 for x in val_h5_paths]

    predictions = [] #name, prediction

    print("Performing inference for {:d} files".format(len(val_h5_paths)))
    for h5_path in val_h5_paths:
        h5_file = h5_path.split("\\")[-1]
        print("Working on: " + h5_file)
        tumor_name = h5_file.split(".")[0]

        #load x
        x = torch.from_numpy(load_bag(h5_path, E_DIM))

        with torch.no_grad():
            prediction, _, _ = model(x.to(device, dtype=torch.float))
            predictions.append([tumor_name, prediction.item()])
        
    save_object(predictions, os.path.join(out_dir, "val_predictions"))

if __name__ == "__main__":
    for e in ["1", "2", "3", "4", "5"]:
        infer(weights_path="results/train/best_vae_experiments/" + e + "/checkpoint.best.pth.tar",
            out_dir="results/test/vae/" + e,
            data_dir="data/test/encoded_images", # Path to dir with .h5 files containing encoded WSI
            E_DIM=256,
            ATTN_NEURONS=128)