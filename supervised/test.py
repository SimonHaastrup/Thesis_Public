import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import *
from model import get_model
from data_split import generate_split

def evaluate(model, test_loader, loss_f, device):
    model.eval()
    predictions=[]
    targets=[]
    test_loss = 0
    with torch.no_grad():
        for x, target in test_loader:
            output = model(x.to(device, dtype=torch.float))
            output = torch.sigmoid(output) #Apply sigmoid to raw model output
            test_loss += loss_f(output, target.to(device, dtype=torch.float)).item()
            predictions.append(output.cpu())
            targets.append(target.cpu())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    test_loss /= len(predictions) #Division necessary since test_loss is sum of loss over all elements.
    auc_score = roc_auc_score(targets, predictions)
    print('\nTest set: Average loss: {:.6f}, AUROC: {:.4f}\n'.format(test_loss, auc_score))
    #print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))
    return test_loss, predictions, targets

def test(weights_path,
         BATCH_SIZE,
         out_dir,   # Path to output directory
         train_dir, # Path to training data
         train_labels_path, # Path to train labels csv
         train_WSIlabels_path, # Path to patch wsi labels csv
         val_wsi_keys_path = None, # Path to validation wsi keys generated during training
         small_test = False,
         NUM_WORKERS = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model = get_model(use_aux_loss=False, PRETRAINED=True, FREEZE=False, weight_path=weights_path).to(device)
    
    val_wsi_keys = load_object(val_wsi_keys_path)
    _, test_X, _, test_Y, _ = generate_split(train_labels_path, train_WSIlabels_path, keys_for_cv=val_wsi_keys)

    if small_test:
        x = 32
        test_X = test_X[-32:]
        test_Y = test_Y[-32:] 
    
    validation_aug = aug_val()
    test_loader = torch.utils.data.DataLoader(DataGenerator(test_X, test_Y, validation_aug, train_dir),
                                             num_workers = 2,
                                             batch_size=BATCH_SIZE)
    
    loss_f = get_loss()

    _, predictions, targets = evaluate(model, test_loader, loss_f, device)
    save_object(predictions, os.path.join(out_dir, "predictions"))
    save_object(targets, os.path.join(out_dir, "targets"))

def infer(weights_path, # Path to model weights
          BATCH_SIZE,
          out_dir,  # Path to output
          data_dir): #Path to dir with .h5 files containing tiled WSI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    invalid_numbers = [48]

    model = get_model(use_aux_loss=False, PRETRAINED=True, FREEZE=False, weight_path=weights_path).to(device)
    model.eval()
    
    h5_files =  [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    print("Performing inference for {:d} files".format(len(h5_files)))
    for h5_file in h5_files:
        if int(h5_file.split(".")[0].split("_")[1]) in invalid_numbers:
            continue
        print("Working on: " + h5_file)
        tumor_name = h5_file.split(".")[0]
  
        validation_aug = aug_val()
        data_path = os.path.join(data_dir, h5_file)
        test_loader = torch.utils.data.DataLoader(DataGenerator_H5(data_path, validation_aug),
                                                  batch_size=BATCH_SIZE)

        predictions=[]
        with torch.no_grad():
            for x in test_loader:
                output = model(x.to(device, dtype=torch.float))
                output = torch.sigmoid(output) #Apply sigmoid to raw model output
                predictions.append(output.cpu())
        predictions = np.vstack(predictions)
        save_object(predictions, os.path.join(out_dir, tumor_name) + "_predictions")

if __name__ == "__main__":
    # train_out_dir = "supervised_results/test"
    # test(weights_path = train_out_dir + "/checkpoint.best.pth.tar",
    #      BATCH_SIZE = 29,
    #      out_dir = train_out_dir,
    #      train_dir = "../train",
    #      train_labels_path = "../train_labels.csv",
    #      train_WSIlabels_path = "../patch_id_wsi_full.csv",
    #      val_wsi_keys_path = train_out_dir + "/val_wsi_keys",
    #      small_test = True)
    train_out_dir = "kagglePatchCAM/speciale/supervised_results/test"
    infer(weights_path = train_out_dir + "/checkpoint.best.pth.tar",
          BATCH_SIZE = 2,
          out_dir = "tmp_dir",
          data_dir = "tmp_dir")

    # infer(weights_path = train_out_dir + "/checkpoint.best.pth.tar",
    #       BATCH_SIZE = 512,
    #       out_dir = "../../CAM16/tiled_slides/patches",
    #       data_path = "../../CAM16/tiled_slides/patches/tumor_100.h5")

