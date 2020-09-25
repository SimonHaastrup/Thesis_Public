import os
import torch
import numpy as np

from utils import *
from model import get_model
from test import evaluate
from data_split import generate_split

def train_epoch(model, train_loader, optimizer, epoch, loss_f, epoch_size, device):
    model.train()
    epoch_losses = []
    for batch_idx, x in enumerate(train_loader):
        optimizer.zero_grad()
        device_x = x.to(device, dtype=torch.float)

        #Apply model
        x_rec, mu, log_var = model(device_x)

        #Calculate loss
        loss, rec_loss, kl_loss = loss_f(device_x, x_rec, mu, log_var)

        #Adjust weights via backprop
        loss.backward()
        optimizer.step()

        #Log loss
        print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}, Rec Loss: {:.6f}, KL Loss: {:.6f}'.format(
            epoch,
            (batch_idx + 1) * len(x),
            epoch_size, 100. * (batch_idx + 1) * len(x) / epoch_size,
            loss,
            rec_loss,
            kl_loss))
        epoch_losses.append((loss.item(), rec_loss.item(), kl_loss.item()))

    return epoch_losses

def train(BATCH_SIZE,
          LEARNING_RATE,
          N_EPOCHS,
          aug_prob,     # Probability of applying augmentation pipeline
          aug_type,     # Either ["spatial", "luminal", "all"]. Otherwise defaults to none
          REC_LOSS_TYPE,# Either ["L1", "L2", "BCE"]. Otherwise raises exception.
          out_dir,      # Path to output directory
          train_dir,    # Path to training data
          train_labels_path, # Path to train labels csv
          train_WSIlabels_path, # Path to patch wsi labels csv
          NUM_WORKERS = 2,
          small_test = False,
          RESIZE = True, # Should data be resized to [128, 128]
          DECAY_FACTOR = None, # Factor decaying learning rate on plateau.
          DECAY_PATIENCE = 5, # Decay plateau length
          RANDOM_VAL_SPLIT = False,
          GAMMA = 5e-5, # KL weight
          E_DIM = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Create directory for checkpointing and metrics
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if RANDOM_VAL_SPLIT:
        val_wsi_keys = None
    else:
        val_wsi_keys = [1, 12, 16, 21, 24, 25, 27, 32, 36, 50, 53, 58, 62, 67, 74, 76, 82, 89, 98, 105, 108, 110, 115, 119, 121, 138, 140, 141, 151, 156]
    train_X, val_X, _, _, val_wsi_keys = generate_split(train_labels_path, train_WSIlabels_path, keys_for_cv = val_wsi_keys)
    print("Validation WSIs are:", val_wsi_keys)

    if small_test:
        N = 32
        
        train_X = train_X[:2560]
        val_X = val_X[:640]

    train_aug = aug_train(aug_type=aug_type, p=aug_prob, RESIZE=RESIZE)
    val_aug = aug_val(RESIZE=RESIZE)

    train_loader = torch.utils.data.DataLoader(DataGenerator(train_X, train_aug, train_dir),
                                               pin_memory=False,
                                               num_workers=NUM_WORKERS,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(DataGenerator(val_X, val_aug, train_dir),
                                             num_workers=NUM_WORKERS,
                                             batch_size=BATCH_SIZE)

    loss_f = get_loss(rec_loss_type=REC_LOSS_TYPE, gamma=GAMMA)
    model = get_model(device, E_DIM)
    optimizer = get_optimizer(params=model.parameters(), LEARNING_RATE=LEARNING_RATE)
    
    scheduler = None
    if DECAY_FACTOR != None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=DECAY_PATIENCE, verbose=True, min_lr = 1e-5, factor=DECAY_FACTOR)
    #Calculate initial validation loss
    best_loss = evaluate(model, val_loader, loss_f, device)
    train_losses = []
    val_losses = [best_loss]
    
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, loss_f, len(train_X), device)
        val_loss = evaluate(model, val_loader, loss_f, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if scheduler != None:
            scheduler.step(val_loss[0])

        if val_loss[0]<best_loss[0]:
                print('Val loss improved from {:.6f} to {:.6f},'.format(best_loss[0], val_loss[0]))
                best_loss = val_loss
                save_model(out_dir, model, is_best=True)
    
    # Log results
    save_model(out_dir, model, is_best=False)
    save_object(val_wsi_keys, os.path.join(out_dir, "val_wsi_keys"))
    save_object(train_losses, os.path.join(out_dir, "train_losses"))
    save_object(val_losses, os.path.join(out_dir, "val_losses"))
    save_object(optimizer.param_groups[0]['lr'], os.path.join(out_dir, "final_lr"))
    
if __name__ == "__main__":
    train(BATCH_SIZE = 16,
          LEARNING_RATE = 10e-3,
          N_EPOCHS = 1,
          aug_prob = 1,
          aug_type = "none",
          out_dir = 'results',
          train_dir = '../../train',
          train_labels_path = '../train_labels.csv',
          train_WSIlabels_path = '../patch_id_wsi_full.csv',
          small_test = True,
          DECAY_FACTOR=0.1,
          DECAY_PATIENCE=2,
          REC_LOSS_TYPE="L2",
          GAMMA = 5e-5,
          E_DIM = 128)
