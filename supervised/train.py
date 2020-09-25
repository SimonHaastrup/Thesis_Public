import os
import torch
import numpy as np

from utils import *
from model import get_model
from data_split import generate_split
from test import evaluate

def train_epoch(model, train_loader, optimizer, epoch, loss_f, epoch_size, device):
    model.train()
    epoch_losses = []
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        #Apply model
        output = model(x.to(device, dtype=torch.float))
        output = torch.sigmoid(output) #Apply sigmoid to raw model output
        
        #Calculate loss
        loss = loss_f(output, target.to(device, dtype=torch.float)) / len(target) # Division to counteract loss accum

        #Adjust weights via backprop
        loss.backward()
        optimizer.step()

        #Log loss
        print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(x), epoch_size,
            100. * (batch_idx + 1) * len(x) / epoch_size, loss))
        epoch_losses.append(loss.item())

    train_loss_mean = np.mean(epoch_losses)
    print('Mean train loss on epoch {} : {:.6f}'.format(epoch, train_loss_mean))
    return epoch_losses

def train(BATCH_SIZE,
          LEARNING_RATE,
          N_EPOCHS,
          aug_prob,     # Probability of applying augmentation pipeline
          aug_type,     # Either ["spatial", "luminal", "all"]. Otherwise defaults to none
          optim_type,   # Either ["adam", "rmsprop"]. Otherwise throws exception
          out_dir,      # Path to output directory
          train_dir,    # Path to training data
          train_labels_path, # Path to train labels csv
          train_WSIlabels_path, # Path to patch wsi labels csv
          init_weight_path = None, # Path to initial model weights
          NUM_WORKERS = 2,
          small_test = False,
          AUX_LOSS = False,
          RESIZE = False,
          WARMUP = False,
          DECAY_FACTOR = None, # Factor decaying learning rate every 10'th epoch.
          PRETRAINED = True,
          RANDOM_VAL_SPLIT = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if init_weight_path != None:
        PRETRAINED = False

    #Create directory for checkpointing and metrics
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if RANDOM_VAL_SPLIT:
        val_wsi_keys = None
    else:
        val_wsi_keys = [1, 12, 16, 21, 24, 25, 27, 32, 36, 50, 53, 58, 62, 67, 74, 76, 82, 89, 98, 105, 108, 110, 115, 119, 121, 138, 140, 141, 151, 156]
    train_X, val_X, train_Y, val_Y, val_wsi_keys = generate_split(train_labels_path, train_WSIlabels_path, keys_for_cv = val_wsi_keys)
    print("Validation WSIs are:", val_wsi_keys)

    if small_test:
        N = 32
        train_X = train_X[:N]
        train_Y = [x%2 for x in range(N)]
        val_X = val_X[:N]
        val_Y = [x%2 for x in range(N)]

    train_aug = aug_train(aug_type=aug_type, p=aug_prob, RESIZE=RESIZE)
    val_aug = aug_val(RESIZE=RESIZE)

    train_loader = torch.utils.data.DataLoader(DataGenerator(train_X, train_Y, train_aug, train_dir),
                                               pin_memory=False,
                                               num_workers=NUM_WORKERS,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(DataGenerator(val_X, val_Y, val_aug, train_dir),
                                             num_workers=NUM_WORKERS,
                                             batch_size=BATCH_SIZE)

    loss_f = get_loss()
    model = get_model(use_aux_loss=AUX_LOSS, PRETRAINED=PRETRAINED, FREEZE=WARMUP, weight_path=init_weight_path).to(device)
    optimizer = get_optimizer(params=model.parameters(), optim_type=optim_type, LEARNING_RATE=LEARNING_RATE)
    
    scheduler = None
    if DECAY_FACTOR != None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=DECAY_FACTOR)
    #optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.9, eps=1.0) #These params made model unable to learn
    #optimizer = torch.optim.RMSprop(params=model.parameters(), lr=2e-5) #These params made val loss follow train loss but train stops at 0.27

    #Calculate initial validation loss
    best_loss, _, _ = evaluate(model, val_loader, loss_f, device)
    train_losses = []
    val_losses = [best_loss]
    
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, loss_f, len(train_Y), device)
        val_loss, _, _ = evaluate(model, val_loader, loss_f, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if scheduler != None:
            scheduler.step(val_loss)

        if val_loss<best_loss:
                print('Val loss improved from {:.6f} to {:.6f},'.format(best_loss, val_loss))
                best_loss = val_loss
                save_model(out_dir, model, is_best=True)
    
    # Log results
    save_model(out_dir, model, is_best=False)
    save_object(val_wsi_keys, os.path.join(out_dir, "val_wsi_keys"))
    save_object(train_losses, os.path.join(out_dir, "train_losses"))
    save_object(val_losses, os.path.join(out_dir, "val_losses"))

if __name__ == "__main__":
    train(BATCH_SIZE = 8,
          LEARNING_RATE = 1e-4,
          N_EPOCHS = 2,
          aug_prob = 0.5,
          aug_type = "all",
          optim_type = "adam",
          out_dir = 'supervised_results/test',
          train_dir = '../train',
          train_labels_path = 'train_labels.csv',
          train_WSIlabels_path = 'patch_id_wsi_full.csv',
          AUX_LOSS= False,
          PRETRAINED= True,
          small_test = True,
          DECAY_FACTOR=0.5)
