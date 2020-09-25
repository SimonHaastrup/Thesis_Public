import os
import numpy as np
import torch
import random
import time
from sklearn.utils.class_weight import compute_class_weight

from utils import DataGenerator_Bag, save_model, save_object
from model import get_model
from test import evaluate

def train_epoch(model, train_loader, optimizer, epoch, loss_f, epoch_size, device, class_weights):
    model.train()
    epoch_losses = []
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        #Apply model
        output, _, _ = model(x[0,:,:].to(device, dtype=torch.float)) # Indexing into x for batch_size=1
        
        output = torch.clamp(output, min=1e-5, max=1. - 1e-5) #Clamp probability for numerical probability
        #Calculate loss

        loss = loss_f(output, target.to(device, dtype=torch.float)) / len(target) # Division to counteract loss accum
        if class_weights: #Loss weighting for batch_size=1
            loss = loss * class_weights[target]

        #Adjust weights via backprop
        loss.backward()
        optimizer.step()

        #Log loss
        # print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
        #     epoch, (batch_idx + 1) * len(x), epoch_size,
        #     100. * (batch_idx + 1) * len(x) / epoch_size, loss))
        epoch_losses.append(loss.item())

    train_loss_mean = np.mean(epoch_losses)
    print('Mean train loss on epoch {} : {:.6f}'.format(epoch, train_loss_mean))
    return epoch_losses

def train(data_dir_normal,
          data_dir_tumor,
          out_dir,
          E_DIM,
          BATCH_SIZE,
          N_EPOCHS,
          ATTN_NEURONS,
          DECAY_PATIENCE,
          DECAY_FACTOR,
          LEARNING_RATE,
          SMALL_TEST=False,
          FIXED_SEED=True,
          WEIGHT_LOSS=False):
    if FIXED_SEED:
        random.seed(1)

    #Create directory for checkpointing and metrics
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader
    X_normal = os.listdir(data_dir_normal)
    X_normal = [os.path.join(data_dir_normal, e) for e in X_normal]
    X_tumor = os.listdir(os.path.join(data_dir_tumor)) 
    X_tumor = [os.path.join(data_dir_tumor, e) for e in X_tumor]
    random.shuffle(X_normal)
    random.shuffle(X_tumor)
    
    train_X = X_normal[len(X_normal)//5:] + X_tumor[len(X_tumor)//5:]
    val_X = X_normal[:len(X_normal)//5] + X_tumor[:len(X_tumor)//5]
    random.shuffle(train_X)
    random.shuffle(val_X)

    train_Y = [0 if x.split("\\")[-1].split("_")[0] == "normal" else 1 for x in train_X]
    val_Y = [0 if x.split("\\")[-1].split("_")[0] == "normal" else 1 for x in val_X]
    print("{} positive train bags".format(sum(train_Y)/len(train_Y)))
    print("{} positive val bags".format(sum(val_Y)/len(val_Y)))

    NUM_WORKERS = 4
    train_loader = torch.utils.data.DataLoader(DataGenerator_Bag(train_X, train_Y, E_DIM),
                                               pin_memory=False,
                                               num_workers=NUM_WORKERS,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(DataGenerator_Bag(val_X, val_Y, E_DIM),
                                             num_workers=NUM_WORKERS,
                                             batch_size=BATCH_SIZE)

    # Model
    model = get_model(device=device, attn_neurons=ATTN_NEURONS, enc_size=E_DIM) 

    # Loss function, optimizer
    loss_f = torch.nn.BCELoss(reduction="sum")
    class_weights = None
    if WEIGHT_LOSS:
        class_weights = list(compute_class_weight('balanced', np.unique(train_Y), train_Y))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=DECAY_PATIENCE, verbose=True, min_lr = 1e-5, factor=DECAY_FACTOR)
    
    # Training
    best_loss, _, _ = evaluate(model, val_loader, loss_f, device)
    train_losses = []
    val_losses = [best_loss]
    
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, loss_f, len(train_Y), device, class_weights)
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
    save_object(val_X, os.path.join(out_dir, "val_wsi_bag_paths"))
    save_object(train_losses, os.path.join(out_dir, "train_losses"))
    save_object(val_losses, os.path.join(out_dir, "val_losses"))
    save_object(optimizer.param_groups[0]['lr'], os.path.join(out_dir, "final_lr"))
    
    print("Done")

if __name__ == "__main__":
    #Final VAE training
    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/train/best_vae_experiments/1",
          E_DIM = 256,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 128,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-4,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)

    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/train/best_vae_experiments/2",
          E_DIM = 256,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 128,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-4,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)

    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/train/best_vae_experiments/3",
          E_DIM = 256,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 128,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-4,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)

    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/train/best_vae_experiments/4",
          E_DIM = 256,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 128,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-4,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)

    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/train/best_vae_experiments/5",
          E_DIM = 256,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 128,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-4,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)

    #Final RGB training
    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/best_rgb/1",
          E_DIM = 3,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 32,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-3,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)

    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/best_rgb/2",
          E_DIM = 3,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 32,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-3,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)
    
    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/best_rgb/3",
          E_DIM = 3,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 32,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-3,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)
    
    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/best_rgb/4",
          E_DIM = 3,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 32,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-3,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)
    
    train(data_dir_normal = "data/training/encoded_normal",
          data_dir_tumor = "data/training/encoded_tumor",
          out_dir = "results/best_rgb/5",
          E_DIM = 3,
          BATCH_SIZE = 1,
          N_EPOCHS = 100,
          ATTN_NEURONS = 32,
          DECAY_PATIENCE = 10,
          DECAY_FACTOR = 0.5,
          LEARNING_RATE = 1e-3,
          WEIGHT_LOSS=False,
          FIXED_SEED=False)
    
    
    


    
