import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import *
from model import get_model
from data_split import generate_split

def evaluate(model, test_loader, loss_f, device):
    model.eval()
    #targets = []
    #predictions= []
    losses = []
    rec_losses = []
    kl_losses = []
    with torch.no_grad():
        for x in test_loader:
            device_x = x.to(device, dtype=torch.float)
            #Apply model
            x_rec, mu, var = model(device_x)
            #Calculate loss
            loss, rec_loss, kl_loss = loss_f(device_x, x_rec, mu, var)
            losses.append(loss.item())
            rec_losses.append(rec_loss.item())
            kl_losses.append(kl_loss.item())
            #predictions.append(x_rec.cpu())
            #targets.append(x)
    #predictions = np.vstack(predictions)
    #targets = np.vstack(targets)
    loss = np.mean(losses)
    rec_loss = np.mean(rec_losses)
    kl_loss = np.mean(kl_losses) 

    print('\nTest set: Average loss: {:.6f}, Rec loss: {:.6f}, KL loss: {:.6f}'.format(loss, rec_loss, kl_loss))
    return (loss, rec_loss, kl_loss)#, predictions, targets