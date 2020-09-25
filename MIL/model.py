import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(device, attn_neurons, enc_size):
    return Attention(attn_neurons, enc_size).to(device)

# This model is a modification of the work of Ilse et al.
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL
class Attention(nn.Module):
    def __init__(self, attn_neurons=128, enc_size=256):
        super(Attention, self).__init__()
        self.M = enc_size
        self.L = attn_neurons
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x): #Ensure dimensionality
        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        Z = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A