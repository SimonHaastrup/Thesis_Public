import torch
import torch.nn as nn
import os

def get_model(device, e_dim=128, use_sigmoid=True):
    return VAE(device, e_dim=e_dim, use_sigmoid=use_sigmoid).to(device)

def load_model(model, weight_path, device):
    if not os.path.isfile(weight_path):
        raise Exception("Attempted to load model weights with invalid filename: " + weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 512, 4, 4)

class VAE(nn.Module):
    def __init__(self, device, use_sigmoid=False, image_channels=3, h_dim=512, e_dim=128):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1), #32x64x64 Output dim
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #64x32x32
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #128x16x16
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #256x8x8
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), #512x4x4
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),

            Flatten(), #Nx8192

            nn.Linear(8192, h_dim), #Nx512
            nn.BatchNorm1d(num_features=h_dim),
            nn.LeakyReLU()
        )
        
        self.fc_mu = nn.Linear(h_dim, e_dim)
        self.fc_var =  nn.Sequential(
            nn.Linear(h_dim, e_dim),
            nn.Softplus() #Try experiment with Relu instead of softplus
        )
        self.fc_rehide = nn.Linear(e_dim, h_dim)
        
        if use_sigmoid:
            self.decoder = nn.Sequential(
                nn.Linear(h_dim, 8192), #Nx8192
                nn.BatchNorm1d(num_features=8192),
                nn.LeakyReLU(),
                
                UnFlatten(), #512x4x4

                nn.Upsample(scale_factor=2, mode="nearest"), #512x8x8
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), #256x8x8
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #256x16x16
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), #128x16x16
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #128x32x32
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), #64x32x32
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #64x64x64
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), #32x64x64
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #32x128x128
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), #16x128x128
                nn.BatchNorm2d(num_features=16),
                nn.LeakyReLU(),

                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1), #3x128x128
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(h_dim, 8192), #Nx8192
                nn.BatchNorm1d(num_features=8192),
                nn.LeakyReLU(),
                
                UnFlatten(), #512x4x4

                nn.Upsample(scale_factor=2, mode="nearest"), #512x8x8
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), #256x8x8
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #256x16x16
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), #128x16x16
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #128x32x32
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), #64x32x32
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #64x64x64
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), #32x64x64
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),

                nn.Upsample(scale_factor=2, mode="nearest"), #32x128x128
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), #16x128x128
                nn.BatchNorm2d(num_features=16),
                nn.LeakyReLU(),

                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1), #3x128x128
                nn.Tanh()
            )
        
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn(*mu.size()).to(self.device)
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu = self.fc_mu(h)
        var = self.fc_var(h) + 1e-5 #Add epsilon for numerical stability
        z = self.reparameterize(mu, var)
        return z, mu, var
        
    def encode(self, x):
        _, mu, var = self.bottleneck(self.encoder(x))
        embedding = torch.cat((mu, var), dim = 1) 
        return embedding

    def forward(self, x):
        h = self.encoder(x)
        z, mu, var = self.bottleneck(h)
        z = self.fc_rehide(z)
        return self.decoder(z), mu, var