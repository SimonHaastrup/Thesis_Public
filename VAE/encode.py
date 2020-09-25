import os
import h5py
from model import *
from utils import DataGenerator_H5, aug_val
import numpy as np

def encode(h5_dir, weight_path, save_dir, BATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    model = load_model(model, weight_path, device)
    model.eval()
    
    h5_names =  [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    h5_names.sort()
    print("Encoding", len(h5_names), "images...")
    for h5_name in h5_names:
        h5_path = os.path.join(h5_dir, h5_name)
        h5_file = h5py.File(h5_path, 'r')
        n_images, _, _, _ = h5_file['imgs'].shape
        print("Working on:", h5_name, "Encoding", n_images, "patches")

        encoding_loader = torch.utils.data.DataLoader(DataGenerator_H5(h5_path, aug_val(RESIZE=True)),
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False,
                                                      drop_last=False)

        vae_embeddings = []
        rgb_embeddings = []
        with torch.no_grad():
            for x in encoding_loader:
                #VAE embedding
                vae_embedding = model.encode(x.to(device, dtype=torch.float))
                vae_embeddings.append(vae_embedding.cpu())

                #RGB embedding
                N, _, _, _ = x.shape
                rgb_embedding = np.zeros((N, 3))
                for i in range(N):
                    rgb_embedding[i,0] = x[i,0,:,:].mean()
                    rgb_embedding[i,1] = x[i,1,:,:].mean()
                    rgb_embedding[i,2] = x[i,2,:,:].mean()
                rgb_embeddings.append(rgb_embedding)

        rgb_embeddings = np.concatenate(rgb_embeddings)
        vae_embeddings = torch.cat(vae_embeddings, dim=0).numpy()
        with h5py.File(os.path.join(save_dir, h5_name),'w') as f: 
            f.create_dataset("x_vae", data = vae_embeddings)
            f.create_dataset("x_rgb", data = rgb_embeddings)
            f.create_dataset("coords", data=h5_file["coords"], dtype="i4")
    print("Successfully encoded images")

if __name__ == "__main__":
    encode(h5_dir="tiled_CAM16/train",
           weight_path="results\initial_paper_runs_2707\paper_long_all_aug\checkpoint.best.pth.tar",
           save_dir="encoded_CAM16/train",
           BATCH_SIZE = 16)