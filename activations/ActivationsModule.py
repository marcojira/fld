import pickle
import torch
import torchvision.transforms as TF
from tqdm import tqdm
import os
import numpy as np

class ActivationsModule():
    def __init__(self, recompute=False, save=True):
        self.recompute = recompute
        self.save = save
        self.path = f"/home/mila/m/marco.jiralerspong/scratch/ills/activations/{self.name}/"
        os.makedirs(self.path, exist_ok=True)

    def preprocess_batch(self, img_batch):
        """ TO BE IMPLEMENTED BY EACH MODULE """
        pass
    
    def get_activation_batch(self, img_batch):
        """ TO BE IMPLEMENTED BY EACH MODULE """
        pass

    def get_activations(self, dataset, size=None, get_indices=False, verbose=False):
        file_path = os.path.join(self.path, f"{dataset.name}.pkl")
        
        if not self.recompute and os.path.exists(file_path):
            if verbose:
                print(f"Loading {file_path} from cache")

            if get_indices:
                return self.load_activations(file_path)
            else:
                return self.load_activations(file_path)[0]
        
        activations, indices = self.compute_activations(dataset, size)
        if self.save:
            self.save_activations(activations, indices, file_path)

        if get_indices:
            return activations, indices
        else:
            return activations
    
    def compute_activations(self, dataset, size=None):
        if not size:
            size = len(dataset)

        activations = torch.zeros(size, self.activations_size)
        indices = np.random.choice(len(dataset), size=size, replace=False)

        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            subset, batch_size=256, drop_last=False
        )
        
        start_idx = 0
        for img_batch, _ in tqdm(dataloader, leave=False):
            img_batch = self.preprocess_batch(img_batch)
            activation = self.get_activation_batch(img_batch.cuda())

            # If going to overflow, just get required amount and break
            if size and start_idx + activation.shape[0] > size:
                activations[start_idx:] = activation[:size-start_idx]
                break

            activations[start_idx:start_idx + activation.shape[0]] = activation
            start_idx = start_idx + activation.shape[0]

        return activations, indices
    
    def get_path(self, dataset):
        path = f"data/activations/{self.name}/"
        file_path = os.path.join(path, f"{dataset.name}.pkl")
        return file_path

    def save_activations(self, activations, indices, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((activations, indices), f)
            return activations
            
    def load_activations(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
