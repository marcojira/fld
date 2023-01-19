import pickle
import torch
import torchvision.transforms as TF
from tqdm import tqdm
import os

class ActivationsModule():
    def __init__(self, recompute=False):
        self.recompute = recompute
        self.path = f"/home/mila/m/marco.jiralerspong/scratch/ills/activations/{self.name}/"
        os.makedirs(self.path, exist_ok=True)
    
    def get_activation_batch(self, img_batch):
        """ TO BE IMPLEMENTED BY EACH MODULE """
        pass

    def get_activations(self, dataset):
        file_path = os.path.join(self.path, f"{dataset.name}.pkl")
        
        if not self.recompute and os.path.exists(file_path):
            print(f"Loading {file_path} from cache")
            return self.load_activations(file_path)
        
        activations = self.compute_activations(dataset)
        self.save_activations(activations, file_path)
        return activations
    
    def compute_activations(self, dataset):
        activations = torch.zeros(len(dataset), self.activations_size)
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, drop_last=False
        )
        
        start_idx = 0
        for img_batch, _ in tqdm(dataloader, leave=False):
            if self.transform:
                img_batch = self.transform(img_batch)

            activation = self.get_activation_batch(img_batch.cuda())

            activations[start_idx:start_idx + activation.shape[0]] = activation
            start_idx = start_idx + activation.shape[0]

        return activations
    
    def get_path(self, dataset):
        path = f"data/activations/{self.name}/"
        file_path = os.path.join(path, f"{dataset.name}.pkl")
        return file_path

    def save_activations(self, activations, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(activations, f)
            return activations
            
    def load_activations(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
