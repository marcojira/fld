from torch.utils.data import Dataset


class PlaceholderDataset(Dataset):
    """ 
    Placeholder dataset for datasets that take a long time to create (e.g. ImageNet) 
    but for which activations have already been computed
    """
    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def __len__(self):
        return self.size