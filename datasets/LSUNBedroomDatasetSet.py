from datasets.DatasetSet import DatasetSet
import torchvision
import torchvision.transforms as TF
import os
from datasets.SamplesDataset import SamplesDataset


class LSUNBedroomDatasetSet(DatasetSet):
    def __init__(self):
        super().__init__()
        self.name = "LSUNBedroom256"
        self.sample_datasets = []
