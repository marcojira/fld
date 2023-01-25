from fls.datasets.DatasetSet import DatasetSet
import torchvision
import torchvision.transforms as TF
import os
from fls.datasets.SamplesDataset import SamplesDataset


class LSUNBedroomDatasetSet(DatasetSet):
    def __init__(self):
        super().__init__()
        self.name = "LSUNBedroom256"
        self.sample_datasets = []

        self.train_dataset = SamplesDataset(
            "lsun_bedroom256_train",
            path="/home/mila/m/marco.jiralerspong/scratch/lsun/bedroom",
            extension="jpg",
            transform=TF.Resize(
                (256, 256), interpolation=TF.InterpolationMode.BILINEAR
            ),
        )

        self.test_dataset = SamplesDataset(
            "lsun_bedroom256_val",
            path="/home/mila/m/marco.jiralerspong/scratch/lsun/bedroom_val/0/0/0/",
            extension="webp",
            transform=TF.Resize(
                (256, 256), interpolation=TF.InterpolationMode.BILINEAR
            ),
        )

        """ Add sample datasets """
        path_1 = "/home/mila/m/marco.jiralerspong/scratch/ills/samples"
        main_path_models = [
            "diffusion_projgan",
            "diffusion_stylegan2",
            "guided_diffusion",
            "improved_diffusion",
            "latent_diffusion",
            "projgan",
        ]

        for model_name in main_path_models:
            self.sample_datasets.append(
                SamplesDataset(
                    f"lsun_bedroom256_{model_name}",
                    os.path.join(path_1, model_name, "lsun_bedroom256"),
                )
            )

        self.create_dataset_dict()
