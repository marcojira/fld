from fls.datasets.DatasetSet import DatasetSet
import torchvision
import torchvision.transforms as TF
import os
from fls.datasets.SamplesDataset import SamplesDataset


class CifarDatasetSet(DatasetSet):
    def __init__(self):
        super().__init__()
        self.name = "CIFAR10"
        self.sample_datasets = []

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.train_dataset.name = "cifar10_train"

        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True
        )
        self.test_dataset.name = "cifar10_test"

        """ Add non studiogan datasets """
        path_1 = "/home/mila/m/marco.jiralerspong/scratch/ills/sample_generation/image_samples/"
        main_path_models = ["ddpm", "stylegan2_ada", "dcgan_batch", "nvae"]

        for model_name in main_path_models:
            self.sample_datasets.append(
                SamplesDataset(
                    f"cifar10_{model_name}", os.path.join(path_1, model_name)
                )
            )

        path_2 = "/home/mila/m/marco.jiralerspong/scratch/ills/samples"
        model_names = [
            "diffusion_projgan",
            "diffusion_stylegan2",
            "improved_diffusion",
            "stylegan_xl",
        ]
        for model_name in model_names:
            self.sample_datasets.append(
                SamplesDataset(
                    f"cifar10_{model_name}", os.path.join(path_2, model_name, "cifar32")
                )
            )

        """ Add studiogan datasets """
        SAVE_PATH = "/home/mila/m/marco.jiralerspong/scratch/ills/sample_generation/studiogan/CIFAR10_tailored/samples/"
        studiogan_models = [
            "ACGAN-Mod",
            "BigGAN-CR",
            "DCGAN",
            "LOGAN",
            "LSGAN",
            "ProjGAN",
            "ReACGAN",
            "SAGAN",
            "SNGAN",
            "WGAN-GP",
        ]
        self.get_studiogan_datasets(studiogan_models, SAVE_PATH, "cifar10")

        self.create_dataset_dict()
