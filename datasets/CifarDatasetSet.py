from fls.datasets.DatasetSet import DatasetSet
import torchvision
import torchvision.transforms as TF
import os
from fls.datasets.SamplesDataset import SamplesDataset
import torchvision.datasets
import numpy as np


class ciFAIR10(torchvision.datasets.CIFAR10):
    base_folder = "ciFAIR-10"
    url = "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip"
    filename = "ciFAIR-10.zip"
    tgz_md5 = "ca08fd390f0839693d3fc45c4e49585f"
    test_list = [
        ["test_batch", "01290e6b622a1977a000eff13650aca2"],
    ]


class ciFAIR100(torchvision.datasets.CIFAR100):
    base_folder = "ciFAIR-100"
    url = "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip"
    filename = "ciFAIR-100.zip"
    tgz_md5 = "ddc236ab4b12eeb8b20b952614861a33"
    test_list = [
        ["test", "8130dae8d6fc6a436437f0ebdb801df1"],
    ]


class CifarDatasetSet(DatasetSet):
    def __init__(self):
        super().__init__()
        self.name = "CIFAR10"
        self.sample_datasets = []

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.train_dataset.name = "cifar10_train"

        self.test_dataset = ciFAIR10(
            root="./data",
            train=False,
            download=True,
            transform=None,
        )
        self.test_dataset.name = "cifair10_test"
        # self.test_dataset = torchvision.datasets.CIFAR10(
        #     root="./data", train=False, download=True
        # )
        # self.test_dataset.name = "cifar10_test"

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
