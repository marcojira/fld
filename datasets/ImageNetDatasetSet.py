import torchvision
import torchvision.transforms as TF
import os
import shutil

from fls.datasets.DatasetSet import DatasetSet
from fls.datasets.SamplesDataset import SamplesDataset
from fls.datasets.ImageNetSampleDataset import ImageNetSampleDataset


class ImageNetDatasetSet(DatasetSet):
    def __init__(self, slurm_tmpdir=None):
        super().__init__()
        self.name = "ImageNet"
        self.sample_datasets = []

        if slurm_tmpdir:
            self.load_dataset(slurm_tmpdir)
        else:
            self.train_dataset = ImageNetSampleDataset("imagenet_train", "train")
            self.test_dataset = ImageNetSampleDataset("imagenet_val", "val")

        """ Add non studiogan datasets """
        base_path = "/home/mila/m/marco.jiralerspong/scratch/ills/samples"
        main_path_models = ["guided_diffusion"]

        for model_name in main_path_models:
            self.sample_datasets.append(
                SamplesDataset(
                    f"imagenet_{model_name}",
                    os.path.join(base_path, model_name, "imagenet256"),
                    transform=TF.Resize(
                        (128, 128), interpolation=TF.InterpolationMode.BILINEAR
                    ),
                )
            )

        self.sample_datasets.append(
            SamplesDataset(
                f"imagenet_styleganxl",
                "/home/mila/m/marco.jiralerspong/scratch/truncations_styleganxl/unconditional/0.70",
            )
        )

        """ Add studiogan datasets """
        SAVE_PATH = "/home/mila/m/marco.jiralerspong/scratch/ills/sample_generation/studiogan/ImageNet_tailored/samples"
        studiogan_models = [
            "BigGAN-256",
            "ContraGAN-256",
            "ReACGAN-256",
            "SAGAN-256",
            "SNGAN-256",
            "StyleGAN2-SPD",
            "StyleGAN3-t-SPD",
        ]
        self.get_studiogan_datasets(studiogan_models, SAVE_PATH, "imagenet128")

        self.create_dataset_dict()

    def load_dataset(self, slurm_tmpdir):
        base_path = "/network/datasets/imagenet/"

        file_names = [
            "ILSVRC2012_img_val.tar",
            "ILSVRC2012_img_train.tar",
            "ILSVRC2012_devkit_t12.tar.gz",
        ]

        for file_name in file_names:
            shutil.copy(
                os.path.join(base_path, file_name),
                os.path.join(slurm_tmpdir, file_name),
            )

        self.train_dataset = torchvision.datasets.ImageNet(
            root=slurm_tmpdir,
            split="train",
            transform=TF.Compose(
                [TF.Resize((128, 128), interpolation=TF.InterpolationMode.BILINEAR)]
            ),
        )
        self.train_dataset.name = "imagenet_train"

        self.test_dataset = torchvision.datasets.ImageNet(
            root=slurm_tmpdir,
            split="val",
            transform=TF.Compose(
                [TF.Resize((128, 128), interpolation=TF.InterpolationMode.BILINEAR)]
            ),
        )
        self.test_dataset.name = "imagenet_val"
