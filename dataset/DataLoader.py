from base.torchvision_dataset import TorchvisionDataset
from torchvision.datasets import CIFAR10
from PIL import Image
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random
import numpy as np


class CIFAR10Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class: int = 5, known_outlier_class: int = 3, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)
        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)  # (0, 1, 2, 3, 4, 6, 7, 8, 9)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCIFAR10(root=self.root, train=True, transform=transform, target_transform=target_transform,
                              download=True)

        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier,
                                                             ratio_pollution)

        train_set.semi_targets[idx] = torch.tensor(semi_targets)

        self.test_set = MyCIFAR10(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)


class MyCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

    def __getitem__(self, index):
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
