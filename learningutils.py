from typing import List, Any, Tuple

import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset


class LabeledImageDataset(torch.utils.data.Dataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """
    samples: List[Tuple[np.ndarray, int]]

    def __init__(self,
                 images,
                 labels):
        """
        Args:
          images (ndarray):
              The images.
              shape -> N x height x width x channels or N x height x width grayscale image.
          labels (ndarray):
              The corresponding labels
        """
        self.n_images = images.shape[0]
        self.n_channels = images.shape[-1]

        # ToTensor accept uint8 [0, 255] numpy image of shape H x W x C
        # print("A", images.shape)
        if len(images.shape) == 3:
            # images = np.concatenate((images[..., None], images[..., None], images[..., None]), axis=-1)
            # if shape is NxHxW -> NxHxWx1
            images = images[..., np.newaxis]

        self.transform = torchvision.transforms.Compose([
            # you can add other transformations in this list
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

        # print("B", images.shape)
        self.samples = [(image, label) for image, label in zip(images, labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img, target = sample[0], sample[1]

        # print('C', img.shape)
        if self.transform:
            img = self.transform(img)
            # print('D', img.shape)

        # print(img.shape)
        return img, target


class ImageDataset(torch.utils.data.Dataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """
    def __init__(self, images):
        """
        Args:
          images (ndarray):
              The images.
              shape -> N x height x width x channels
        """
        self.n_images = images.shape[0]

        self.transform = torchvision.transforms.Compose([
            # you can add other transformations in this list
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])

        self.samples = images
        # print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.samples[idx].copy()
        # print(image.shape)
        # print(type(image))
        # print(image.dtype)
        # print("Is tensor?", self.transform and not torch.is_tensor(image))
        if self.transform and not torch.is_tensor(image):
            image = self.transform(image)

        return image
