import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset


class LabeledImageDataset(torch.utils.data.Dataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """

    def __init__(self,
                 images,
                 labels):
        """
        Args:
          images (ndarray):
              The images.
              shape -> N x height x width x channels
          labels (ndarray):
              The corresponding labels
        """
        self.n_images = images.shape[0]
        self.n_channels = images.shape[-1]

        images = images.astype(np.float32)
        self.transform = torchvision.transforms.Compose([
            # you can add other transformations in this list
            torchvision.transforms.ToTensor()
        ])

        self.samples = [(image, label) for image, label in zip(images, labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img, target = sample[0], sample[1]

        # print(img.shape)
        # print(type(img))
        # print(img.dtype)
        if self.transform:
            img = self.transform(img)
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
            torchvision.transforms.ToTensor()
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
