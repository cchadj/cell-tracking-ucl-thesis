from typing import List, Tuple

import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(torch.utils.data.Dataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """
    def __init__(self, images, standardize=True, to_grayscale=False, device='cuda'):
        """
        Args:
          images (ndarray):
              The images.
              shape -> N x height x width x channels or N x height x width. dtype should be uint8.

        """
        assert device in ['cuda', 'cpu'], f'Device must be  one of {["cuda", "cpu"]}'
        self.device = device

        #  Handle images ( self.images should be NxHxWxC even when C is 1, and image type should be uint8
        assert len(images.shape) == 3 or len(images.shape) == 4, \
            f'Expected images shape to be one of NxHxWxC or NxHxW. Shape given {images.shape}.'
        assert images.dtype == np.uint8, f'Images expected to have type uint8, not {images.dtype}.'
        assert 3 <= len(images.shape) <= 4, f'The images should be an array of shape N x H x W x [ C ] not {images.shape}'
        if len(images.shape) == 3:
            #  NxHxW  -> NxHxWx1
            images = images[..., None]
        self.n_images, self.height, self.width, self.n_channels = images.shape
        self.images = images

        # Handle transforms
        transforms = []
        if self.n_channels > 1 and to_grayscale:
            # Takes ndarray input with shape HxWxC
            transforms.append(torchvision.transforms.ToPILImage())
            # Grayscale takes a PILImage as an input
            transforms.append(torchvision.transforms.Grayscale(num_output_channels=1))
        # ToTensor accept uint8 [0, 255] numpy image of shape H x W x C and scales to [0, 1]
        transforms.append(torchvision.transforms.ToTensor())
        if standardize:
            # Standardization brings values to -1 and 1
            # Normalise takes input a tensor image of shape CxHxW and brings to target mean and standard deviation
            transforms.append(torchvision.transforms.Normalize(mean=[0.5], std=[0.5]))
        self.transform = torchvision.transforms.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()

        if self.transform and not torch.is_tensor(image):
            image = self.transform(image)

        return image


class LabeledImageDataset(ImageDataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """
    samples: List[Tuple[np.ndarray, int]]

    def __init__(self, images, labels, standardize=False, to_grayscale=False, device='cuda'):
        """
        Args:
          images (ndarray):
              The images.
              shape -> N x height x width x channels or N x height x width grayscale image.\
              Type must be uint8 ( values from 0 to 255)
          labels (ndarray):
              The corresponding list of labels.
              Should have the same length as images. (one label for each image)
        """
        super().__init__(images, standardize, to_grayscale, device)

        # if labels already ndarray nothing changes, if list makes to a numpy array
        labels = np.array(labels).squeeze()
        assert len(images) == len(labels), \
            f'Expected to have equal amount of labels and images. n images:{len(images)} n labels:{len(labels)}'
        assert labels.dtype == np.int, f'Labels must be integers not {labels.dtype}'
        assert len(labels.shape) == 1, f'Labels should be a list of one label for each image, shape given {labels.shape}'

        self.labels = torch.from_numpy(labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    # A test to check that the LabeledImageDataset and ImageDataset yields the same results
    from generate_datasets import get_cell_and_no_cell_patches, create_dataset_from_cell_and_no_cell_images

    _, _, cell_images, non_cell_images, _, _, _ = get_cell_and_no_cell_patches()

    labeled_dataset = LabeledImageDataset(cell_images, np.ones((len(cell_images), ), dtype=np.int), standardize=True)
    image_dataset = ImageDataset(cell_images, standardize=True)

    assert len(labeled_dataset) == len(image_dataset), 'The number of samples in the two datasets should be the same.'

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset), shuffle=False)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False)

    for images_1, (images_2, labels) in zip(image_loader, labeled_loader):
        assert images_1.allclose(images_2)

    labeled_dataset = LabeledImageDataset(cell_images, np.ones((len(cell_images), ), dtype=np.int), standardize=False)
    image_dataset = ImageDataset(cell_images, standardize=False)

    assert len(labeled_dataset) == len(image_dataset), 'The number of samples in the two datasets should be the same.'

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset), shuffle=False)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False)

    for images_1, (images_2, labels) in zip(image_loader, labeled_loader):
        assert images_1.allclose(images_2)

    labeled_dataset = LabeledImageDataset(cell_images, np.ones((len(cell_images), ), dtype=np.int), standardize=True)
    image_dataset = ImageDataset(cell_images, standardize=False)

    assert len(labeled_dataset) == len(image_dataset), 'The number of samples in the two datasets should be the same.'

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset), shuffle=False)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False)

    for images_1, (images_2, labels) in zip(image_loader, labeled_loader):
        assert not images_1.allclose(images_2)
