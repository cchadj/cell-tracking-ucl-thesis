import torch
import matplotlib.pyplot as plt
import torchvision
from matplotlib.patches import Rectangle
import numpy as np


def plot_patch_rois_at_positions(positions, patch_size=(21, 21),
                                 edgecolor='r', pointcolor='b', label=None,
                                 ax=None, image=None):
    assert type(patch_size) is int or type(patch_size) is tuple
    if type(patch_size) is int:
        patch_size = patch_size, patch_size
    patch_height, patch_width = patch_size

    if ax is None:
        _, ax = plt.subplots()
    if image is not None:
        ax.imshow(image, cmap='gray')

    positions = positions.astype(np.int32)
    ax.scatter(positions[:, 0], positions[:, 1], label=label, c=pointcolor)
    for patch_count, (x, y) in enumerate(positions.astype(np.int32)):
        rect = Rectangle((x - patch_width / 2,
                          y - patch_height / 2),
                         patch_width, patch_height,
                         linewidth=1, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)
        ax.annotate(patch_count - 1, (x, y))


def plot_dataset_as_grid(dataset, title=None):
    """ Plots a stack of images in a grid.

    Arguments:
        dataset: The dataset
        title: Plot title
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=60000,
        shuffle=False
    )

    for batch in loader:
        images = batch[0]
        labels = batch[1]

        print("Images:", images.shape)
        print("Labels:", labels.shape)
        grid_img = torchvision.utils.make_grid(images, nrow=50)

        plt.figure(num=None, figsize=(70, 50), dpi=80, facecolor='w', edgecolor='k')
        plt.title(title)
        plt.grid(b=None)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()


def plot_images_as_grid(images, ax=None, title=None):
    """
    Plots a stack of images in a grid.

    Arguments:
        images: The images as NxHxWxC
        title: Plot title
    """
    if len(images.shape) == 3:
        images = images[..., None]

    batch_tensor = torch.from_numpy(images)
    # NxHxWxC -> NxCxHxW
    batch_tensor = batch_tensor.permute(0, -1, 1, 2)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=90)
    if ax is None:
        _, ax = plt.subplots(num=None, figsize=(70, 50), dpi=80, facecolor='w', edgecolor='k')
    if title is not None:
        ax.set_title(title)

    plt.grid(b=None)
    ax.imshow(grid_img.permute(1, 2, 0))
    ax.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        labelleft=False,
        bottom=False,
        top=False,
        labelbottom=False)
