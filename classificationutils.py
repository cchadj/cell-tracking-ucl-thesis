import torch
import numpy as np
from torch.utils.data import Dataset
from learningutils import ImageDataset, LabeledImageDataset
from patchextraction import extract_patches
from imageprosessing import imextendedmax
from matplotlib import pyplot as plt
import mahotas as mh
import cv2

NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


@torch.no_grad()
def classify_labeled_dataset(dataset, model, device="cuda"):
    model = model.eval()
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
    )

    n_correct = 0
    c = 0
    predictions = torch.empty(len(dataset), dtype=torch.long).to(device)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        predictions[c:c + pred.shape[0]] = pred

        n_correct += (pred == labels).sum().item()
        c += pred.shape[0]

    accuracy = n_correct / len(dataset)
    return predictions, accuracy


@torch.no_grad()
def get_label_probability(images, model, standardize=True, to_grayscale=False, device='cuda'):
    """ Make a prediction for the images giving probabilities for each labels.

    Arguments:
        images -- NxHxWxC or NxHxW. The images
        model  -- The model to do the prediction

    Returns:
        Returns the probability per label for each image.
    """
    model = model.to(device)
    model = model.eval()

    if len(images.shape) == 3:
        # Add channel dimension when images are single channel grayscale
        # i.e (Nx100x123 -> Nx100x123x1)
        images = images[..., None]

    image_dataset = ImageDataset(images, standardize=standardize, to_grayscale=to_grayscale)
    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1024 * 3,
    )

    c = 0
    predictions = torch.empty(len(image_dataset), dtype=torch.float32)
    for images in loader:
        images = images.to(device)

        pred = model(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        # print('_-_-_-_-_-_-_-_-_')
        # print(batch.shape)
        # print(pred.shape)
        # print(predictions[c:c + len(pred), ...].shape)
        # print(c, len(pred))
        # print('_-_-_-_-_-_-_-_-_')
        predictions[c:c + len(pred), ...] = pred
        c += pred.shape[0]

    return predictions


@torch.no_grad()
def predict_probability_single_patch(patch, model, device='cuda'):
    """ Makes a prediction on a single patch
    """

    if patch.dtype == np.uint8:
        patch = np.float32(patch) / 255

    patch = np.squeeze(patch)
    patch = patch[np.newaxis, :, :, np.newaxis]

    dataset = ImageDataset(patch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for batch in dataloader:
        batch = batch.to(device)
        output = model(batch)

        prediction = torch.nn.functional.softmax(output, dim=1)[:, 1]

    return prediction.item()


@torch.no_grad()
def create_probability_map(
        image,
        model,
        patch_size=(21, 21),
        padding=cv2.BORDER_REPLICATE,
        standardize=True,
        to_grayscale=False,
        device='cuda',
        mask=None,
):
    assert type(patch_size) == int or type(patch_size) == tuple, 'Patch size must be tuple or int.'
    if type(patch_size) == int:
        patch_size = patch_size, patch_size

    model = model.to(device)
    model = model.eval()

    if len(image.shape) == 2:
        # Add channel at end if grayscale. HxW -> HxWx1
        image = image[:, :, None]

    # print('Image shape', image.shape)
    # if mask is not None then create patches for every pixel.
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.bool)

    # print('Mask shape', mask.shape)
    # flatten mask to get indices to index patches
    mask_flattened = mask.reshape(-1)
    vessel_pixel_indices = np.where(mask_flattened)[0]

    patches = extract_patches(image, patch_size, padding=padding)[vessel_pixel_indices]
    label_probabilities = get_label_probability(patches, model, standardize=standardize,
                                                to_grayscale=to_grayscale, device=device)

    probability_map = np.zeros(image.shape[:2], dtype=np.float32)
    rows, cols = np.unravel_index(vessel_pixel_indices, probability_map.shape[:2])
    probability_map[rows, cols] = label_probabilities[:, 1]

    return probability_map


def get_cell_positions_from_probability_map(probability_map,
                                            gauss_sigma,
                                            extended_maxima_H,
                                            visualise_intermediate_results=False):
    pm_blurred = mh.gaussian_filter(probability_map, gauss_sigma)
    pm_extended_max_bw = imextendedmax(pm_blurred, extended_maxima_H)

    labeled, nr_objects = mh.label(pm_extended_max_bw)

    # print(np.where(pm_extended_max_bw)[0])
    pm_extended_max = probability_map.copy()
    pm_extended_max[pm_extended_max_bw] = 0

    # print(pm_extended_max)
    # Notice, the positions from the csv is x,y. The result from the probability is y,x so we swap.
    predicted_cell_positions = mh.center_of_mass(pm_extended_max_bw, labeled)[:, [1, 0]]
    if visualise_intermediate_results:
        fig, axes = plt.subplots(1, 3)
        fig_size = fig.get_size_inches()
        fig.set_size_inches((fig_size[0] * 5,
                             fig_size[1] * 5))

        axes[0].imshow(probability_map)
        axes[0].set_title('Unprocessed probability map')

        axes[1].imshow(pm_blurred)
        axes[1].set_title(f'Gaussian Blurring with sigma={gauss_sigma}')

        axes[2].imshow(pm_extended_max_bw)
        axes[2].set_title(f'Extended maximum, H={extended_maxima_H}')

        axes[2].scatter(predicted_cell_positions[:, 0], predicted_cell_positions[:, 1], s=4)
        # axes[2].scatter(predicted_cell_positions[:, 0], predicted_cell_positions[:, 1], s=9)

    return predicted_cell_positions[1:, ...]


@torch.no_grad()
def classify_images(images, model, standardize_dataset=True, device="cuda"):
    """ Classify images.

    Arguments:
        images -- NxHxWxC or NxHxW. The images
        model  -- The model to do the prediction

    Returns:
        N predictions. A prediction (label) for each image.
    """
    if len(images.shape) == 3:
        # Add channel dimension when image is single channel grayscale
        # i.e (Nx100x123 -> Nx100x123x1)
        images = images[..., None]

    image_dataset = ImageDataset(images, standardize=standardize_dataset)
    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1024 * 5,
        shuffle=False
    )

    c = 0
    predictions = torch.zeros(len(image_dataset), dtype=torch.uint8)
    for batch in loader:
        pred = model(batch.to(device))
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        predictions[c:c + len(pred)] = pred

        c += len(pred)

    return predictions


if __name__ == '__main__':
    from generate_datasets import get_cell_and_no_cell_patches
