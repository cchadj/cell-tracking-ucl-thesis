import numpy as np
from torch.utils.data import Dataset
from imageprosessing import imextendedmax
from matplotlib import pyplot as plt

import mahotas as mh
import torch
from learningutils import ImageDataset


NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


@torch.no_grad()
def classify_labeled_dataset(dataset, model, ret_pos_and_neg_acc=False, device="cuda"):
    model = model.eval()
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
    )

    n_correct = 0

    n_positive_correct = 0
    n_negative_correct = 0
    n_positive_samples = 0
    n_negative_samples = 0

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

        positive_indices = torch.where(labels == 1)[0]
        n_positive_correct += (pred[positive_indices] == labels[positive_indices]).sum().item()
        n_positive_samples += len(positive_indices)

        negative_indices = torch.where(labels == 0)[0]
        n_negative_correct += (pred[negative_indices] == labels[negative_indices]).sum().item()
        n_negative_samples += len(negative_indices)

        c += pred.shape[0]

    accuracy = n_correct / len(dataset)
    positive_accuracy = n_positive_correct / n_positive_samples
    negative_accuracy = n_negative_correct / n_negative_samples

    if ret_pos_and_neg_acc:
        return predictions, accuracy, positive_accuracy, negative_accuracy
    else:
        return predictions, accuracy


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
def get_label_probability(images, model, standardize=True, to_grayscale=False, n_output_classes=2, device='cuda'):
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
    predictions = torch.empty((len(image_dataset), n_output_classes), dtype=torch.float32)
    for images in loader:
        images = images.to(device)
        pred = model(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        predictions[c:c + len(pred), ...] = pred
        c += pred.shape[0]

    return predictions


def create_probability_map(patches, model, im_shape, mask=None, standardize=True, to_grayscale=False,
                           device='cuda'):
    if mask is None:
        mask = np.ones(im_shape, dtype=np.bool8)

    mask_indices = np.where(mask.flatten())[0]
    assert len(mask_indices) == len(patches), 'Number of patches must match the number of pixles in mask'

    model = model.to(device)
    model = model.eval()

    label_probabilities = get_label_probability(patches, model, standardize=standardize,
                                                to_grayscale=to_grayscale, device=device)

    probability_map = np.zeros(im_shape, dtype=np.float32)
    rows, cols = np.where(mask)
    probability_map[rows, cols] = label_probabilities[:, 1]

    return probability_map


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
