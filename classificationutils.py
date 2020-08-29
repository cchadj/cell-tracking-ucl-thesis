from typing import Tuple, Dict, Any

import numpy as np
from torch.utils.data import Dataset

from evaluation import evaluate_results, EvaluationResults
from imageprosessing import imextendedmax
from matplotlib import pyplot as plt

import torch
from learningutils import ImageDataset

from patchextraction import SessionPatchExtractor
from video_session import VideoSession

NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


class ClassificationResults:
    def __init__(self,
                 positive_accuracy, negative_accuracy, accuracy, n_positive, n_negative,
                 loss=None, predictions=None, output_probabilities=None, dataset=None, model=None):
        self.model = model
        self.dataset = dataset

        self.loss = loss

        self.predictions = predictions
        self.output_probabilities = output_probabilities

        self.accuracy = accuracy

        self.positive_accuracy = positive_accuracy
        self.negative_accuracy = negative_accuracy
        self.n_positive = n_positive
        self.n_negative = n_negative


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

    n_positive_correct = 0
    n_negative_correct = 0
    n_positive_samples = 0
    n_negative_samples = 0

    c = 0
    predictions = torch.empty(len(dataset), dtype=torch.long).to(device)
    output_probabilities = torch.empty((len(dataset), 2), dtype=torch.float32).to(device)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        output_probabilities[c:c + len(pred)] = pred

        pred = torch.argmax(pred, dim=1)
        predictions[c:c + len(pred)] = pred

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

    return ClassificationResults(
        model=model,
        dataset=dataset,
        n_positive=n_positive_samples,
        n_negative=n_negative_samples,

        predictions=predictions,
        output_probabilities=output_probabilities,

        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        accuracy=accuracy,
    )


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


def get_cell_positions_from_probability_map(
        probability_map,
        extended_maxima_h,
        sigma=1,
        visualise_intermediate_results=False):
    assert 0.1 <= extended_maxima_h <= 0.9, f'Extended maxima h must be between .1 and .9 not {extended_maxima_h}'
    from skimage.filters import gaussian
    from skimage import measure
    from scipy.ndimage import center_of_mass

    pm_blurred = gaussian(probability_map, sigma)
    pm_extended_max_bw = imextendedmax(pm_blurred, extended_maxima_h)

    labeled_img, nr_objects = measure.label(pm_extended_max_bw, return_num=True)

    # print(np.where(pm_extended_max_bw)[0])
    pm_extended_max = probability_map.copy()
    pm_extended_max[pm_extended_max_bw] = 0

    # print(pm_extended_max)
    # Notice, the positions from the csv is x,y. The result from the probability is y,x so we swap.
    region_props = measure.regionprops(labeled_img, intensity_image=pm_blurred)
    estimated_cell_positions = np.empty((len(region_props), 2))

    for i, region in enumerate(region_props):
        y, x = region.weighted_centroid
        estimated_cell_positions[i] = x, y

    if visualise_intermediate_results:
        fig, axes = plt.subplots(1, 3)
        fig_size = fig.get_size_inches()
        fig.set_size_inches((fig_size[0] * 5,
                             fig_size[1] * 5))

        axes[0].imshow(probability_map)
        axes[0].set_title('Unprocessed probability map')

        axes[1].imshow(pm_blurred)
        axes[1].set_title(f'Gaussian Blurring with sigma={sigma}')

        axes[2].imshow(pm_extended_max_bw)
        axes[2].set_title(f'Extended maximum, H={extended_maxima_h}')
        axes[2].scatter(estimated_cell_positions[:, 0], estimated_cell_positions[:, 1], s=4,
                        label='estimated locations')
        axes[2].legend()

    return estimated_cell_positions[1:, ...]


@torch.no_grad()
def get_label_probability(images, model, standardize=True, to_grayscale=False, n_output_classes=2, device='cuda'):
    """ Make a prediction for the images giving output_probabilities for each labels.

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


class MutualExclusiveArgumentsException(Exception):
    pass


class SessionClassifier:
    model: torch.nn.Module
    patch_extractor: SessionPatchExtractor
    session: VideoSession
    probability_maps: Dict[int, np.ndarray]
    estimated_locations: Dict[int, np.ndarray]
    evaluation_results: Dict[int, EvaluationResults]

    def __init__(self, video_session, model,
                 mixed_channels=False,
                 patch_size=21,
                 temporal_width=0,
                 standardise=True,
                 to_grayscale=False,

                 n_negatives_per_positive=15,
                 negative_extraction_mode=SessionPatchExtractor.CIRCLE,
                 ):
        from copy import deepcopy

        self.model = deepcopy(model)
        self.model = self.model.eval()

        self.session = video_session

        self.standardise = standardise
        self.to_grayscale = to_grayscale

        self._mixed_channels = False
        self._temporal_width = 0

        self.mixed_channels = mixed_channels
        self.temporal_width = temporal_width

        self.evaluation_results = {}
        self.probability_maps = {}
        self.estimated_locations = {}

        self.patch_size = patch_size
        self.patch_extractor = SessionPatchExtractor(
            self.session,
            patch_size=patch_size,
            temporal_width=temporal_width,
            extraction_mode=SessionPatchExtractor.ALL_MODE,

            n_negatives_per_positive=n_negatives_per_positive,
            negative_extraction_mode=negative_extraction_mode
        )

    def classify_cells(self, frame_idx=None):
        from cnnlearning import LabeledImageDataset

        if frame_idx is None:
            if self.mixed_channels:
                cell_patches = self.patch_extractor.mixed_channel_cell_patches
                non_cell_patches = self.patch_extractor.mixed_channel_non_cell_patches
            elif self.temporal_width > 0:
                cell_patches = self.patch_extractor.temporal_cell_patches_oa790
                non_cell_patches = self.patch_extractor.temporal_non_cell_patches_oa790
            else:
                cell_patches = self.patch_extractor.cell_patches_oa790
                non_cell_patches = self.patch_extractor.non_cell_patches_oa790
        else:
            if self.mixed_channels:
                cell_patches = self.patch_extractor.mixed_channel_cell_patches_at_frame[frame_idx]
                non_cell_patches = self.patch_extractor.mixed_channel_non_cell_patches_at_frame[frame_idx]
            elif self.temporal_width > 0:
                cell_patches = self.patch_extractor.temporal_cell_patches_oa790_at_frame[frame_idx]
                non_cell_patches = self.patch_extractor.temporal_non_cell_patches_oa790_at_frame[frame_idx]
            else:
                cell_patches = self.patch_extractor.cell_patches_oa790_at_frame[frame_idx]
                non_cell_patches = self.patch_extractor.non_cell_patches_oa790_at_frame[frame_idx]

        dataset = LabeledImageDataset(
            np.concatenate((cell_patches,                               non_cell_patches), axis=0),
            np.concatenate((np.ones(len(cell_patches), dtype=np.int32), np.zeros(len(non_cell_patches), dtype=np.int32))),
            standardize=self.standardise, to_grayscale=self.to_grayscale, data_augmentation_transforms=None,
        )
        return classify_labeled_dataset(dataset, model=self.model)

    def estimate_locations(self, frame_idx, use_frame_mask=True, use_vessel_mask=True, mask=None,
                           extended_maxima_h=0.5, sigma=1.2):
        if mask is None:
            mask = np.ones(self.session.frames_oa790.shape[1:3], dtype=np.bool8)

        if use_vessel_mask:
            mask &= self.session.vessel_mask_oa790

        if use_frame_mask:
            mask &= self.session.mask_frames_oa790[frame_idx]

        if self.mixed_channels:
            patches = self.patch_extractor.mixed_channel_cell_patches(frame_idx, mask=mask)
        else:
            patches = self.patch_extractor.all_patches_oa790(frame_idx, mask=mask)

        probability_map = create_probability_map(patches, self.model, im_shape=mask.shape, mask=mask,
                                                 standardize=self.standardise, to_grayscale=self.to_grayscale)

        estimated_positions = get_cell_positions_from_probability_map(probability_map,
                                                                      extended_maxima_h=extended_maxima_h,
                                                                      sigma=sigma)

        if self.session.is_marked and frame_idx in self.session.cell_positions:
            sigmas = np.arange(0.2, 2, step=.1)
            extended_maxima_hs = np.arange(0.1, 0.8, step=.1)

            dice_coefficients = np.zeros((len(sigmas), len(extended_maxima_hs)))
            for i, s in enumerate(sigmas):
                for j, h in enumerate(extended_maxima_hs):
                    estimated_positions = get_cell_positions_from_probability_map(probability_map, extended_maxima_h=h,
                                                                                  sigma=s)
                    evaluation_results = evaluate_results(ground_truth_positions=self.session.cell_positions[frame_idx],
                                                          estimated_positions=estimated_positions,
                                                          image=self.session.frames_oa790[frame_idx],
                                                          mask=mask,
                                                          patch_size=self.patch_size)
                    dice_coefficients[i, j] = evaluation_results.dice

            max_idx = np.argmax(dice_coefficients)
            sigma_idx, h_idx = np.unravel_index(max_idx, dice_coefficients.shape)

            best_sigma = sigmas[sigma_idx]
            best_h = extended_maxima_hs[h_idx]
            estimated_positions = get_cell_positions_from_probability_map(
                probability_map, extended_maxima_h=best_h, sigma=best_sigma
            )

            self.evaluation_results[frame_idx] = evaluate_results(
                self.session.cell_positions[self.session.validation_frame_idx],
                estimated_positions=estimated_positions,
                image=self.session.frames_oa790[
                    self.session.validation_frame_idx],
                mask=mask,
                patch_size=21)

        self.estimated_locations[frame_idx] = estimated_positions
        self.probability_maps[frame_idx] = probability_map
        return estimated_positions

    @property
    def temporal_width(self):
        return self._temporal_width

    @temporal_width.setter
    def temporal_width(self, width):
        if width > 0 and self.mixed_channels:
            raise MutualExclusiveArgumentsException('Temporal width > 0 can not work with mixed channels.'
                                                    'Set mixed channel to False first.')

    @property
    def mixed_channels(self):
        return self._mixed_channels

    @mixed_channels.setter
    def mixed_channels(self, mixed_channel_extraction):
        if self.temporal_width > 0 and mixed_channel_extraction:
            raise MutualExclusiveArgumentsException(
                'Mixed channel extraction can not work with temporal width greater than 0.'
                'Set temporal width to 0 first.')
        self._mixed_channels = mixed_channel_extraction


if __name__ == '__main__':
    from generate_datasets import get_cell_and_no_cell_patches
