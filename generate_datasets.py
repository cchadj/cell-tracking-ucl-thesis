import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2

import matplotlib
from tqdm.contrib import tzip

from imageprosessing import hist_match_images
from learningutils import LabeledImageDataset
from sharedvariables import *
from patchextraction import extract_patches_at_positions


def get_frames_from_video(video_filename, normalise=False):
    """
    Get the frames of a video as an array.

    Arguments:
        video_filename: The path of the video.
        normalise: Normalise frame values between 0 and 1.
            If normalised, the return numpy array type is float32 instead of uint8.

    Returns:
        frames as a NxHxWxC array (Number of frames x Height x Width x Channels)
    """
    vidcap = cv2.VideoCapture(video_filename)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = vidcap.read()

    frame_type = np.uint8
    if normalise:
        frame_type = np.float32
    frames = np.zeros([n_frames] + list(image.shape), dtype=frame_type)

    count = 0
    while success:
        if normalise:
            image = image.astype(frame_type) / 255
        frames[count, ...] = image
        success, image = vidcap.read()
        count += 1

    return frames


def get_random_point_on_rectangle(cx, cy, rect_size):
    """

    Args:
        cx: Rectangle center x component.
        cy: Rectangle center y component.
        rect_size (tuple): Rectangle height, width.

    Returns:
        Random points on the rectangles defined by centre cx, cy and height, width

    """
    height, width = rect_size
    if isinstance(cx, int):
        cx, cy = np.array([cx]), np.array([cy])

    assert len(cx) == len(cy)

    # random number to select edge along which to get the random point (up/down, left/right)
    # (0 or 1 displacement)
    r1 = np.random.rand(len(cx))

    # random  number for swapping displacements
    r2 = np.random.rand(len(cy))

    # Controls 0 or 1 displacement
    t = np.zeros(len(cx))
    t[r1 > 0.5] = 1

    dx = t * width
    dy = np.random.rand(len(cx)) * width

    # print('dx', dx)
    # print('dy', dy)

    dx[r2 > 0.5], dy[r2 > 0.5] = dy[r2 > 0.5], dx[r2 > 0.5]

    # print("r1", r1)
    # print("r2", r2)
    # print("t", t)
    #
    # print('dx', dx)
    # print('dy', dy)

    # if r2 > 0.5:
    #     dy, dx = dx, dy

    rx = (cx - width / 2) + dx
    ry = (cy - height / 2) + dy

    return rx, ry


def get_cell_and_no_cell_patches_from_video(video_filename,
                                            csv_filename,
                                            patch_size=(21, 21),
                                            padding='valid',
                                            n_negatives_per_positive=1,
                                            normalise=True,
                                            visualize_patches=False,
                                            ):
    """ Get the cell and non cell patches from video.

    TODO: PADDING DOES NOT CURRENTLY WORK
    TODO: PATCH SIZE MUST HAVE EQUAL DIMENSIONS (Height == Width)

    Args:
        video_filename (str): The filename of the video.
        csv_filename (str): The filename of the csv file with the cell position centres.
        patch_size (tuple): The height, width of the patches to be extracted.
        n_negatives_per_positive (int):  Number of negative patches to extract per positive patch
        padding:
            CURRENTLY NOT IMPLEMENTED
            'valid' If you want only patches that are entirely inside the image.
            If not valid then one of : [
                cv2.BORDER_REPLICATE,
                cv2.BORDER_REFLECT,
                cv2.BORDER_WRAP,
                cv2.BORDER_ISOLATED
            ]

    Returns:
       tuple: Cell and no cell patches, both with NxHxWxC shape.
    """
    patch_height, patch_width = patch_size

    csv_cell_positions_df = pd.read_csv(csv_filename, delimiter=',')

    csv_cell_positions_coordinates = np.int32(csv_cell_positions_df[['X', 'Y']].to_numpy())
    csv_cell_positions_frame_indices = np.int32(csv_cell_positions_df[['Slice']].to_numpy())
    n_cells_in_vid = len(csv_cell_positions_coordinates)

    frame_indices = np.unique(csv_cell_positions_frame_indices)

    # Number of cells in videos is the same as the number of entries in the csv_file
    cell_positions = {}
    for frame_idx in frame_indices:
        curr_coordinates = csv_cell_positions_coordinates[np.where(csv_cell_positions_frame_indices == frame_idx)[0]]
        cell_positions[frame_idx] = curr_coordinates

    frames = get_frames_from_video(video_filename, normalise)

    cell_patches = np.zeros_like(frames, shape=[n_cells_in_vid, *patch_size])
    non_cell_patches = np.zeros_like(frames, shape=[n_negatives_per_positive * n_cells_in_vid, *patch_size])

    cell_count = 0
    non_cell_count = 0
    for i, frame_idx in enumerate(frame_indices):
        # Slices in the csv file start from 1, we - 1 to match python 0 indexing
        frame = frames[int(frame_idx) - 1, ...]
        curr_frame_cell_positions = cell_positions[frame_idx].astype(np.int)

        cell_xs, cell_ys = curr_frame_cell_positions[:, 0], curr_frame_cell_positions[:, 1]

        curr_frame_non_cell_positions = np.empty([0, 2], dtype=np.int32)
        for _ in range(n_negatives_per_positive):
            rxs, rys = get_random_point_on_rectangle(cell_xs, cell_ys, patch_size)
            curr_frame_non_cell_positions = np.concatenate(
                (curr_frame_non_cell_positions, np.array([rys, rxs]).T), axis=0)
        # print(curr_frame_cell_positions.shape)
        # print(curr_frame_non_cell_positions.shape)

        if padding is not 'valid':
            raise NotImplementedError("Padding other than 'valid', not implemented")
        #     padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)
        #
        #     rxs, rys = rxs + padding_width, rys + padding_height
        #     cell_xs, cell_ys = cell_xs + padding_width, cell_ys + padding_height
        # curr_frame_non_cell_positions = np.array([rys, rxs]).T

        curr_frame_cell_patches = extract_patches_at_positions(frame, curr_frame_cell_positions,
                                                               patch_size, padding)[..., 0]
        curr_frame_non_cell_patches = extract_patches_at_positions(frame, curr_frame_non_cell_positions,
                                                                   patch_size, padding)[..., 0]

        cell_patches[cell_count:cell_count + len(curr_frame_cell_patches), ...] = curr_frame_cell_patches
        non_cell_patches[non_cell_count:non_cell_count + len(curr_frame_non_cell_patches), ...] = curr_frame_non_cell_patches

        cell_count += len(curr_frame_cell_patches)
        non_cell_count += len(curr_frame_non_cell_patches)

        if visualize_patches and frame_idx == 1:
            padding_height, padding_width = 0, 0
            if padding is not 'valid':
                padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)
                frame = cv2.copyMakeBorder(frame,
                                           padding_height,
                                           padding_height,
                                           padding_width,
                                           padding_width,
                                           padding)

            fig, ax = plt.subplots(1, figsize=(20, 10))
            ax.imshow(frame)

            rxs, rys = get_random_point_on_rectangle(cell_xs, cell_ys, patch_size)
            curr_frame_non_cell_positions = np.array([rxs, rys]).T

            xs, ys = curr_frame_cell_positions[:, 0], curr_frame_cell_positions[:, 1]
            rxs, rys = curr_frame_non_cell_positions[:, 0], curr_frame_non_cell_positions[:, 1]

            ax.scatter(xs, ys, c='#0000FF', label='Cell positions')
            ax.scatter(rxs, rys, c='#FF0000', label='Non cell positions')
            ax.legend()
            for frame_pos, non_frame_pos in zip(curr_frame_cell_positions.astype(np.int),
                                                curr_frame_non_cell_positions.astype(np.int)):
                x, y = frame_pos
                rx, ry = non_frame_pos

                x, y = x + padding_width, y + padding_height
                rx, ry = rx + padding_width, ry + padding_height

                rect = matplotlib.patches.Rectangle((x - patch_width / 2, y - patch_height / 2),
                                                    patch_width, patch_height, linewidth=1,
                                                    edgecolor='b', facecolor='none')

                # Create a Rectangle patch
                rect2 = matplotlib.patches.Rectangle((rx - patch_width / 2, ry - patch_height / 2),
                                                     patch_width, patch_height, linewidth=1,
                                                     edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
                ax.add_patch(rect2)
                # ax.scatter(rx, ry)

    # Cells whose bounding box is gets out of image are still zeros, remove them for now (TODO handle them)
    cell_patches = cell_patches[:cell_count, ...]
    non_cell_patches = non_cell_patches[:non_cell_count, ...]

    idxs_to_delete = []
    for i in range(non_cell_patches.shape[0]):
        if np.all(np.equal(non_cell_patches[i, ...], np.zeros([patch_height, patch_width]))):
            idxs_to_delete.append(i)

    non_cell_patches = np.delete(non_cell_patches, idxs_to_delete, axis=0)
    return cell_patches, non_cell_patches


def get_cell_and_no_cell_patches(patch_size=(21, 21), n_negatives_per_positive=3, do_hist_match=False):
    # Input
    height, width = patch_size
    print(f'patch size {(height, width)}')
    print(f'do hist match: {do_hist_match}')
    print(f'Negatives per positive {n_negatives_per_positive}')
    print()

    patch_size = (height, width)

    pathlib.Path(CACHED_DATASETS_FOLDER).mkdir(parents=True, exist_ok=True)

    trainset_filename = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'trainset_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.pt')
    validset_filename = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'validset_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.pt')

    cell_images_filename = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.npy')
    non_cell_images_filename = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'non_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.npy')
    cell_images_marked_filename = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}_marked.npy')
    non_cell_images_marked_filename = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'non_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}_marked.npy')

    try:
        print('Dataset loading from cache')
        print('--------------------------')
        print(f"loading training set from '{trainset_filename}'...")
        trainset = torch.load(trainset_filename)
        print(f"loading validation set from '{validset_filename}'...")
        validset = torch.load(validset_filename)

        #
        print(f"loading bloodcell patches from '{cell_images_filename}'...")
        cell_images = np.load(cell_images_filename)
        print(f"loading non bloodcell patches from '{non_cell_images_filename}'...")
        non_cell_images = np.load(non_cell_images_filename)

        print(f"loading marked bloodcell patches from '{cell_images_marked_filename}'...")
        cell_images_marked = np.load(cell_images_marked_filename)
        print(f"loading marked non bloodcell patches from '{non_cell_images_marked_filename}'")
        non_cell_images_marked = np.load(non_cell_images_marked_filename)

        print('Done')
        print()
    except FileNotFoundError:
        print('Not all data found fom cache. Creating datasets...')
        cell_images = np.zeros([0, *patch_size], dtype=np.float32)
        cell_images_marked = np.zeros_like(cell_images)

        non_cell_images = np.zeros_like(cell_images)
        non_cell_images_marked = np.zeros_like(cell_images)

        print('Creating cell and no cell images from videos...')
        for video_file, csv_file in tzip(unmarked_video_OA790_filenames, csv_OA790_filenames):
            print(video_file, csv_file, sep='<->\n')
            curr_cell_images, curr_non_cell_images = get_cell_and_no_cell_patches_from_video(
                video_file, csv_file,
                patch_size=patch_size,
                n_negatives_per_positive=n_negatives_per_positive,
                normalise=True)

            cell_images = np.concatenate((cell_images, curr_cell_images), axis=0).astype(np.float32)
            non_cell_images = np.concatenate((non_cell_images, curr_non_cell_images), axis=0).astype(np.float32)

        hist_match_template = cell_images[0]
        if do_hist_match:
            cell_images = hist_match_images(cell_images, hist_match_template)
            non_cell_images = hist_match_images(non_cell_images, hist_match_template)
            cell_images_marked = hist_match_images(cell_images_marked, hist_match_template)

        print('Creating cell patches from marked videos for debugging...')
        for video_file, csv_file in tzip(marked_video_OA790_filenames, csv_OA790_filenames):
            print(video_file, csv_file, sep='<->\n')
            curr_cell_images_marked, curr_non_cell_images_marked = get_cell_and_no_cell_patches_from_video(
                video_file, csv_file,
                patch_size=patch_size,
                n_negatives_per_positive=n_negatives_per_positive,
                normalise=True)

            cell_images_marked = np.concatenate((cell_images_marked, curr_cell_images_marked),
                                                axis=0).astype(np.float32)
            non_cell_images_marked = np.concatenate((non_cell_images_marked, curr_non_cell_images_marked),
                                                    axis=0).astype(np.float32)
        print()

        print('Creating dataset from cell and non cell patches')
        print('-----------------------------------------------')

        dataset = LabeledImageDataset(
            np.concatenate((cell_images[:len(cell_images), ...],      non_cell_images[:len(non_cell_images), ...]),   axis=0),
            np.concatenate((np.ones(len(cell_images)).astype(np.int), np.zeros(len(non_cell_images)).astype(np.int)), axis=0)
        )
        print('Splitting into training set and validation set')
        trainset_size = int(len(dataset) * 0.80)
        validset_size = len(dataset) - trainset_size
        trainset, validset = torch.utils.data.random_split(dataset, (trainset_size, validset_size))
        print()

        print('Saving datasets')
        print('---------------')
        torch.save(trainset, os.path.join(trainset_filename))
        torch.save(validset, os.path.join(validset_filename))
        print(f"Saved training set as: '{trainset_filename}'")
        print(f"Saved validation set as: '{validset_filename}'")

        print('Saving cell and non cell images')
        np.save(cell_images_filename, cell_images)
        print(f"Saved cell images as: '{cell_images_filename}'")
        np.save(non_cell_images_filename, non_cell_images)
        print(f"Saved non cell images as: '{non_cell_images_filename}'")
        np.save(cell_images_marked_filename, cell_images_marked)
        print(f"Saved marked cell images (for debugging) as: '{cell_images_marked_filename}'")
        np.save(non_cell_images_marked_filename, non_cell_images_marked)
        print(f"Saved marked non cell images (for debugging) as: '{non_cell_images_marked_filename}'")

    print("Cell images:", cell_images.shape)
    print("Non cell images", non_cell_images.shape)

    hist_match_template = cell_images[0]
    return trainset, validset, cell_images, non_cell_images, cell_images_marked, non_cell_images_marked, hist_match_template


def main():
    get_cell_and_no_cell_patches()
    pass


if __name__ == '__main__':
    main()

