import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2

from IPython import display
from evaluation import get_cell_positions_from_probability_map, evaluate_results
from learningutils import ImageDataset, LabeledImageDataset
from cnnlearning import CNN, train
import collections
from patchextraction import extract_patches, get_patch, extract_patches_at_positions
from sharedvariables import *
from classificationutils import classify_images, classify_labeled_dataset, create_probability_map
import guitools


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


def get_number_of_cells_in_csv(csv_filename):
    csv_cell_positions = np.genfromtxt(csv_filename, delimiter=',')
    n_cells = csv_cell_positions.shape[0] - 1  # First row are the labels
    return n_cells


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


def plot_images_as_grid(images, title=None):
    """
    Plots a stack of images in a grid.

    Arguments:
        images: The images as NxHxWxC
        title: Plot title
    """
    if len(images.shape) == 3:
        images = images[..., np.newaxis]

    batch_tensor = torch.from_numpy(images)
    # NxHxWxC -> NxCxHxW
    batch_tensor = batch_tensor.permute(0, -1, 1, 2)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=50)

    plt.figure(num=None, figsize=(70, 50), dpi=80, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.grid(b=None)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


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
                                            normalise=True):
    """ Get the cell and non cell patches from video.

    TODO: PADDING DOES NOT CURRENTLY WORK
    TODO: PATCH SIZE MUST HAVE EQUAL DIMENSIONS (Height == Width)

    Args:
        video_filename (str): The filename of the video.
        csv_filename (str): The filename of the csv file with the cell position centres.
        patch_size (tuple): The height, width of the patches to be extracted.
        padding:
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
    csv_cell_positions = np.genfromtxt(csv_filename, delimiter=',')

    # Remove index and column labels
    csv_cell_positions = csv_cell_positions[1:, 1:]

    # Sort array based on slice index
    csv_cell_positions = csv_cell_positions[np.argsort(csv_cell_positions[:, -1])]

    # Get the slices and the index where the slice was first found
    [frame_idxs, idxs] = np.unique(csv_cell_positions[:, -1], return_index=True)
    n_cells_in_vid = csv_cell_positions.shape[0]

    # Number of cells in videos is the same as the number of entries in the csv_file

    # cell_positions[frame_idx] will contain 2xN_cells_in_frame array for each frame
    cell_positions = {}
    for i in range(len(frame_idxs)):
        n_cells_in_frame = 0
        curr_idx = idxs[i]
        frame_idx = frame_idxs[i]

        if i == len(frame_idxs) - 1:
            cell_positions[frame_idx] = (csv_cell_positions[curr_idx:-1, :-1])
            # n_cells_in_frame = (csv_cell_positions.shape[0]) - curr_idx
        else:
            cell_positions[frame_idx] = (csv_cell_positions[curr_idx:idxs[i + 1], :-1])
            # n_cells_in_frame = idxs[i + 1] - curr_idx + 1

    frames = get_frames_from_video(video_filename, normalise)

    cell_patches = np.zeros_like(frames, shape=[n_cells_in_vid, *patch_size])
    non_cell_patches = np.zeros_like(frames, shape=[n_cells_in_vid, *patch_size])

    cell_count = 0
    non_cell_count = 0
    for i, frame_idx in enumerate(frame_idxs):
        # Slices in the csv file start from 1, we - 1 to match python 0 indexing
        frame = frames[int(frame_idx) - 1, ...]
        curr_frame_positions = cell_positions[frame_idx].astype(np.int)

        cell_xs, cell_ys = curr_frame_positions[:, 0], curr_frame_positions[:, 1]
        rxs, rys = get_random_point_on_rectangle(cell_xs, cell_ys, patch_size)

        if padding is not 'valid':
            padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)

            rxs, rys = rxs + padding_width, rys + padding_height
            cell_xs, cell_ys = cell_xs + padding_width, cell_ys + padding_height

        curr_frame_non_cell_positions = np.array([rys, rxs]).T

        curr_frame_cell_patches = extract_patches_at_positions(frame, curr_frame_positions,
                                                               patch_size, padding)[..., 0]
        curr_frame_non_cell_patches = extract_patches_at_positions(frame, curr_frame_non_cell_positions,
                                                                   patch_size, padding)[..., 0]

        cell_patches[cell_count:cell_count + len(curr_frame_cell_patches), ...] = curr_frame_cell_patches
        non_cell_patches[non_cell_count:non_cell_count + len(curr_frame_non_cell_patches), ...] = curr_frame_non_cell_patches

        cell_count += len(curr_frame_cell_patches)
        non_cell_count += len(curr_frame_non_cell_patches)

        if frame_idx == 1:
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

            xs, ys = curr_frame_positions[:, 0], curr_frame_positions[:, 1]
            rxs, rys = curr_frame_non_cell_positions[:, 0], curr_frame_non_cell_positions[:, 1]

            ax.scatter(xs, ys, c='#0000FF', label='Cell positions')
            ax.scatter(rxs, rys, c='#FF0000', label='Non cell positions')
            ax.legend()
            for frame_pos, non_frame_pos in zip(curr_frame_positions.astype(np.int),
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # parser = argparse.ArgumentParser(description='Get positions on image')
    # parser.add_argument('image', type=str, help='The image to predict cell positions.')
    # parser.add_argument('--patch-size', type=int, default=(19, 19), help='The patch size.')
    # args = parser.parse_args()
    #
    # patch_size = args.patch_size, args.patch_size
    # patch_size = (19, 19)
    patch_size = (21, 21)

    trainset_filename = os.path.join(CACHED_DATASETS_FOLDER, f'trainset_bloodcells_ps_{patch_size}.pt')
    validset_filename = os.path.join(CACHED_DATASETS_FOLDER, f'validset_bloodcells_ps_{patch_size}.pt')

    # Create dataset or load dataset from cache
    print('Dataset creation')
    print('----------------')
    try:
        print(f"loading training set from '{trainset_filename}'...")
        print(f"loading validation set from '{validset_filename}'...")
        trainset = torch.load(trainset_filename)
        validset = torch.load(validset_filename)
        print('Done')
    except FileNotFoundError:
        print('Datasets not found. Creating datasets...')

        cell_images = np.zeros([0, *patch_size], dtype=np.float32)
        non_cell_images = np.zeros([0, *patch_size], dtype=np.float32)

        for video_file, csv_file in zip(unmarked_video_OA790_filenames, csv_OA790_filenames):
            curr_cell_images, curr_non_cell_images = get_cell_and_no_cell_patches_from_video(video_file,
                                                                                             csv_file,
                                                                                             patch_size=patch_size,
                                                                                             normalise=True)

            cell_images = np.concatenate((cell_images, curr_cell_images), axis=0).astype(np.float32)
            non_cell_images = np.concatenate((non_cell_images, curr_non_cell_images), axis=0).astype(np.float32)

        print("Cell images:", cell_images.shape)
        print("Non cell images", non_cell_images.shape)

        fig, axes = plt.subplots(2, 6, figsize=(20, 10))
        axes[0, 0].imshow(cell_images[0], cmap='gray')
        axes[0, 1].imshow(cell_images[1], cmap='gray')
        axes[0, 2].imshow(cell_images[2], cmap='gray')
        axes[0, 3].imshow(cell_images[3], cmap='gray')
        axes[0, 4].imshow(cell_images[4], cmap='gray')
        axes[0, 5].imshow(cell_images[5], cmap='gray')

        axes[1, 0].imshow(non_cell_images[0], cmap='gray')
        axes[1, 1].imshow(non_cell_images[1], cmap='gray')
        axes[1, 2].imshow(non_cell_images[2], cmap='gray')
        axes[1, 3].imshow(non_cell_images[3], cmap='gray')
        axes[1, 4].imshow(non_cell_images[4], cmap='gray')
        axes[1, 5].imshow(non_cell_images[5], cmap='gray')
        plt.show()

        dataset_size = cell_images.shape[0]

        dataset = LabeledImageDataset(
            np.concatenate((cell_images[:dataset_size, ...], non_cell_images[:dataset_size, ...]), axis=0),
            np.concatenate((np.ones(dataset_size).astype(np.int), np.zeros(dataset_size).astype(np.int)), axis=0)
        )

        trainset_size = int(len(dataset) * 0.80)
        validset_size = len(dataset) - trainset_size

        print('Saving datasets...')
        trainset, validset = torch.utils.data.random_split(dataset, (trainset_size, validset_size))
        torch.save(trainset, os.path.join(trainset_filename))
        torch.save(validset, os.path.join(validset_filename))
        print(f"Saved training set as: '{trainset_filename}'")
        print(f"Saved validation set as: '{validset_filename}'")
        print('Done')
    print()

    # Train a classifier or load from cache
    print('Classifier training or load from cache')
    print('--------------------------------------')
    model = CNN(convolutional=
                nn.Sequential(
                    nn.Conv2d(1, 32, padding=2, kernel_size=5),
                    # PrintLayer("1"),
                    nn.BatchNorm2d(32),
                    # PrintLayer("2"),
                    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                    # PrintLayer("3"),

                    nn.Conv2d(32, 32, padding=2, kernel_size=5),
                    # PrintLayer("4"),
                    nn.BatchNorm2d(32),
                    # PrintLayer("5"),
                    nn.ReLU(),
                    # PrintLayer("6"),
                    nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
                    # PrintLayer("7"),

                    nn.Conv2d(32, 64, padding=2, kernel_size=5),
                    #PrintLayer("9"),
                    nn.BatchNorm2d(64),
                    #PrintLayer("11"),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
                    # PrintLayer("12"),
                ),
                dense=
                nn.Sequential(
                    nn.Linear(576, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(64),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 2),
                )).to(device)

    try:
        if patch_size == (21, 21):
            input_model_file = os.path.join(CACHED_MODELS_FOLDER, 'blood_cell_classifier_(21, 21)_va_0.9528585757271816.pt')
        elif patch_size == (19, 19):
            input_model_file = os.path.join(CACHED_MODELS_FOLDER, 'blood_cell_classifier_(19, 19)_va_0.9481037924151696.pt')

        print(f'Loading model from {input_model_file}')
        model.load_state_dict(torch.load(input_model_file))
        print('Done. You can see the run data at the same folder.')
    except FileNotFoundError:
        pass
        print('Model not found. Training new model. You can interrupt(ctr - C or interrupt kernel) any time to get'
              'the model with the best validation accuracy at the current time.')

        train_params = collections.OrderedDict(
            # lr = .001,
            #optimizer=torch.optim.SGD(model.parameters(), lr=.001, weight_decay=5e-5, momentum=0.9),
            optimizer=torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4),
            batch_size=1024 * 16,
            do_early_stop=True,# Optional default True
            early_stop_patience=80,
            learning_rate_scheduler_patience=100,
            epochs=2000,
            shuffle=True,
            # valid_untrunsformed_normals = valid_untrunsformed_normals,
            trainset=trainset,
            validset=validset,
        )

        results = train(model, train_params,  criterion=torch.nn.CrossEntropyLoss(), device=device)

        output_name = os.path.join(CACHED_MODELS_FOLDER,
                                   f'blood_cell_classifier_{patch_size}_va_{results.recorded_model_valid_accuracy}')
        print(f'Saving model as {output_name}')
        results.save(output_name)
        print('Done')
        model = results.recorded_model

    # Print the final validation accuracy on the training and validation set
    model.eval()
    trainset_predictions, train_accuracy = classify_labeled_dataset(trainset, model)
    validset_predictions, valid_accuracy = classify_labeled_dataset(validset, model)

    print()
    print('Brief evaluation')
    print('----------------')
    print('Training accuracy\t', train_accuracy)
    print('Validation accuracy\t', valid_accuracy)

    # Evaluate on a sample frame
    video_file = unmarked_video_OA790_filenames[0]
    csv_file = csv_OA790_filenames[0]

    frames = get_frames_from_video(video_file)[..., 0]
    all_video_cell_positions = np.genfromtxt(csv_file, delimiter=',')[1:, 1:]
    first_frame_position_indices = np.where(all_video_cell_positions[..., 2] == 1)[0]
    ground_truth_positions = all_video_cell_positions[first_frame_position_indices, :2]
    sample_frame = frames[0]

    print('Frames shape', frames.shape)
    print('Sample frame', sample_frame.shape)
    print('Blood cell positions for all frames', all_video_cell_positions.shape)
    print()

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(sample_frame, cmap='gray')
    axes[0].scatter(ground_truth_positions[:, 0],
                    ground_truth_positions[:, 1],
                    label='Manual positions')
    axes[0].set_title('Sample frame')

    fig_size = fig.get_size_inches()
    fig.set_size_inches((fig_size[0] * 3,
                         fig_size[1] * 3))

    probability_map = create_probability_map(sample_frame,
                                             model,
                                             patch_size=(19, 19),
                                             device=device)

    axes[1].imshow(probability_map)
    axes[1].set_title('Probability Map')
    fig.canvas.set_window_title('Blood cell position estimation')
    plt.legend()
    plt.show()

    # Find sigma and h for maximum dice coefficeint or load cached
    filename = os.path.join(CACHE_FOLDER, 'blood_cell_dices_coefficient_sigma_h.csv')
    try:
        print(f'Loading dices coefficient and correspoding sigmas and hs from:\n {filename}')
        df = pd.read_csv(filename, usecols=(1, 2, 3))
        print('Done')
    except FileNotFoundError:
        print(f"File not found. Finding sigma and h that maximise Dice's coefficient...")
        dices_coefficients = []
        sigmas = []
        Hs = []

        gauss_sigmas = np.arange(1, 9, 0.2)
        extended_maxima_Hs = np.arange(0.05, 0.9, 0.025)
        from tqdm import tqdm

        max_dices_coeff = 0
        best_sigma = gauss_sigmas[0]
        best_H = extended_maxima_Hs[0]
        for s in tqdm(gauss_sigmas):
            for h in extended_maxima_Hs:
                estimated_positions = get_cell_positions_from_probability_map(probability_map, s, h)

                dices_coeff, _, _ = evaluate_results(ground_truth_positions,
                                                     estimated_positions,
                                                     sample_frame,
                                                     patch_size=(19, 19))
                if dices_coeff < 0:
                    continue
                if dices_coeff > max_dices_coeff:
                    max_dices_coeff = dices_coeff
                    best_sigma = s
                    best_H = h

                dices_coefficients.append(dices_coeff)
                sigmas.append(s)
                Hs.append(h)

                df = pd.DataFrame()
                df['dices_coefficient'] = dices_coefficients
                df['sigma'] = sigmas
                df['extended_maxima_h'] = Hs

        print(f'Saving results to {filename}')
        df.to_csv(filename)
        print('Done')

    pd.set_option('display.max_rows', 3)
    display.display(df)

    max_dices_coeff_idx = df['dices_coefficient'].argmax()

    print("Maximum Dice's coefficient values:\n")
    print(df.iloc[max_dices_coeff_idx])
    print()
    best_dices_coefficient = df.loc[max_dices_coeff_idx, 'dices_coefficient']
    best_sigma = df.loc[max_dices_coeff_idx, 'sigma']
    best_h = df.loc[max_dices_coeff_idx, 'extended_maxima_h']

    # Estimate cell positions
    print('Cell position estimation')
    print('------------------------')
    print('Estimating cell positions...')
    gauss_sigma = best_sigma
    extended_maxima_H = best_h

    estimated_positions = get_cell_positions_from_probability_map(probability_map,
                                                                  gauss_sigma,
                                                                  extended_maxima_H,
                                                                  visualise_intermediate_results=True
                                                                  )
    fig = plt.gcf()
    fig.canvas.set_window_title('Blood cell estimation from probability map')
    plt.show()

    dices_coeff, _, _ = evaluate_results(ground_truth_positions,
                                         estimated_positions,
                                         sample_frame,
                                         patch_size=(33, 33))
    fig = plt.figure()
    # print(estimated_positions)
    plt.title(f"Dice's coefficient {dices_coeff:.3f}")
    plt.imshow(sample_frame, cmap='gray')
    plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], s=48, label='Ground truth positions')
    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], s=15, label='Estimated positions')
    plt.legend()

    fig_size = fig.get_size_inches()
    fig.set_size_inches((fig_size[0] * 2,
                         fig_size[1] * 2))
    fig.canvas.set_window_title('Blood cell estimation from probability map')
    print('Done')
    plt.show()

    selected_point_indices = []
    plt.imshow(sample_frame)
    selected_point_indices = guitools.scatter_plot_point_selector(estimated_positions, ax=plt.gca())
    print(selected_point_indices)


if __name__ == "__main__":
    main()
