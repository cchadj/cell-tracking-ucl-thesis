import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2
from generate_datasets import get_cell_and_no_cell_patches_from_video

from IPython import display
from evaluation import evaluate_results
from learningutils import LabeledImageDataset
from cnnlearning import CNN, train
import collections
from patchextraction import extract_patches_at_positions
from sharedvariables import *
from classificationutils import classify_labeled_dataset, create_probability_map, get_cell_positions_from_probability_map
import guitools
from tqdm.contrib import tzip
import pathlib


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


def plot_images_as_grid(images, ax=None, title=None):
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
    if ax is None:
        _, ax = plt.subplots(num=None, figsize=(70, 50), dpi=80, facecolor='w', edgecolor='k')
    if title is not None:
        ax.set_title(title)

    plt.grid(b=None)
    ax.imshow(grid_img.permute(1, 2, 0))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
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
