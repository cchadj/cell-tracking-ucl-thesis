import pathlib
import argparse

import torch
import collections
import pandas as pd
from torch import nn
import os
import numpy as np
from cnnlearning import CNN, train, TrainingTracker
from generate_datasets import get_cell_and_no_cell_patches
from classificationutils import classify_images, classify_labeled_dataset
from sharedvariables import CACHED_MODELS_FOLDER


def extract_value_from_string(string, value_prefix, delimiter='_'):
    strings = pathlib.Path(string).with_suffix('').name.split(delimiter)
    val = None
    for i, s in enumerate(strings):
        if s == value_prefix:
            # val = float(re.findall(r"[-+]?\d*\.\d+|\d+", strings[i + 1])[0])
            val = strings[i + 1]
            break

    return val


def load_model_from_cache(model, patch_size=(21, 21), n_negatives_per_positive=3, hist_match=False):
    """ Attempts to find the model weights to model from cache.

    Args:
        model: The model to attempt to load.
        patch_size: The patch size used to train the model.
        n_negatives_per_positive:  The number of negatives per positive used to train the model.
        hist_match: Whether histogram matching was used to train the model.

    Returns:
        The filename of the model

    """
    potential_model_directories = [
        f for f in os.listdir(CACHED_MODELS_FOLDER) if os.path.isdir(os.path.join(CACHED_MODELS_FOLDER, f))
    ]
    potential_model_directories = [
        f for f in potential_model_directories if int(extract_value_from_string(f, 'npp')) == n_negatives_per_positive
    ]
    potential_model_directories = [
        f for f in potential_model_directories if int(extract_value_from_string(f, 'ps')) == patch_size[0]
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'hm') == str(hist_match).lower()
    ]

    if len(potential_model_directories) == 0:
        raise FileNotFoundError('No model directory in cache')

    best_model_idx = np.argmax([float(extract_value_from_string(f, 'va')) for f in potential_model_directories])
    best_model_directory = os.path.join(CACHED_MODELS_FOLDER, potential_model_directories[best_model_idx])

    best_model_file = \
        [os.path.join(best_model_directory, f) for f in os.listdir(best_model_directory) if f.endswith('.pt')][0]
    model.load_state_dict(torch.load(best_model_file))
    model.eval()

    return best_model_directory


def train_model_demo(patch_size=(21, 21),
                     do_hist_match=False,
                     n_negatives_per_positive=3,
                     device='cuda',
                     standardize_dataset=True,
                     load_from_cache=True,
                     train_params=None):
    assert type(patch_size) is int or type(patch_size) is tuple
    if type(patch_size) is int:
        patch_size = patch_size, patch_size

    trainset, validset, \
        cell_images, non_cell_images, \
        cell_images_marked, non_cell_images_marked, \
        hist_match_template = \
        get_cell_and_no_cell_patches(
            patch_size=patch_size,
            n_negatives_per_positive=n_negatives_per_positive,
            standardize_dataset=standardize_dataset,
            do_hist_match=do_hist_match,
            overwrite_cache=True,
        )

    assert cell_images.dtype == np.uint8 and non_cell_images.dtype == np.uint8
    assert cell_images.min() >= 0 and cell_images.max() <= 255
    assert non_cell_images.min() >= 0 and non_cell_images.max() <= 255

    # noinspection PyUnresolvedReferences
    model = CNN().to(device)

    pathlib.Path(CACHED_MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

    try:
        if not load_from_cache:
            print(f'Not loading from cache.')
            raise FileNotFoundError
        print(f'Attempting to load model from cache with patch_size:{patch_size}, '
              f' histogram_match: {do_hist_match}, n negatives per positive: {n_negatives_per_positive}')
        model_filename = load_model_from_cache(model, patch_size=patch_size,
                                               hist_match=do_hist_match,
                                               n_negatives_per_positive=n_negatives_per_positive)
        print(f'Model found. Loaded model from {model_filename}')
    except FileNotFoundError:
        pass
        print('Model not found. Training new model. You can interrupt(ctr - C or interrupt kernel) any time to get'
              'the model with the best validation accuracy at the current time.')

        if train_params is None:
            train_params = collections.OrderedDict(
                # lr = .001,
                # optimizer=torch.optim.SGD(model.parameters(), lr=.001, weight_decay=5e-5, momentum=0.9),
                optimizer=torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4),
                batch_size=1024 * 7,
                do_early_stop=True,  # Optional default True
                early_stop_patience=80,
                learning_rate_scheduler_patience=100,
                epochs=4000,
                shuffle=True,
                # valid_untrunsformed_normals = valid_untrunsformed_normals,
                trainset=trainset,
                validset=validset,
            )
        else:
            if 'trainset' not in train_params:
                train_params['trainset'] = trainset
            if 'validset' not in train_params:
                train_params['validset'] = validset

        display_dict = collections.OrderedDict({
            "patch_size": patch_size,
            "do_hist_match": do_hist_match,
            "nnp": n_negatives_per_positive
        })
        additional_display = [
            pd.DataFrame.from_dict(display_dict)
        ]
        results: TrainingTracker = train(model,
                                         train_params,
                                         criterion=torch.nn.CrossEntropyLoss(),
                                         device=device, additional_display_dfs=additional_display)

        output_directory = os.path.join(CACHED_MODELS_FOLDER,
                                        f'blood_cell_classifier_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}'
                                        f'_npp_{n_negatives_per_positive}_va_{results.recorded_model_valid_accuracy:.3f}')
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        output_name = os.path.join(output_directory,
                                   f'blood_cell_classifier_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}'
                                   f'_npp_{n_negatives_per_positive}_va_{results.recorded_model_valid_accuracy:.3f}')

        print(f'Saving model as {output_name}')
        results.save(output_name)
        print('Done')
        model = results.recorded_model

    # Print the final validation accuracy on the training and validation set
    model.eval()
    trainset_predictions, train_accuracy = classify_labeled_dataset(trainset, model)
    validset_predictions, valid_accuracy = classify_labeled_dataset(validset, model)

    print()
    print(f'Model trained on {len(cell_images)} cell patches and {len(non_cell_images)} non cell patches.')
    print()
    print('Brief evaluation')
    print('----------------')
    print('Training accuracy:\t', f'{train_accuracy:.3f}')
    print('Validation accuracy:\t', f'{valid_accuracy:.3f}')
    print()
    print('Positive accuracy:\t',
          f'{classify_images(cell_images, model).sum().item() / len(cell_images):.3f}')
    print('Negative accuracy:\t',
          f'{(1 - classify_images(non_cell_images, model)).sum().item() / len(non_cell_images):.3f}')

    return model, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch-size', default=21, type=int, help='Patch size')
    parser.add_argument('-s', '--standardize', action='store_true', help='Set to standardize output between -1 and 1')
    parser.add_argument('--hist-match', action='store_true',
                        help='Set this flag to do histogram match.')
    parser.add_argument('-n', '--n-negatives-per-positive', default=3, type=int)
    parser.add_argument('-d', '--device', default='cuda', type=str, help="Device to use for training. 'cuda' or 'cpu'")

    args = parser.parse_args()
    available_devices = ['cuda', 'gpu']
    assert args.device in available_devices, f'Device must be one of {available_devices}.'
    print('---------------------------------------')
    device = args.device
    patch_size = args.patch_size, args.patch_size
    hist_match = args.hist_match
    npp = args.n_negatives_per_positive
    standardize = args.standardize
    print('---------------------------------------')

    train_model_demo(
        patch_size=patch_size,
        do_hist_match=hist_match,
        n_negatives_per_positive=npp,
        standardize_dataset=standardize,
        device=device,
        load_from_cache=False,
    )


if __name__ == '__main__':
    main()
