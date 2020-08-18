import pathlib
import argparse

import torch
import collections
# import cPickle as pickle
import pickle
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


def load_model_from_cache(model,
                          patch_size=(21, 21),
                          n_negatives_per_positive=3,
                          temporal_width=0,
                          standardize_dataset=True,
                          hist_match=False):
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
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'st') == str(standardize_dataset).lower()
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'tw') == temporal_width
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
                     temporal_width=0,
                     do_hist_match=False,
                     n_negatives_per_positive=3,
                     device='cuda',
                     standardize_dataset=True,
                     try_load_data_from_cache=True,
                     try_load_model_from_cache=True,
                     train_params=None,
                     additional_displays=None,
                     ):
    assert type(patch_size) is int or type(patch_size) is tuple
    if type(patch_size) is int:
        patch_size = patch_size, patch_size

    print('Creating (or loading from cache) data...')
    trainset, validset, \
        cell_images, non_cell_images, \
        cell_images_marked, non_cell_images_marked, \
        hist_match_template = \
        get_cell_and_no_cell_patches(
            patch_size=patch_size,
            n_negatives_per_positive=n_negatives_per_positive,
            standardize_dataset=standardize_dataset,
            temporal_width=temporal_width,
            do_hist_match=do_hist_match,
            try_load_from_cache=try_load_data_from_cache,
        )

    assert cell_images.dtype == np.uint8 and non_cell_images.dtype == np.uint8,\
        print(f'Cell images dtype {cell_images.dtype} non cell images dtype {non_cell_images.dtype}')
    assert cell_images.min() >= 0 and cell_images.max() <= 255
    assert non_cell_images.min() >= 0 and non_cell_images.max() <= 255

    # noinspection PyUnresolvedReferences
    model = CNN(dataset_sample=trainset, output_classes=2).to(device)

    pathlib.Path(CACHED_MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

    try:
        if not try_load_model_from_cache:
            print(f'Not loading model and results from cache.')
            raise FileNotFoundError
        print(f'Attempting to load model and results from cache with patch_size:{patch_size}, '
              f' histogram_match: {do_hist_match}, n negatives per positive: {n_negatives_per_positive}')
        model_directory = load_model_from_cache(model, patch_size=patch_size,
                                                hist_match=do_hist_match,
                                                n_negatives_per_positive=n_negatives_per_positive)
        print(f"Model found. Loaded model from '{model_directory}'")
        with open(os.path.join(model_directory, 'results.pkl'), 'rb') as results_file:
            results = pickle.load(results_file)

    except FileNotFoundError:
        if try_load_model_from_cache:
            additional_displays.append('Model or results not found.')
        print('Training new model.\n'
              'You can interrupt(ctr - C or interrupt kernel) any time to get '
              'the model with the best validation accuracy at the current time.')

        if train_params is None:
            train_params = collections.OrderedDict(
                optimizer=torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4),
                batch_size=1024 * 7,
                do_early_stop=True,  # Optional default True
                early_stop_patience=80,
                learning_rate_scheduler_patience=100,
                epochs=4000,
                shuffle=True,
                trainset=trainset,
                validset=validset,
            )
        else:
            if 'trainset' not in train_params:
                train_params['trainset'] = trainset
            if 'validset' not in train_params:
                train_params['validset'] = validset

        if additional_displays is None:
            additional_displays = []
        run_configuration_display = collections.OrderedDict({
            'patch_size': patch_size[0],
            'temporal width': temporal_width,
            'hist match': do_hist_match,
            'standardize data': standardize_dataset,
            'nnp': n_negatives_per_positive
        })
        additional_displays.append(run_configuration_display)

        results: TrainingTracker = train(model,
                                         train_params,
                                         criterion=torch.nn.CrossEntropyLoss(),
                                         device=device, additional_displays=additional_displays)

        output_directory = os.path.join(CACHED_MODELS_FOLDER,
                                        f'blood_cell_classifier_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}'
                                        f'_npp_{n_negatives_per_positive}_va_{results.recorded_model_valid_accuracy:.3f}')
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        output_name = os.path.join(output_directory,
                                   f'blood_cell_classifier_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}'
                                   f'_npp_{n_negatives_per_positive}_va_{results.recorded_model_valid_accuracy:.3f}'
                                   f'_st_{str(standardize_dataset).lower()}_tw_{temporal_width}')

        print(f'Saving model as {output_name}')
        results.save(output_name)
        results_file = os.path.join(output_directory, 'results.pkl')
        print(f"Saving results as {results_file}")
        with open(results_file, 'wb') as output_file:
            pickle.dump(results, output_file, pickle.HIGHEST_PROTOCOL)

        print('Done')

    # Print the final validation accuracy on the training and validation set
    model = results.recorded_model
    model.eval()

    trainset_predictions, train_accuracy = classify_labeled_dataset(trainset, model)
    validset_predictions, valid_accuracy = classify_labeled_dataset(validset, model)
    positive_accuracy = classify_images(cell_images, model, standardize_dataset=standardize_dataset).sum().item() / len(cell_images)
    negative_accuracy = (1 - classify_images(non_cell_images, model, standardize_dataset=standardize_dataset)).sum().item() / len(non_cell_images)

    print()
    print(f'Model trained on {len(cell_images)} cell patches and {len(non_cell_images)} non cell patches.')
    print()
    print('Brief evaluation - best validation accuracy model')
    print('----------------')
    print('Training accuracy:\t', f'{train_accuracy:.3f}')
    print('Validation accuracy:\t', f'{valid_accuracy:.3f}')
    print()
    print('Positive accuracy:\t', f'{positive_accuracy:.3f}')
    print('Negative accuracy:\t', f'{negative_accuracy:.3f}')

    train_model = results.recorded_train_model
    train_model.eval()

    _, train_accuracy = classify_labeled_dataset(trainset, train_model)
    _, valid_accuracy = classify_labeled_dataset(validset, train_model)
    positive_accuracy = classify_images(cell_images, train_model, standardize_dataset=standardize_dataset).sum().item() / len(cell_images)
    negative_accuracy = (1 - classify_images(non_cell_images, train_model, standardize_dataset=standardize_dataset)).sum().item() / len(non_cell_images)

    print()
    print('Brief evaluation - best training accuracy model')
    print('----------------')
    print('Training accuracy:\t', f'{train_accuracy:.3f}')
    print('Validation accuracy:\t', f'{valid_accuracy:.3f}')
    print()
    print('Positive accuracy:\t', f'{positive_accuracy:.3f}')
    print('Negative accuracy:\t', f'{negative_accuracy:.3f}')

    return model, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ps', '-p', '--patch-size', default=21, type=int, help='Patch size')
    parser.add_argument('-tw', '-t', '--temporal-width', default=0, type=int, help='Temporal width')
    parser.add_argument('-st', '-s', '--standardize', action='store_true', help='Set this flag to standardize dataset output between -1 and 1')
    parser.add_argument('-hm', '--hist-match', action='store_true',
                        help='Set this flag to do histogram match.')
    parser.add_argument('-npp', '-n', '--n-negatives-per-positive', default=3, type=int)
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
    temporal_width = args.temporal_width
    print('---------------------------------------')

    train_model_demo(
        patch_size=patch_size,
        do_hist_match=hist_match,
        n_negatives_per_positive=npp,
        standardize_dataset=standardize,
        temporal_width=temporal_width,
        device=device,
        try_load_data_from_cache=True,
        try_load_model_from_cache=True,
    )


def main_tmp():
    patch_size = 21
    temporal_width = 1
    hist_match = False
    standardize = True
    npp = 1
    device = 'cuda'

    train_model_demo(
        patch_size=patch_size,
        temporal_width=temporal_width,
        do_hist_match=hist_match,
        n_negatives_per_positive=npp,
        standardize_dataset=standardize,
        device=device,

        try_load_data_from_cache=True,
        try_load_model_from_cache=True,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        main_tmp()
