import pathlib
import argparse

import torch
from matplotlib import collections
from torch import nn

from cell_no_cell import load_model_from_cache
from cnnlearning import CNN
from generate_datasets import get_cell_and_no_cell_patches
from classificationutils import classify_images, classify_labeled_dataset
from sharedvariables import *
from cell_no_cell import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


def train_model_demo(patch_size=(21, 21), do_hist_match=False, n_negatives_per_positive=3):
    trainset, validset, \
    cell_images, non_cell_images, \
    cell_images_marked, non_cell_images_marked, hist_match_template = \
        get_cell_and_no_cell_patches(
            patch_size=patch_size,
            n_negatives_per_positive=n_negatives_per_positive,
            do_hist_match=do_hist_match,
        )

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
        # PrintLayer("9"),
        nn.BatchNorm2d(64),
        # PrintLayer("11"),
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
            #   nn.Softmax()
        )).to(device)

    pathlib.Path(CACHED_MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

    try:
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

        results = train(model, train_params, criterion=torch.nn.CrossEntropyLoss(), device=device)

        output_name = os.path.join(CACHED_MODELS_FOLDER,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch-size', default=21, type=int, help='Patch size')
    parser.add_argument('--hist-match', action='store_true',
                        help='Set this flag to do histogram match.')
    parser.add_argument('-n', '--n-negatives-per-positive', default=3, type=int)

    args = parser.parse_args()
    print('---------------------------------------')
    patch_size = args.patch_size, args.patch_size
    hist_match = args.hist_match
    npp = args.n_negatives_per_positive
    print('---------------------------------------')

    train_model_demo(
        patch_size=patch_size,
        do_hist_match=hist_match,
        n_negatives_per_positive=npp,
    )


if __name__ == '__main__':
    main()

