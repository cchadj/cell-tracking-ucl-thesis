import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from cnnlearning import CNN, train
from sharedvariables import *
import pathlib




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
    potential_models_filenames = [
        f for f in os.listdir(CACHED_MODELS_FOLDER)
        if int(extract_value_from_string(f, 'ps')) == patch_size[0] and
        int(extract_value_from_string(f, 'npp')) == n_negatives_per_positive and
        extract_value_from_string(f, 'hm') == str(hist_match).lower()
    ]

    if len(potential_models_filenames) == 0:
        raise FileNotFoundError('No model file in cache')

    best_model_idx = np.argmax([float(extract_value_from_string(f, 'va')) for f in potential_models_filenames])
    best_model_filename = os.path.join(CACHED_MODELS_FOLDER, potential_models_filenames[best_model_idx])
    model.load_state_dict(torch.load(best_model_filename))
    model.eval()

    return best_model_filename


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

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

    print(load_model_from_cache(model))


if __name__ == "__main__":
    main()
