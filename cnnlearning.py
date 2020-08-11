from typing import List, Any

import torch
import tqdm
from torch.utils import data
import pandas as pd
import numpy as np
import time
import collections
from torch import nn
import copy
from learningutils import ImageDataset, LabeledImageDataset
from IPython.display import clear_output
from IPython.display import display

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import sys
import signal


class PrintLayer(nn.Module):
    def __init__(self, msg=""):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.msg, x.shape)
        return x


# Build the neural network, expand on top of nn.Module
class CNN(nn.Module):
    def __init__(self, input_dims=3, output_classes=2, dense_input_dims=576, padding=0, convolutional=None, dense=None):
        super().__init__()

        # Feature extraction layer
        if convolutional is None:
            self.convolutional = nn.Sequential(
                nn.Conv2d(input_dims, 16, kernel_size=3, padding=padding, padding_mode='replicate'),
                # Print(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 8, kernel_size=3),
                # Print(),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, stride=1),
                # Print(),
                nn.Conv2d(8, 4, kernel_size=2),
                # Print("Conv2d 8, 4, kernel_size=2"),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                # Print(),
                nn.AvgPool2d(kernel_size=(3, 3), stride=1),
                # Print("nn.AvgPool2d(kernel_size=(3, 3), stride=2"),
            )
        else:
            self.convolutional = convolutional

        if dense is None:
            # Fully connected layer
            self.dense = nn.Sequential(
                nn.Dropout(),
                nn.Linear(16, 8),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(4, output_classes),
            )
        else:
            self.dense = dense
        self.dense_input_dims = dense_input_dims

    # define forward function
    def forward(self, t):
        t = self.convolutional(t)
        t = t.reshape(-1, self.dense_input_dims)
        t = self.dense(t)
        return t


# Helper class, help track loss, accuracy, epoch time, run time,
# hyper-parameters etc.
class TrainingTracker:
    additional_display_dfs: List[pd.DataFrame]

    def __init__(self, device, additional_display_dfs= []):
        self.additional_display_dfs = additional_display_dfs
        self.epoch_count = 0
        self.epoch_start_time = None
        self.epoch_duration = None

        self.run_start_time = None
        self.run_duration = None
        self.run_data = []

        # track every loss and performance
        self.train_losses = []
        self.valid_losses = []

        self.train_accuracies = []
        self.valid_accuracies = []

        # Track training performance metrics
        self.epoch_durations = []

        self.best_valid_loss = np.inf
        self.is_best_valid_loss_recorded = False

        self.best_valid_accuracy = 0
        self.is_best_valid_accuracy_recorded = False

        self.is_model_recorded = False

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_start_time = None

        # Model is updated each epoch
        self.model = None

        # Recorded model is updated each time record_model() is called
        self.recorded_model = None
        self.recorded_model_weights = None
        self.recorded_model_epoch = None
        self.recorded_model_valid_accuracy = None
        self.recorded_model_valid_loss = None

        self.train_loader = None
        self.valid_loader = None
        self.criterion = None

        # self.valid_untrunsformed_normals = None
        # self.valid_pca3_transforms = None
        # self.valid_pca2_transforms = None
        self.device = device

        self.do_early_stop = True
        self.early_stop_patience = 20
        self._times_since_last_best_valid_loss = 0
        self._times_since_last_best_valid_accuracy = 0

    def start_run(self, model, params, train_loader, valid_loader, criterion):
        self.run_start_time = time.time()

        if 'do_early_stop' in params:
            self.do_early_stop = params['do_early_stop']
        if 'early_stop_patience' in params:
            self.early_stop_patience = params['early_stop_patience']

        self.run_params = params
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion

    def end_run(self):
        self.run_duration = time.time() - self.run_start_time
        print("Run duration ", self.run_duration)

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self):
        self.epoch_count += 1
        self.epoch_duration = time.time() - self.epoch_start_time

        self.is_best_valid_accuracy_recorded = False
        self.is_best_valid_loss_recorded = False
        self.is_model_recorded = False

    def display_results(self):
        # Write into 'results' (OrderedDict) for all run related data
        results = collections.OrderedDict()
        results["epoch"] = self.epoch_count
        # record epoch loss and accuracy

        if self.train_losses:
            results["train loss"] = self.train_losses[-1]
        if self.valid_losses:
            results["valid loss"] = self.valid_losses[-1]

        if self.train_accuracies:
            results["train accuracy"] = self.train_accuracies[-1]
        if self.valid_accuracies:
            results["valid accuracy"] = self.valid_accuracies[-1]

        results["Best Loss?"] = self.is_best_valid_loss_recorded
        results["Best Accuracy?"] = self.is_best_valid_accuracy_recorded
        results["Model Recorded?"] = self.is_model_recorded

        if self.do_early_stop:
            results["Times since last best loss"] = self._times_since_last_best_valid_loss
            results["Times since last best accuracy"] = self._times_since_last_best_valid_accuracy

        run_parameters = collections.OrderedDict()
        for param_group in self.run_params['optimizer'].param_groups:
            run_parameters["learning rate"] = param_group["lr"]
            run_parameters["weight decay"] = param_group["weight_decay"]

        # Record hyper-params into 'results'
        for k, v in self.run_params.items():
            if k in ["batch_size", "do_early_stop",
                     "early_stop_patience", "learning_rate_scheduler_patience",
                     "epochs", "shuffle"]:
                run_parameters[k] = v
            elif k == "trainset":
                run_parameters["Trainset size"] = len(self.run_params[k])
            elif k == "validset":
                run_parameters["Validset size "] = len(self.run_params[k])
            elif k not in ["trainset", "validset", "testset", "optimizer"]:
                results[k] = v

        self.run_data.append(results)
        run_parameters_df = pd.DataFrame.from_dict([run_parameters], orient='columns')
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # display epoch information and show progress
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):  # more options can be specified also
            clear_output()
            for additional_display_df in self.additional_display_dfs:
                display(additional_display_df)
            display(run_parameters_df)
            display(df)

    # noinspection PyUnresolvedReferences
    @torch.no_grad()
    def load(self, filename, input_dims):
        model = CNN(input_dims=input_dims)
        model.load_state_dict(torch.load(filename))
        model.eval()

    @torch.no_grad()
    def save(self, output_name, model=None):
        """ Saves the recorded model as {output_name}.pt among other info files.

        Makes {output_name}.pt, {output_name}.txt with recorded epoch loss, accuracy and other run parameters
        and {output_name}_run_parameters.txt with the hyperparameters that the network was ran (learning_rate, patience
        e.t.c)

        :param output_name:
            The prefix for the files (model, run parameters, run data)

        Args:
            model:
        """
        if model == None:
            model = self.recorded_model
        # https: // pytorch.org / tutorials / beginner / saving_loading_models.html  # save-load-state-dict-recommended
        torch.save(model.state_dict(), f'{output_name}.pt')

        run_parameters = collections.OrderedDict()
        for param_group in self.run_params['optimizer'].param_groups:
            run_parameters["learning rate"] = param_group["lr"]

        # Record hyper-params into 'results'
        for k, v in self.run_params.items():
            if k in ["batch_size", "do_early_stop",
                     "early_stop_patience", "learning_rate_scheduler_patience",
                     "epochs", "shuffle"]:
                run_parameters[k] = v

        run_parameters_df = pd.DataFrame.from_dict([run_parameters], orient='columns')
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        with open(f'{output_name}.txt', 'w') as fo:
            fo.write(df.__repr__())

        with open(f'{output_name}_run_parameters.txt', 'w') as fo:
            fo.write(run_parameters_df.__repr__())

    @torch.no_grad()
    def track_loss_and_accuracy(self, train_loss=None, train_accuracy=None):
        if train_loss is None or train_accuracy is None:
            train_loss, train_accuracy = self._get_loss_and_accuracy(self.train_loader)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)

        valid_loss, valid_accuracy = self._get_loss_and_accuracy(self.valid_loader)
        self.valid_losses.append(valid_loss)
        self.valid_accuracies.append(valid_accuracy)

        if valid_loss < self.best_valid_loss:
            self.is_best_valid_loss_recorded = True
            self.best_valid_loss = valid_loss
            self._times_since_last_best_valid_loss = 0
        else:
            self.is_best_valid_loss_recorded = False
            self._times_since_last_best_valid_loss += 1

        if valid_accuracy > self.best_valid_accuracy:
            self.is_best_valid_accuracy_recorded = True
            self.best_valid_accuracy = valid_accuracy
            self._times_since_last_best_valid_accuracy = 0
        else:
            self.is_best_valid_accuracy_recorded = False
            self._times_since_last_best_valid_accuracy += 1

    def get_last_valid_accuracy(self):
        return self.valid_accuracies[-1]

    def get_last_valid_loss(self):
        return self.valid_losses[-1]

    def record_model(self, model=None):
        if model is None:
            model = self.model

        self.recorded_model_weights = copy.deepcopy(model.state_dict())
        self.recorded_model = copy.deepcopy(model)
        self.recorded_model = self.recorded_model.eval()
        self.recorded_model_epoch = self.epoch_count
        self.recorded_model_valid_accuracy = self.valid_accuracies[-1]
        self.recorded_model_valid_loss = self.valid_losses[-1]

        self.is_model_recorded = True

    @torch.no_grad()
    # accumulate loss of batch into entire epoch loss
    def track_loss(self, train_loss=None):
        if train_loss is None:
            train_loss = self._get_loss(self.train_loader)
        self.train_losses.append(train_loss)

        valid_loss = self._get_loss(self.valid_loader)
        self.valid_losses.append(valid_loss)

    @torch.no_grad()
    def track_accuracy(self, train_accuracy=None):
        if train_accuracy is None:
            train_accuracy = self._get_accuracy(self.train_loader)
        self.train_accuracies.append(train_accuracy)

        valid_accuracy = self._get_accuracy(self.train_accuracies)
        self.valid_accuracies.append(valid_accuracy)

    def should_early_stop(self):
        return self.do_early_stop and self._times_since_last_best_valid_loss >= self.early_stop_patience

    @torch.no_grad()
    def _get_accuracy(self, loader):
        n_samples = 0
        n_correct = 0

        for batch in loader:
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device).type(torch.long)

            output = self.model(images)
            predictions = torch.zeros_like(output, dtype=torch.long)
            predictions[output >= 0.5] = 1

            n_samples += images.shape[0]
            n_correct += torch.sum(predictions == targets).item()

        return n_samples/n_correct

    @torch.no_grad()
    def _get_loss(self, loader):
        total_loss = 0
        n_samples = 0

        for batch in loader:
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            # print(images.device)
            # print(targets.device)
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)

            n_samples += targets.shape[0]
            total_loss += loss.item()

        total_loss /= n_samples

        return total_loss

    @torch.no_grad()
    def _get_loss_and_accuracy(self, loader):
        self.model.eval()

        total_loss = 0

        n_samples = 0
        n_correct = 0
        for batch in loader:
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device).type(torch.long)

            output = self.model(images)
            loss = self.criterion(output, targets)

            print("Output", output.shape)
            print("Targets", targets.shape)
            total_loss += loss.item()

            predictions = torch.argmax(output, dim=1).type(torch.int)
            n_correct += torch.sum(predictions == targets).item()
            n_samples += targets.shape[0]

        total_loss /= n_samples
        accuracy = n_correct / n_samples
        print("Accuracy", accuracy)

        self.model.train()
        return total_loss, accuracy


class SignalHandler(object):
    def __init__(self):
        self.signal_raised = False

    def handle(self, sig, frm):
        print(f"Signal {sig} captured.")
        self.signal_raised = True


def train(cnn, params,
          device="cuda",
          criterion=nn.BCELoss(),
          additional_display_dfs=[]
          ):
    # if params changes, following line of code should reflect the changes too
    train_loader = torch.utils.data.DataLoader(
        params['trainset'],
        batch_size=params['batch_size'],
        shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        params['validset'],
        batch_size=params['batch_size'],
        shuffle=False
    )

    epochs = params['epochs']

    # Set up Optimizer
    if 'optimizer' not in params:
        lr = .001
        weight_decay = 5e-4
        if 'lr' in params:
            lr = params['lr']
        if 'weight_decay' in params:
            weight_decay = params['weight_decay']
        params['optimizer'] = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = params['optimizer']

    # Set up learning rate scheduler
    if 'learning_rate_scheduler_patience' not in params:
        if 'early_stop_patience' in params:
            params['learning_rate_scheduler_patience'] = int(0.5 * params['early_stop_patience'])
        else:
            params['learning_rate_scheduler_patience'] = 10
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                         'min',
                                                                         patience=params[
                                                                             'learning_rate_scheduler_patience'])

    # Tracker tracks the process and helps with early stopping
    tracker = TrainingTracker(device, additional_display_dfs)
    tracker.start_run(cnn, params, train_loader, valid_loader, criterion)

    interrupt_handler = SignalHandler()

    signal.signal(signal.SIGINT, interrupt_handler.handle)

    for epoch in tqdm.tqdm(range(epochs)):
        if tracker.should_early_stop() or interrupt_handler.signal_raised:
            break

        tracker.start_epoch()

        total_loss = 0
        n_samples = 0
        n_correct = 0
        for batch in train_loader:
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()

            images = batch[0].to(device).type(torch.float32)
            # print("Images", images.shape)
            labels = batch[1].to(device).type(torch.long)
            output = cnn(images).type(torch.float32).to(device)

            # print("output", output.shape, output.dtype)
            # print("labels", labels.shape, labels.dtype)
            # print("Output max", output.max())
            # print("Output min", output.min())
            loss = criterion(output, labels)

            predictions = torch.argmax(output, dim=1)

            n_samples += images.shape[0]
            n_correct += torch.sum(predictions == labels).item()

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            total_loss /= n_samples

        accuracy = n_correct / n_samples
        if epoch % 20 == 0 or epoch == epochs - 1:
            tracker.track_loss_and_accuracy(train_loss=total_loss, train_accuracy=accuracy)

            if tracker.is_best_valid_accuracy_recorded:
                tracker.record_model()

            tracker.display_results()

            # Learning rate scheduler based on accuracy
            learning_rate_scheduler.step(tracker.get_last_valid_accuracy())

        tracker.end_epoch()

    tracker.end_run()

    return tracker
