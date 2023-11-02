import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss as Loss
from classes.pytorch_data_loader import USAPytorchDataloader
import torch
import os
from torchmetrics.metric import Metric

def r2oos(y, yhat):
    num = np.sum((y - yhat)**2)
    den = np.sum((y)**2)
    return 1 - num/den

def train_nn(model: torch.nn.Module,
             train_dataloader: DataLoader,
             validation_dataloader: DataLoader,
             optimizer: Optimizer,
             loss_fn: Loss,
             metric: Metric,
             experiment_name: str,
             model_name: str,
             verbose=False,
             EPOCHS=50):
    """
    Train and validation loop of PyTorch NN model
    :param model: nn.Module
    :param train_dataloader:
    :param validation_dataloader:
    :param optimizer:
    :param loss_fn:
    :param metric:
    :param experiment_name:
    :param model_name:
    :param verbose: defaults to False
    :param EPOCHS:
    :return: tuple(dict, float, float) state_dict snapshot of the best model in validation, train_measure,
        validation_measure
    """

    # device-agnostic setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = metric.to(device)
    model = model.to(device)

    # to be returned as lists
    train_measurements = []
    validation_measurements = []
    best_validation_measure = float('-Inf')
    best_train_measure = None
    best_model_snapshot = None

    # to save checkpoints and outputs
    folder_dir = os.path.join("outputs", model_name, experiment_name)
    try:
        os.makedirs(folder_dir)
    except FileExistsError:
        pass

    for epoch in tqdm(range(EPOCHS)):
        # Training loop
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            model.train()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.inference_mode():
            X, y = train_dataloader.dataset[:]
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            train_loss = loss_fn(y_pred, y)
            train_measurement = metric(y_pred, y)
            train_measurements.append(train_measurement)

            X, y = validation_dataloader.dataset[:]
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            val_loss = loss_fn(y_pred, y)
            val_measurement = metric(y_pred, y)
            validation_measurements.append(val_measurement)

        if verbose:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train measure: {train_measurement: .5f}| "
                  f"Val loss: {val_loss: .5f}| Val measure: {val_measurement: .5f}")

        if val_measurement > best_validation_measure:
            best_validation_measure = val_measurement
            best_train_measure = train_measurement
            best_model_snapshot = model.state_dict().copy()

    return best_model_snapshot, best_train_measure, best_validation_measure


def test_nn(model: nn.Module,
            test_dataloader: DataLoader,
            metric: Metric) -> tuple[float, np.array]:
    """
    Test NN on a given test dataset
    :param model: nn.Module model
    :param test_dataloader: PyTorch data loader
    :param metric:
    :return: any measurement outputed by metric parameter (can be R^2, accuracy, etc.), and prediction results as 1D
        numpy array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = metric.to(device)
    model = model.to(device)

    X, y = test_dataloader.dataset[:]
    X, y = X.to(device), y.to(device)

    model.eval()
    with torch.inference_mode():
        y_pred = model(X)
        measure = metric(y_pred, y)
        y_pred = np.array(y_pred).reshape((y_pred.numel(),))

    return float(measure), y_pred


def evaluate_nn(model_type: nn.Module.__class__,
                model_state: dict,
                data,
                test_start: int,
                test_end: int,
                metric: Metric) -> tuple[list, list]:
    """
    Evaluate NN model on a monthly basis
    :param model_type:
    :param model_state:
    :param data:
    :param test_start:
    :param test_end:
    :param metric:
    :return: a list of floating point measurements; and a list of predictions as numpy arrays
    """

    model = model_type()
    model.load_state_dict(model_state)
    measures = []
    monthly_predicitons = []

    start_year, end_year = test_start // 10000, test_end // 10000
    for year in range(start_year, end_year):
        for month in range(1, 13):
            start = int(f"{year}{month:02d}01")
            if month == 12:
                end = int(f"{year + 1}0101")
            else:
                end = int(f"{year}{month + 1:02d}01")

            test_dataloader = USAPytorchDataloader(data, start, end, batch_size=1024)
            measure, prediction = test_nn(model, test_dataloader, metric)
            measures.append(measure)
            monthly_predicitons.append(prediction)

    return measures, monthly_predicitons
