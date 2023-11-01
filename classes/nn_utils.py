from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss as Loss
import torch
import os
from torchmetrics.metric import Metric


def train_nn(model: torch.nn.Module,
             train_dataloader: DataLoader,
             validation_dataloader: DataLoader,
             optimizer: Optimizer,
             loss_fn: Loss,
             metric: Metric,
             experiment_name: str,
             model_name: str,
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

        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train measure: {train_measurement: .5f}| "
              f"Val loss: {val_loss: .5f}| Val measure: {val_measurement: .5f}")

        if val_measurement > best_validation_measure:
            best_validation_measure = val_measurement
            best_train_measure = train_measurement
            best_model_snapshot = model.state_dict().copy()

    return best_model_snapshot, best_train_measure, best_validation_measure
