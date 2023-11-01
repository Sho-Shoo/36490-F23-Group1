import torchmetrics
from classes.nn3 import NN3
from classes.nn_utils import train_nn
from classes.pytorch_data_loader import USAPytorchDataloader
import torch
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics.metric import Metric

if __name__ == "__main__":
    nn3 = NN3()
    train_dataloader = USAPytorchDataloader(19900101, 20000101, batch_size=1024)
    validation_dataloader = USAPytorchDataloader(20000101, 20050101, batch_size=1024)
    loss_fn: Loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=nn3.parameters(), lr=0.001, momentum=0.9)
    metric: Metric = torchmetrics.R2Score()

    model_state, train_r2, val_r2 = train_nn(nn3, train_dataloader, validation_dataloader, optimizer, loss_fn, metric,
                                             "dummy_run", "nn3", EPOCHS=5)

    print(model_state)
    print(train_r2, val_r2)
