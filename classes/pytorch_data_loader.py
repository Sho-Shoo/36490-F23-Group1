import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from classes.data_loader import DataLoader as NormalDataLoader


class USAPytorchDataset(Dataset):
    def __init__(self, start: int, end: int):
        super().__init__()
        normal_dataloader = NormalDataLoader("data/usa_short.csv")
        data = normal_dataloader.slice(start, end)
        self.X = normal_dataloader.get_x(data)
        self.y = normal_dataloader.get_y(data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.Tensor(self.X[idx, :])
        y = torch.Tensor(np.reshape(self.y[idx], (self.y[idx].size,)))
        if y.numel() > 1:
            y = y.view(y.numel(), 1)

        return X, y


class USAPytorchDataloader(DataLoader):
    def __init__(self, start, end, **kwargs):
        dataset = USAPytorchDataset(start, end)
        super().__init__(dataset, **kwargs)


# if __name__ == '__main__':
#     dataloader = USAPytorchDataloader(19900101, 20000101, batch_size=128, shuffle=True)
#     for item in dataloader:
#         print(item)
