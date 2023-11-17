from torch import nn
from torchinfo import summary
import numpy as np


class NN5(nn.Module):

    def __init__(self, dropout_rate=0.15):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(21, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.layer5 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out) + out
        out = self.layer3(out) + out
        out = self.layer4(out) + out
        out = self.layer5(out)
        return out



if __name__ == "__main__":
    nn5 = NN5()
    summary(nn5, input_size=(1024, 21))

