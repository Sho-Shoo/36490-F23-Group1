from torch import nn


class NN3(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=21, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        )

    def forward(self, x):
        return self.linear_layers(x)


class NN3Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=21, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=3),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.linear_layers(x)

