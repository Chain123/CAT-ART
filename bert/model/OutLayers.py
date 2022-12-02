import torch.nn as nn


class gender_out_layer(nn.Module):

    def __init__(self, MidDims, OutDim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MidDims[0], MidDims[1]),
            nn.BatchNorm1d(MidDims[1]),
            nn.PReLU(),
            nn.Linear(MidDims[1], MidDims[2]),
            nn.BatchNorm1d(MidDims[2]),
            nn.PReLU(),
            nn.Linear(MidDims[2], OutDim)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
