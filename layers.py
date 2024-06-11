import torch


class NormLayer(torch.nn.Module):
    """ Custom Normalization Layer """
    def forward(self, x):
        return x / 255.0