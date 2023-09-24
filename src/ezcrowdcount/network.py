"""Network implementation"""
# pylint: disable=invalid-name,too-many-arguments
import torch
from torch import nn
import numpy as np
import h5py


class Conv2d(nn.Module):
    """Implementation for 2D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        NL="relu",
        same_padding=False,
        bn=False,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=padding
        )
        self.bn = (
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
            if bn
            else None
        )
        if NL == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif NL == "prelu":
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        """Forward pass through the model network"""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    """Fully connect implementation"""

    def __init__(self, in_features, out_features, NL="relu"):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif NL == "prelu":
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Foward pass through the FC Model"""
        input_tensor = self.fc(input_tensor)
        if self.relu is not None:
            input_tensor = self.relu(input_tensor)
        return input_tensor


def load_net(fname, net):
    """Load network weights"""
    h5f = h5py.File(fname, mode="r")
    for key, value in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[key]))
        value.copy_(param)


def np_to_variable(data: np.ndarray, is_cuda=False):
    """Convert a numpy variable to a tensor"""
    with torch.no_grad():
        variable = torch.tensor(data, dtype=torch.float32)
    if is_cuda:
        variable = variable.cuda()
    return variable
