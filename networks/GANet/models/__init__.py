import torch.nn as nn
from conv_libs.separable_convolutions import FwSC,FDwSC

__convolutions__={
    "convolution":nn.Conv3d,
    "FwSC": FwSC,
    "FDwSC": FDwSC
}