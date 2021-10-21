import torch.nn as nn


class FwSC(nn.Module):
    """
    This class defines Feature-wise seprable convolution (FwSC)
    FwSCs works in two steps: in the first step, the 5D input of
    size (N, C, D, H, W) is split into C cubes, and C kernels of
    size k × k × k are applied to generate C output cubes.
    In the second step, C_out point-wise kernels of size 1 × 1 × 1× C are
    applied to aggregate information across the C cubes.

    """

    def __init__(self, in_channels=3,
                 out_channels=2, kernel_size=(3, 3, 3), stride=1, padding=0, bias=False,  number_kernels=1):
        """

        :param in_channels: Number of input channels
        :param number_kernels: Number of kernels, to map intermediate feature between two steps
        :param out_channels: Number of output channels
        :param kernel_size(tuple): Size of kernel, Default: (3,3,3)
        :param stride (int, optional): Stride. Default: 1
        :param bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        """
        super().__init__()

        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2)  # padding
        self.depth_wise = nn.Conv3d(in_channels, in_channels * number_kernels, kernel_size, stride=stride,
                                    padding=padding, groups=in_channels, bias=False)
        self.point_wise = nn.Conv3d(in_channels * number_kernels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)

        return x


class FDwSC(nn.Module):
    """
    This class implements Feature and Disparity wise separable 3D
    convolution (FDwSC) using Spatial, temporal and pointwise 3D convlutions.
    It takes a 5D (N, C, D, H, W) volume and applies spatial convlution via
    1xkernel[1]xkernel[2] then applies temporal convolution via kernel[0]x1x1
    (also stride is managed according in conv_S and conv_T)
    and then merges the results via point-wise convolutions.
    """

    def __init__(self, in_channels=3,
                 out_channels=2, kernel_size=(3, 3, 3), stride=1, padding=0, bias=False, number_kernels=1):
        """
        :param in_channels: Number of input channels
        :param number_kernels: Number of kernels, to map intermediate feature between two steps
        :param out_channels: Number of output channels
        :param kernel_size(tuple): Size of kernel, Default: (3,3,3)
        :param stride (int, optional): Stride. Default: 1
        :param bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        """

        super().__init__()
        # kernel dimensions are D x H x W
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size, kernel_size)  # if kernel=3 transform it to (3,3,3) for padding

        padding = (0, kernel_size[1] // 2, kernel_size[2] // 2)  # padding, 0 in position 0 because of spatail convolution (1x3x3)

        self.conv_S = nn.Conv3d(in_channels, in_channels * number_kernels,
                                kernel_size=(1, kernel_size[1], kernel_size[2]),
                                stride=(1, stride, stride), padding=padding, groups=in_channels)

        padding = (kernel_size[0] // 2, 0, 0)  # padding, 0 in position 1 and 2 because of temporal convolution (3x3x1)

        self.conv_T = nn.Conv3d(in_channels * number_kernels, in_channels * number_kernels,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=padding, stride=(stride, 1, 1), groups=in_channels)

        self.point_wise = nn.Conv3d(in_channels * number_kernels, out_channels, kernel_size=1)

    def forward(self, x):
        out_conv_S = self.conv_S(x)
        out_conv_T = self.conv_T(out_conv_S)
        out = self.point_wise(out_conv_T)
        return out
