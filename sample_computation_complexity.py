from ptflops import get_model_complexity_info
import torch.nn.parallel
import torch.nn as nn

from conv_libs.separable_convolutions import FwSC, FDwSC

def calculate(model):
    return get_model_complexity_info(model, input_res=input_res,
                                     input_constructor=None,
                                     as_strings=True,
                                     print_per_layer_stat=False)  # make it true to see layer wise stats

def relative(baseline, new):
    return f"flops are {float(baseline.split()[0])/float(new.split()[0]):.2f}x less than 3D conv"

# Sample example
in_channel=32
out_channel=4
input_res=(in_channel, 48, 240, 528)  #( 32, 48, 240, 528)
kernel_size= (3,3,3)
padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


print(f"************ For a given sample inputs we have following results ************")

# creating 3D convolution with sample example
model=  nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding,bias=False)
model.to(device)
flops_b, params_b = calculate(model)
print(f"For 3D convolution: flops={flops_b} and params={params_b}")

# creating 3D FwSC with sample example
model=  FwSC(in_channels=in_channel, number_kernels=1, out_channels=out_channel, kernel_size=kernel_size[0],bias=False)
model.to(device)
flops, params = calculate(model)
print(f"For Feature-wise separable convolution (FwSC) has : flops={flops} and params={params}, {relative(flops_b,flops)}")

# creating 3D FDwSC with sample example
model=  FDwSC(in_channels=in_channel, number_kernels=1, out_channels=out_channel, kernel_size=kernel_size[0],bias=False)
model.to(device)
flops, params = calculate(model)
print(f"For Feature and Dispisparity-wise separable convolution (FwSC) has : flops={flops} and params={params}, {relative(flops_b,flops)}")

