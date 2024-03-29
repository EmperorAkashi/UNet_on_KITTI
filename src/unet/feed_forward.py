import torch
import torch.nn as nn

"construct Double Convolution block for UNet"
class FeedForward(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        # self.conv = nn.Sequential( 
        #     nn.Conv2d(in_ch, out_ch, kernel_size = kernel),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, kernel_size = kernel),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        #  )
        self.relu = nn.ReLU()
    def forward(self, x) -> torch.Tensor:
        x = self.relu(x)
        return x

