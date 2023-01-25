import torch
import torch.nn as nn

class feed_forward(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel, stride=1, padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size = kernel, stride=1, padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
         )
    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        return x

