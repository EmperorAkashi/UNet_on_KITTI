import torch
from UNet_on_KITTI.attention import Attention

"construct Double Convolution block for UNet"
class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size = kernel, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
         )
    def forward(self, x):
        x = self.conv(x)
        return x

"Down Sampling"
class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2,2),
                                 DoubleConv(in_ch, out_ch, kernel))   #how Down class inherits DoubleConv?
        
    def forward(self, x):
        x = self.down(x)
        return x

"Up sampling"   
class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch //2, in_ch //2, kernel_size = 2, stride = 2)
        self.conv = DoubleConv(in_ch, out_ch, 3) 
        
    def forward(self, feature, context):
        x = self.up(feature)
        
        skip = torch.cat([x, context], dim = 1)   #concatenate context&feature map
        up_x = self.conv(skip)          
        return up_x

"output layer with binary feature maps"
class out_layer(torch.nn.Module):
    def __init__(self, in_ch, num_class):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_class, 1) #final layer by 1x1 conv
        self.sigmoid = nn.Sigmoid() #elementwise calculation of sigmoid, each layer (class)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x  

"construct Unet"
class UNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64, 3)
        self.attn1 = Attention(64)
        self.down1 = Down(64, 128, 3)
        self.attn2 = Attention(128)
        self.down2 = Down(128, 256, 3)
        self.attn3 = Attention(256)
        self.down3 = Down(256, 512, 3)
        self.attn4 = Attention(512)
        self.down4 = Down(512, 512, 3)
        self.attn5 = Attention(512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = out_layer(64, num_classes)  #out layer w/ different classes, each channel w/ binary label
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.attn1(x1)
        x2 = self.down1(x1)
        x2 = self.attn2(x2)
        x3 = self.down2(x2)
        x3 = self.attn3(x3)
        x4 = self.down3(x3)
        x4 = self.attn4(x4)
        x5 = self.down4(x4)
        x5 = self.attn5(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x