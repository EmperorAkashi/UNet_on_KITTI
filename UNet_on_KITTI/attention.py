import torch

def hw_flattern(x):
    return x.view(x.size()[0],x.size()[1],-1)

class Attention(torch.nn.Module):
    def __init__(self, c):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        f = self.conv1(x)   # [bs,c',h,w]
        g = self.conv2(x)   # [bs,c',h,w]
        h = self.conv3(x)   # [bs,c',h,w]

        f = hw_flattern(f)
        f = torch.transpose(f, 1, 2)    # [bs,N,c']
        g = hw_flattern(g)              # [bs,c',N]
        h = hw_flattern(h)              # [bs,c,N]
        h = torch.transpose(h, 1, 2)    # [bs,N,c]

        s = torch.matmul(f,g)           # [bs,N,N]
        beta = nn.functional.softmax(s, dim=-1)

        o = torch.matmul(beta,h)        # [bs,N,c]
        o = torch.transpose(o, 1, 2)
        o = o.view(x.shape)
        x = o + x
        return x