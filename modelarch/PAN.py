from torch import nn, sigmoid
import torch
from torch.nn.functional import interpolate
from torch.nn.quantized import FloatFunctional
import pytorch_lightning as pl

class ConvBlock(nn.Module):
    def __init__(self, in_ch,out_ch,k_size,stride=1,pad=0,dil=1,grp=1,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,k_size,stride,pad,dil,grp,bias)
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.lrelu(x)
        return x

class Branch(nn.Module):
    def __init__(self, chn):
        super().__init__()
        self.c1 = ConvBlock(chn,chn//4,1)
        self.c3 = nn.Conv2d(chn,chn,3,1,1)
        self.c31 = ConvBlock(chn,chn,1)
    
    def forward(self,x):
        x1 = self.c1(x)
        x2 = self.c3(x)
        x2 = self.c31(x2)
        return x1,x2

class Node(nn.Module):
    def __init__(self, chn=64):
        super().__init__()
        self.mb1 = Branch(chn//2)
        self.mb2 = Branch(chn//2)
        self.mb3 = Branch(chn//2)
        self.lastsqueeze = nn.Conv2d(chn//2, chn//8, 3, 1, 1)
        self.catconv = ConvBlock(chn//2,chn//2,1)
        self.xp1 = nn.Conv2d(chn,chn,1)
        self.sig = nn.Sigmoid()
    
    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups,
                channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self,inp):
        b1,b2 = inp.chunk(2, dim=1)
        mbo1, b2 = self.mb1(b2)
        mbo2, b2 = self.mb2(b2)
        mbo3, b2 = self.mb3(b2)
        mbo4 = self.lastsqueeze(b2)
        b2 = torch.cat([mbo1,mbo2,mbo3,mbo4], dim=1)
        b2 = self.catconv(b2)
        x = torch.cat([b1,b2], dim=1)
        x1 = self.xp1(x)
        x1 = self.sig(x1)
        x = torch.mul(x,x1)
        x = self.channel_shuffle(x,2)
        return x

class UPA(nn.Module):
    def __init__(self,chn,hchn=False):
        super().__init__()
        if not hchn:
            hchn = chn
        self.xc1 = nn.Conv2d(chn,hchn,3,1,1)
        self.xc2 = nn.Conv2d(hchn,hchn,3,1,1)
        self.xp1 = nn.Conv2d(hchn,hchn,1)
        self.sig = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, inp):
        x = interpolate(inp,scale_factor=2.0,mode='nearest')
        x = self.xc1(x)
        xp = self.xp1(x)
        xp = self.sig(xp)
        x = torch.mul(x,xp)
        x = self.xc2(x)
        x = self.lrelu(x)
        return x

class PAN(pl.LightningModule):
    def __init__(self, chn=48, hchn=24):
        super().__init__()
        self.firstconv = nn.Conv2d(1,chn,3,1,1)
        self.nodes = nn.Sequential(*[Node(48) for _ in range(16)])
        self.midconv = nn.Conv2d(chn,chn,3,1,1)
        self.upa1 = UPA(chn,hchn)
        self.upa2 = UPA(hchn)
        self.lastconv = nn.Conv2d(hchn,1,3,1,1)
    
    def forward(self, inp):
        x = self.firstconv(inp)
        x = self.nodes(x)
        x = self.midconv(x)
        x = self.upa1(x)
        x = self.upa2(x)
        x = self.lastconv(x)
        return x
