from torch import nn, sigmoid
from torch.nn.functional import interpolate
from torch.nn.quantized import FloatFunctional
import pytorch_lightning as pl

class SCPA(nn.Module):
    def __init__(self, chn=64):
        super().__init__()
        self.xd1 = nn.Conv2d(chn,chn//2,1,1,0)
        self.xdd1 = nn.Conv2d(chn,chn//2,1,1,0)
        self.relu = nn.ReLU()
        self.xd31 = nn.Conv2d(chn//2,chn//2,3,1,1)
        self.xd1p = nn.Conv2d(chn//2,chn//2,1,1,0)
        self.xd32 = nn.Conv2d(chn//2,chn//2,3,1,1)
        self.xdd3 = nn.Conv2d(chn//2,chn//2,3,1,1)
        self.x1 = nn.Conv2d(chn,chn,1,1,0)
        self.qf = FloatFunctional()
    
    def forward(self,inp):
        x1 = self.xd1(inp)
        x1 = self.relu(x1)
        x1p = x1
        x1 = self.xd31(x1)
        x1p = self.xd1p(x1p)
        x1p = sigmoid(x1p)
        x1 = self.qf.mul(x1,x1p)
        x1 = self.xd32(x1)
        x1 = self.relu(x1)
        x2 = self.xdd1(inp)
        x2 = self.relu(x2)
        x2 = self.xdd3(x2)
        x2 = self.relu(x2)
        x = self.qf.cat([x1,x2],dim=1)
        x = self.x1(x)
        x = self.qf.add(x,inp)
        return x

class UPA(nn.Module):
    def __init__(self,chn,hchn=False):
        super().__init__()
        if not hchn:
            hchn = chn
        self.shuf = nn.PixelShuffle(4)
        self.xc1 = nn.Conv2d(chn,hchn,3,1,1)
        self.xc2 = nn.Conv2d(hchn,hchn,3,1,1)
        self.relu = nn.ReLU()

    def forward(self, inp):
        x = self.shuf(inp)
        x = self.xc1(x)
        x = self.xc2(x)
        x = self.relu(x)
        return x

class PAN(pl.LightningModule):
    def __init__(self, chn=20, hchn=12):
        super().__init__()
        self.firstconv = nn.Conv2d(1,chn,3,1,1)
        self.scpa1 = nn.Sequential(
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn)
        )
        self.scpa2 = nn.Sequential(
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn)
        )
        self.scpa3 = nn.Sequential(
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn)
        )
        self.scpa4 = nn.Sequential(
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn),
            SCPA(chn)
        )
        self.convup1 = nn.Conv2d(chn,chn*4,1)
        self.convup2 = nn.Conv2d(chn,chn*4,1)
        self.convup3 = nn.Conv2d(chn,chn*4,1)
        self.convup4 = nn.Conv2d(chn,chn*4,1)
        self.upa1 = UPA(chn,hchn)
        self.lastconv = nn.Conv2d(hchn,1,3,1,1)
        self.qf = FloatFunctional()
    
    def forward(self, inp):
        x = self.firstconv(inp)
        x1 = self.scpa1(x)
        x1 = self.convup1(x1)
        x2 = self.scpa2(x)
        x2 = self.convup2(x2)
        x3 = self.scpa3(x)
        x3 = self.convup3(x3)
        x4 = self.scpa4(x)
        x4 = self.convup4(x4)
        x = self.qf.cat([x1,x2,x3,x4], dim=1)
        x = self.upa1(x)
        x = self.lastconv(x)
        return x

from ffmpeg._filters import 